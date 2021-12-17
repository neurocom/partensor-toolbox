#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      10/02/2021
* @author    Technical University of Crete team:
*            Athanasios P. Liavas
*            Paris Karakasis
*            Christos Kolomvakis
*            John Papagiannakos
*            Siaminou Nina
* @author    Neurocom SA team:
*            Christos Tsalidis
*            Georgios Lourakis
*            George Lykoudis
*/
#endif // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      CpdCUDA.hpp
* @details
* Implements the Canonical Polyadic Decomposition(cpd) using both Shared
* memory multithreading via @c OpenMP and GPU parallelism via @c CUDA.
* Make use of @c spdlog library in order to write output in
* a log file in @c "../log". In case of using parallelism
* with mpi, then the functions from @c CpdMpi.hpp will be called.
********************************************************************/

#if !defined(PARTENSOR_CPD_HPP)
#error "CPD_CUDA can only included inside CPD"
#endif /* PARTENSOR_CPD_HPP */

#include "CUDAMTTKRP.hpp"

namespace partensor {

  inline namespace v1 {

    namespace internal {
        /**
         * Includes the implementation of CPDCUDA factorization. Based on the 
         * given parameters one of the four overloaded operators will be called.
         * @tparam Tensor_ The Type of The given @c Tensor to be factorized.
         */
        template <typename Tensor_>
        struct CPD<Tensor_,execution::cuda_policy> : public CPD_Base<Tensor_>
        {
            using          CPD_Base<Tensor_>::TnsSize;
            using          CPD_Base<Tensor_>::lastFactor;
            using typename CPD_Base<Tensor_>::Dimensions;
            using typename CPD_Base<Tensor_>::MatrixArray;
            using typename CPD_Base<Tensor_>::DataType;

            using Options = partensor::Options<Tensor_,execution::cuda_policy,DefaultValues>;
            using Status = partensor::Status<Tensor_, execution::cuda_policy, DefaultValues>;

            // Variables that will be used in cpd implementations. 
            struct Member_Variables {
                // MatrixArray  krao;
                MatrixArray  factor_T_factor;
                MatrixArray  mttkrp;
                MatrixArray  tns_mat;
                MatrixArray  norm_factors;
                MatrixArray  old_factors;
                MatrixArray  true_factors;

                Matrix       cwise_factor_product;
                Matrix       temp_matrix;

                Tensor_      tnsX;

                Dimensions   tnsDims;

                bool         all_orthogonal = true;
                int          weight_factor;

                // CUDA variables
                cublasHandle_t                                             handle;
                // std::array<cublasHandle_t, TnsSize>                        handle;
                std::array<cudaStream_t, MAX_NUM_STREAMS>                  Stream;
                std::array<std::array<double *, MAX_NUM_STREAMS>, TnsSize> CUDA_tns_mat;
                std::array<std::array<double *, MAX_NUM_STREAMS>, TnsSize> CUDA_PartialKRP;
                std::array<std::array<double *, MAX_NUM_STREAMS>, TnsSize> CUDA_MTTKRP;

                Member_Variables() = default;
                Member_Variables(Member_Variables const &) = default;
                Member_Variables(Member_Variables      &&) = default;

                Member_Variables &operator=(Member_Variables const &) = default;
                Member_Variables &operator=(Member_Variables      &&) = default;
            };

            /*
            * In case option variable @c writeToFile is enabled, then, before the end
            * of the algorithm, it writes the resulted factors in files, whose
            * paths are specified before compiling in @ options.final_factors_path.
            *  
            * @param  st  [in] Struct where the returned values of @c Cpd are stored.
            */
            void writeFactorsToFile(Status const &st)
            {
                std::size_t size;
                for(std::size_t i=0; i<TnsSize; ++i)
                { 
                    size = st.factors[i].rows() * st.factors[i].cols(); 
                    partensor::write(st.factors[i],
                                    st.options.final_factors_paths[i],
                                    size);
                }
            }

            /*
            * Compute the cost function value based on the initial factors.
            * 
            * @param  mv  [in]     Struct where ALS variables are stored.
            * @param  st  [in,out] Struct where the returned values of @c Cpd are stored.
            *                      In this case the cost function value is updated.
            */
            void cost_function_init(Member_Variables const &mv,
                                    Status                 &st)
            {
                st.f_value = sqrt( ( mv.tns_mat[lastFactor] - st.factors[lastFactor] * PartialKhatriRao(st.factors, lastFactor).transpose() ).squaredNorm() );
            }

            /*
            * Compute the cost function value at the end of each outer iteration
            * based on the last factor.
            * 
            * @param  mv  [in]     Struct where ALS variables are stored.
            * @param  st  [in,out] Struct where the returned values of @c Cpd are stored.
            *                      In this case the cost function value is updated.
            */
            // void cost_function(Member_Variables const &mv,
            //                 Status                 &st)
            // {
            //     st.f_value = sqrt( ( mv.tns_mat[lastFactor] - st.factors[lastFactor] * mv.krao[lastFactor].transpose() ).squaredNorm() );
            // }

            void cost_function(Member_Variables const &mv,
                               Status                 &st)
            {
                st.f_value = sqrt( st.frob_tns - 2 * (mv.mttkrp[lastFactor].cwiseProduct(st.factors[lastFactor])).sum() + 
                    (mv.cwise_factor_product.cwiseProduct(mv.factor_T_factor[lastFactor])).sum() );
            }
            
            /*
            * Compute the cost function value at the end of each outer iteration
            * based on the last accelerated factor.
            * 
            * @param  mv                [in] Struct where ALS variables are stored.
            * @param  st                [in] Struct where the returned values of @c Cpd are stored.
            * @param  factors           [in] Accelerated factors.
            * @param  factors_T_factors [in] Gramian matrices of factors.
            * 
            * @returns The cost function calculated with the accelerated factors.
            */
            double accel_cost_function(Member_Variables const &mv,
                                    Status           const &st,
                                    MatrixArray      const &factors,
                                    MatrixArray      const &factors_T_factors)
            {
                return sqrt( st.frob_tns + (PartialCwiseProd(factors_T_factors, lastFactor).cwiseProduct(factors_T_factors[lastFactor])).sum()
                    - 2 * (mv.mttkrp[lastFactor].cwiseProduct(factors[lastFactor])).sum() );
            }

            /*
            * Based on each factor's constraint, a different
            * update function is used at every outer iteration.
            * 
            * Computes also factor^T * factor at the end.
            *
            * @param  idx [in]     Factor to be updated.
            * @param  mv  [in]     Struct where ALS variables are stored.
            * @param  st  [in,out] Struct where the returned values of @c Cpd are stored.
            *                      Updates the @c stl array with the factors.
            */
            void update_factor(int              const  idx,
                            Member_Variables       &mv,
                            Status                 &st   )
            {
                // Update factor
                switch ( st.options.constraints[idx] ) 
                {
                    case Constraint::unconstrained:
                    {
                        st.factors[idx].noalias() = mv.mttkrp[idx] * mv.cwise_factor_product.inverse();
                        break;
                    }
                        case Constraint::nonnegativity:
                    {
                        mv.temp_matrix = st.factors[idx];
                        NesterovMNLS(mv.cwise_factor_product, mv.mttkrp[idx], st.options.nesterov_delta_1, 
                                    st.options.nesterov_delta_2, st.factors[idx]);
                        if(st.factors[idx].cwiseAbs().colwise().sum().minCoeff() == 0)
                            st.factors[idx] = 0.9 * st.factors[idx] + 0.1 * mv.temp_matrix;
                        break;
                    }
                    case Constraint::orthogonality:
                    {
                        mv.temp_matrix = mv.mttkrp[idx].transpose() * mv.mttkrp[idx];
                        Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(mv.temp_matrix);
                        mv.temp_matrix.noalias() = (eigensolver.eigenvectors()) 
                                                    * (eigensolver.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal()) 
                                                    * (eigensolver.eigenvectors().transpose());
                        st.factors[idx].noalias() = mv.mttkrp[idx] * mv.temp_matrix;
                        break;
                    }
                    case Constraint::sparsity:
                        break;
                    default: // in case of Constraint::constant
                        break;
                }

                // Compute A^T * A + B^T * B + ...
                mv.factor_T_factor[idx].noalias() = st.factors[idx].transpose() * st.factors[idx];
            }
            
            /*
            * @brief Line Search Acceleration
            * 
            * Performs an acceleration step on the updated factors, and keeps the accelerated factors  
            * when the step succeeds. Otherwise, the acceleration step is ignored.
            * Line Search Acceleration reduces the number of outer iterations in the ALS algorithm.
            * 
            * @note This implementation ONLY, if factors are of @c Matrix type.
            * 
            * @param  mv  [in,out] Struct where ALS variables are stored.
            *                      In case the acceleration step is successful the Gramian 
            *                      matrices of factors are updated.
            * @param  st  [in,out] Struct where the returned values of @c Cpd are stored.
            *                      If the acceleration succeeds updates @c factors 
            *                      and cost function value.  
            * 
            */
            void line_search_accel(Member_Variables &mv,
                                   Status           &st)
            {
                double       f_accel    = 0.0; // Objective Value after the acceleration step
                double const accel_step = pow(st.ao_iter+1,(1.0/(st.options.accel_coeff)));

                MatrixArray  accel_factors;
                MatrixArray  accel_gramians;

                for(std::size_t i=0; i<TnsSize; ++i)
                {
                    accel_factors[i]  = mv.old_factors[i] + accel_step * (st.factors[i] - mv.old_factors[i]); 
                    accel_gramians[i] = accel_factors[i].transpose() * accel_factors[i];
                }

                // mttkrp(mv.tnsDims, accel_factors, mv.tns_mat[lastFactor], lastFactor, get_num_threads(), mv.mttkrp[lastFactor]);
                MallocManaged::nonTransposed::hybrid_batched_mttkrp(mv.tnsDims, accel_factors, mv.tns_mat[lastFactor], lastFactor, get_num_threads(), mv.CUDA_tns_mat[lastFactor], mv.CUDA_PartialKRP[lastFactor], mv.CUDA_MTTKRP[lastFactor], mv.handle, mv.Stream, mv.mttkrp[lastFactor]);
                f_accel = accel_cost_function(mv, st, accel_factors, accel_gramians);
                if (st.f_value > f_accel)
                {
                    st.factors         = accel_factors;
                    mv.factor_T_factor = accel_gramians;
                    st.f_value         = f_accel;
                    Partensor()->Logger()->info("Acceleration Step SUCCEEDED! at iter: {}", st.ao_iter);
                }
                else
                    st.options.accel_fail++;
                
                if (st.options.accel_fail==5)
                {
                    st.options.accel_fail=0;
                    st.options.accel_coeff++;
                }
            }

            /*
            * Sequential implementation of Alternating Least Squares (ALS) method,
            * using Shared Memory and OpenMP.
            * 
            * @param  R   [in]     The rank of decomposition.
            * @param  mv  [in]     Struct where ALS variables are stored and being updated
            *                      until a termination condition is true.
            * @param  st  [in,out] Struct where the returned values of @c Cpd are stored.
            */
            void als(std::size_t      const  R,
                     Member_Variables       &mv,
                     Status                 &status)
            {
                // Initialize CUDA variables
                cublasCreate(&mv.handle);

                for (int str_id = 0; str_id < MAX_NUM_STREAMS; str_id++)
                {
                  cudaStreamCreate(&mv.Stream[str_id]);
                }
                
                for (std::size_t i=0; i<TnsSize; i++)
                {
                  std::size_t first_mode = (i == 0) ? 1 : 0;

                  for(int str_id = 0; str_id < MAX_NUM_STREAMS; str_id++)
                  {
                    cudaMallocManaged((void **)&mv.CUDA_tns_mat[i][str_id],    mv.tnsDims[i] * mv.tnsDims[first_mode] * sizeof(double)); // unified mem. for Matr. Tensor X,
                    cudaMallocManaged((void **)&mv.CUDA_PartialKRP[i][str_id], mv.tnsDims[first_mode] *         R     * sizeof(double)); // unified mem. for Partial KRP,
                    cudaMallocManaged((void **)&mv.CUDA_MTTKRP[i][str_id],     mv.tnsDims[i] *                  R     * sizeof(double)); // unified mem. for MTTKRP.
                  }
                  cudaDeviceSynchronize();
                }
                // std::cout << "cuda initialization: ok" << std::endl;
                for (std::size_t i=0; i<TnsSize; i++)
                {
                    mv.factor_T_factor[i].noalias() = status.factors[i].transpose() * status.factors[i];
                    mv.tns_mat[i]                   = Matricization(mv.tnsX, i);
                }

                if(status.options.normalization)
                {
                  choose_normilization_factor(status, mv.all_orthogonal, mv.weight_factor);
                }
                
                // Normalize(static_cast<int>(R), mv.factor_T_factor, status.factors);
                // status.frob_tns         = square_norm(mv.tnsX);
                std::array<int, TnsSize>  newTnsDims = {1};
                mv.tnsX.resize(newTnsDims);
                cost_function_init(mv, status);
                status.frob_tns = mv.tns_mat[0].squaredNorm();
                status.rel_costFunction = status.f_value/sqrt(status.frob_tns);
                
                // ---- Loop until ALS converges ----
                partensor::timer.startChronoHighTimer();
                partensor::Timers mttkrp_timer_tuc;
                double end_mttkrp_time_tuc = 0;
                while (1)
                {
                    status.ao_iter++;
                    Partensor()->Logger()->info("iter: {} -- fvalue: {} -- relative_costFunction: {}", status.ao_iter, 
                                                                status.f_value, status.rel_costFunction);
                    
                    for (std::size_t i=0; i<TnsSize; i++)
                    {
                        MallocManaged::nonTransposed::hybrid_batched_mttkrp(mv.tnsDims, status.factors, mv.tns_mat[i], i, get_num_threads(), mv.CUDA_tns_mat[i], mv.CUDA_PartialKRP[i], mv.CUDA_MTTKRP[i], mv.handle, mv.Stream, mv.mttkrp[i]);

                        mv.cwise_factor_product = PartialCwiseProd(mv.factor_T_factor, i);

                        // Update factor
                        update_factor(i, mv, status);

                        // Cost function Computation
                        if(i == lastFactor)
                          cost_function(mv, status);
                    }
                    status.rel_costFunction = status.f_value/sqrt(status.frob_tns);
                    if(status.options.normalization && !mv.all_orthogonal)
                        Normalize(mv.weight_factor, static_cast<int>(R), mv.factor_T_factor, status.factors);

                    // ---- Terminating condition ----
                    if (status.rel_costFunction < status.options.threshold_error || status.ao_iter >= status.options.max_iter)
                    {
                        if(status.options.writeToFile)
                            writeFactorsToFile(status);  
                        break;
                    }

                    if (status.options.acceleration)
                    {
                        mv.norm_factors = status.factors;
                        // ---- Acceleration Step ----
                        if (status.ao_iter > 1)
                            line_search_accel(mv, status);

                        mv.old_factors = mv.norm_factors;
                    }  
                } // end of while

                for (int str_id = 0; str_id < MAX_NUM_STREAMS; str_id++)
                {
                  cudaStreamDestroy(mv.Stream[str_id]);
                }
                
                for (std::size_t i=0; i<TnsSize; i++)
                {
                  std::size_t first_mode = (i == 0) ? 1 : 0;
                  for(int str_id = 0; str_id < MAX_NUM_STREAMS; str_id++)
                  {
                    cudaFree(mv.CUDA_tns_mat[i][str_id]);
                    cudaFree(mv.CUDA_PartialKRP[i][str_id]);
                    cudaFree(mv.CUDA_MTTKRP[i][str_id]);
                  }
                }
                cublasDestroy(mv.handle);
            }

            /**
             * Implementation of CP Decomposition with default values in Options struct
             * and randomly generated initial factors and using OpenMP.
             * 
             * @param  tnsX    [in] The given Tensor to be factorized of @c Tensor_ type, 
             *                      with @c double data.
             * @param  R       [in] The rank of decomposition.
             *
             * @returns An object of type @c Status with the results of the algorithm.
             */
            Status operator()(Tensor_     const &tnsX, 
                              std::size_t const  R)
            {
              Options          options = MakeOptions<Tensor_,execution::cuda_policy,DefaultValues>();
              Status           status(options);
              Member_Variables mv;
              
              // extract dimensions from tensor
              mv.tnsDims = tnsX.dimensions();
              // produce estimate factors using uniform distribution with entries in [0,1].
              makeFactors(mv.tnsDims, status.options.constraints, R, status.factors);
              mv.tnsX = tnsX;
              als(R, mv, status);

              return status;
            }

            /**
             * Implementation of CP Decomposition with user's changed values in Options struct,
             * but with randomly generated initial factors and using OpenMP.
             * 
             * @param  tnsX    [in] The given Tensor to be factorized of @c Tensor_ type, 
             *                      with @c double data.
             * @param  R       [in] The rank of decomposition.
             * @param  options [in] User's @c options, other than the default. It must be of
             *                      @c partensor::Options<partensor::Tensor<order>> type,
             *                      where @c order must be in range of @c [3-8].
             *
             * @returns An object of type @c Status with the results of the algorithm.
             */
            Status operator()(Tensor_     const &tnsX, 
                              std::size_t const  R, 
                              Options     const &options)
            {
              Status           status(options);
              Member_Variables mv;
              
              // extract dimensions from tensor
              mv.tnsDims = tnsX.dimensions();
              // produce estimate factors using uniform distribution with entries in [0,1].
              makeFactors(mv.tnsDims, status.options.constraints, R, status.factors);
              mv.tnsX = tnsX;

              switch ( status.options.method )
              {
                case Method::als:
                {
                  als(R, mv, status);
                  break;
                }
                case Method::rnd:
                  break;
                case Method::bc:
                  break;
                default:
                  break;
              }

              return status;
            };

            /**
             * Implementation of CP Decomposition with default values in Options
             * struct, but with initialized factors and using OpenMP.
             * 
             * @param  tnsX        [in] The given Tensor to be factorized of @c Tensor_ type, 
             *                          with @c double data.
             * @param  R           [in] The rank of decomposition.
             * @param  factorsInit [in] Uses initialized factors instead of randomly generated. The 
             *                          data must be of @c partensor::Matrix type and stored in an
             *                          @c stl array with size same as the @c order of @c tnsX.
             *
             * @returns An object of type @c Status with the results of the algorithm.
             */
            Status operator()(Tensor_     const &tnsX, 
                              std::size_t const  R, 
                              MatrixArray const &factorsInit)
            {
              Options          options = MakeOptions<Tensor_,execution::cuda_policy,DefaultValues>();
              Status           status(options);
              Member_Variables mv;

              status.factors = factorsInit;
              mv.tnsX        = tnsX;
              // extract dimensions from tensor
              mv.tnsDims     = tnsX.dimensions();
              als(R, mv, status);

              return status;
            }

            /**
             * Implementation of CP Decomposition with user's changed values in Options struct,
             * and also initialized factors and using OpenMP.
             * 
             * @param  tnsX        [in] The given Tensor to be factorized of @c Tensor_ type, 
             *                          with @c double data.
             * @param  R           [in] The rank of decomposition.
             * @param  options     [in] User's @c options, other than the default. It must be of
             *                          @c partensor::Options<partensor::Tensor<order>> type,
             *                          where @c order must be in range of @c [3-8].
             * @param  factorsInit [in] Uses initialized factors instead of randomly generated. The 
             *                          data must be of @c partensor::Matrix type and stored in an
             *                          @c stl array with size same as the @c order of @c tnsX.
             *
             * @returns An object of type @c Status with the results of the algorithm.
             */
            Status operator()(Tensor_     const &tnsX, 
                              std::size_t const R, 
                              Options     const &options, 
                              MatrixArray const &factorsInit)
            {
              Status status(options);
              Member_Variables mv;

              status.factors = factorsInit;
              mv.tnsX        = tnsX;
              // extract dimensions from tensor
              mv.tnsDims     = tnsX.dimensions();

              switch ( status.options.method )
              {
                case Method::als:
                {
                  als(R, mv, status);
                  break;
                }
                case Method::rnd:
                  break;
                case Method::bc:
                  break;
                default:
                  break;
              }

              return status;
            }

            /**
             * Implementation of CP Decomposition with default values in Options Struct
             * and randomly generated initial factors and using OpenMP. In this implementation 
             * the Tensor can be read from a file, given the @c path where the Tensor is located.
             * 
             * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
             *                         length must be same as the Tensor order.
             * @param  R          [in] The rank of decomposition.
             * @param  path       [in] The path where the tensor is located.
             *
             * @returns An object of type @c Status with the results of the algorithm.
             */
            Status operator()(std::array<int, TnsSize> const &tnsDims, 
                              std::size_t              const  R, 
                              std::string              const &path)
            {
              Options          options = MakeOptions<Tensor_,execution::cuda_policy,DefaultValues>();
              Status           status(options);
              Member_Variables mv;

              long long int fileSize = 1;
              for(auto &dim : tnsDims) 
                fileSize *= static_cast<long long int>(dim);

              mv.tnsX.resize(tnsDims);
              // Read the whole Tensor from a file
              read( path, 
                    fileSize, 
                    0, 
                    mv.tnsX   );
              // produce estimate factors using uniform distribution with entries in [0,1].
              makeFactors(tnsDims, status.options.constraints, R, status.factors);
              // mv.tnsDims = tnsDims;
              std::copy(tnsDims.begin(),tnsDims.end(),mv.tnsDims.begin());
              als(R, mv, status);

              return status;
            }

            /**
             * Implementation of CP Decomposition with user's changed values in Options struct
             * and randomly generated initial factors and using OpenMP. In this implementation 
             * the Tensor can be read from a file, given the @c path where the Tensor is located.
             * 
             * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
             *                         length must be same as the Tensor order.      
             * @param  R          [in] The rank of decomposition.
             * @param  path       [in] The path where the tensor is located.
             * @param  options    [in] User's @c options, other than the default. It must be of
             *                         @c partensor::Options<partensor::Tensor<order>> type,
             *                         where @c order must be in range of @c [3-8].
             *
             * @returns An object of type @c Status with the results of the algorithm.
             */
            Status operator()(std::array<int, TnsSize> const &tnsDims, 
                              std::size_t              const  R, 
                              std::string              const &path, 
                              Options                  const &options)
            {
              Status           status(options);
              Member_Variables mv;
              
              long long int fileSize = 1;
              for(auto &dim : tnsDims) 
                fileSize *= static_cast<long long int>(dim);

              mv.tnsX.resize(tnsDims);
              // Read the whole Tensor from a file
              read( path, 
                    fileSize, 
                    0, 
                    mv.tnsX   );
              // produce estimate factors using uniform distribution with entries in [0,1].
              makeFactors(tnsDims, status.options.constraints, R, status.factors);
              std::copy(tnsDims.begin(),tnsDims.end(),mv.tnsDims.begin());

              switch ( status.options.method )
              {
                case Method::als:
                {
                  als(R, mv, status);
                  break;
                }
                case Method::rnd:
                  break;
                case Method::bc:
                  break;
                default:
                  break;
              }

              return status;
            }

            /**
             * Implementation of CP Decomposition with default values in Options Struct
             * and initialized factors and using OpenMP. In this implementation the Tensor 
             * and the factors can be read from a file, given the @c paths to the location 
             * in the disk, where the Tensor is stored.
             * 
             * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
             *                         length must be same as the Tensor order.     
             * @param  R          [in] The rank of decomposition.
             * @param  paths      [in] An @c stl array containing paths for the Tensor to be 
             *                         factorized and after that the paths for the initialized 
             *                         factors. 
             * @returns An object of type @c Status with the results of the algorithm.
             */
            Status operator()(std::array<int, TnsSize>          const &tnsDims, 
                              std::size_t                       const  R, 
                              std::array<std::string,TnsSize+1> const &paths)
            {
              Options          options = MakeOptions<Tensor_,execution::cuda_policy,DefaultValues>();
              Status           status(options);
              Member_Variables mv;

              long long int fileSize = 1;
              for(auto &dim : tnsDims) 
                fileSize *= static_cast<long long int>(dim);

              mv.tnsX.resize(tnsDims);
              // Read the whole Tensor from a file
              read( paths.front(), 
                    fileSize, 
                    0, 
                    mv.tnsX   );
              
              // Read initialized factors from files
              for(std::size_t i=0; i<TnsSize; ++i)
              {   
                status.factors[i] = Matrix(tnsDims[i],static_cast<int>(R));
                read( paths[i+1], 
                      tnsDims[i]*R,
                      0, 
                      status.factors[i] );    
              }      
              std::copy(tnsDims.begin(),tnsDims.end(),mv.tnsDims.begin());

              als(R, mv, status);

              return status;
            }

            /**
             * Implementation of CP Decomposition with user's changed values in Options struct
             * and initialized factors and using OpenMP. In this implementation the Tensor and 
             * the factors can be read from a file, given the @c paths to the location in the disk, 
             * where the Tensor is stored.
             * 
             * With this version of @c cpd the Tensor can be read from a file, specified in 
             * @c path variable.
             * 
             * @param  tnsDims     [in] @c Stl array containing the Tensor dimensions, whose
             *                         length must be same as the Tensor order.     
             * @param  R           [in] The rank of decomposition.   
             * @param  paths       [in] An @c stl array containing paths for the Tensor to be 
             *                          factorized and after that the paths for the initialized 
             *                          factors. 
             * @param  options     [in] User's @c options, other than the default. It must be of
             *                          @c partensor::Options<partensor::Tensor<order>> type,
             *                          where @c order must be in range of @c [3-8].
             * 
             * @returns An object of type @c Status with the results of the algorithm.
             */
            Status operator()(std::array<int, TnsSize>           const &tnsDims, 
                              std::size_t                        const  R, 
                              std::array<std::string, TnsSize+1> const &paths, 
                              Options                            const &options)
            {
              Status status(options);
              Member_Variables mv;

              long long int fileSize = 1;
              for(auto &dim : tnsDims) 
                fileSize *= static_cast<long long int>(dim);

              mv.tnsX.resize(tnsDims);
              // Read the whole Tensor from a file
              read( paths.front(), 
                    fileSize, 
                    0, 
                    mv.tnsX   );

              // Read initialized factors from files
              for(std::size_t i=0; i<TnsSize; ++i)
              {    
                status.factors[i] = Matrix(tnsDims[i],static_cast<int>(R));
                read( paths[i+1], 
                      tnsDims[i]*R,
                      0, 
                      status.factors[i] );   
              }  
              std::copy(tnsDims.begin(),tnsDims.end(),mv.tnsDims.begin());

              switch ( status.options.method )
              {
                case Method::als:
                {
                  als(R, mv, status);
                  break;
                }
                case Method::rnd:
                  break;
                case Method::bc:
                  break;
                default:
                  break;
              }

              return status;
            }

            /**
             * Implementation of CP Decomposition with user's changed values in Options struct
             * and no initialized factors and using OpenMP. In this implementation the TRUE factors 
             * can be read from files. The Tensor is computed internally. Also, initialized factors 
             * can be read from a file, given the @c paths to the location in the disk, where they  
             * are stored.
             * 
             * With this version of @c cpd the true factors, that their outer product produce the 
             * Tensor and the initialized points-factors can be read from files. 
             * 
             * @param  tnsDims     [in] @c Stl array containing the Tensor dimensions, whose
             *                         length must be same as the Tensor order.  
             * @param  R           [in] The rank of decomposition.
             * @param  true_paths  [in] An @c stl array containing paths for the true factors.    
             * @param  init_paths  [in] An @c stl array containing paths for initialized 
             *                          factors. 
             * @param  options     [in] User's @c options, other than the default. It must be of
             *                          @c partensor::Options<partensor::Tensor<order>> type,
             *                          where @c order must be in range of @c [3-8].
             * 
             * @returns An object of type @c Status with the results of the algorithm.
             */
            Status operator()(std::array<int, TnsSize>         const &tnsDims, 
                              std::size_t                      const  R, 
                              std::array<std::string, TnsSize> const &true_paths, 
                              std::array<std::string, TnsSize> const &init_paths, 
                              Options                          const &options)
            {
              Status status(options);
              Member_Variables mv;

              for(std::size_t i=0; i<TnsSize; ++i)
              {    
                mv.true_factors[i] = Matrix(tnsDims[i],static_cast<int>(R));
                status.factors[i]  = Matrix(tnsDims[i],static_cast<int>(R));

                // Read initialized factors from files
                read( true_paths[i], 
                      tnsDims[i]*R,
                      0, 
                      mv.true_factors[i] );  
                // Read initialized factors from files
                read( init_paths[i], 
                      tnsDims[i]*R,
                      0, 
                      status.factors[i] );    
              }  
              std::copy(tnsDims.begin(),tnsDims.end(),mv.tnsDims.begin());

              switch ( status.options.method )
              {
                case Method::als:
                {
                  als_true_factors(R, mv, status);
                  break;
                }
                case Method::rnd:
                  break;
                case Method::bc:
                  break;
                default:
                  break;
              }

              return status;
            }
      };
    } // end namespace internal
  } // end namespace v1
} // end namespace partensor

