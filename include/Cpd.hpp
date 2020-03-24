#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      07/10/2019
* @author    Christos Tsalidis
* @author    Yiorgos Lourakis
* @author    George Lykoudis
* @copyright 2019 Neurocom All Rights Reserved.
*/
#endif // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      Cpd.hpp
* @details
* Implements the Canonical Polyadic Decomposition(cpd).
* Make use of @c spdlog library in order to write output in
* a log file in @c "../log". In case of using parallelism
* with mpi, then the functions from @c CpdMpi.hpp will be called.
********************************************************************/

#ifndef PARTENSOR_CPD_HPP
#define PARTENSOR_CPD_HPP

#include "PARTENSOR_basic.hpp"
#include "Matricization.hpp"
#include "PartialCwiseProd.hpp"
#include "MTTKRP.hpp"
#include "NesterovMNLS.hpp"
#include "Normalize.hpp"
#include "Timers.hpp"
#include "ReadWrite.hpp"

namespace partensor
{
  inline namespace v1
  {
    namespace internal
    {
      //template <typename ExecutionPolicy, typename Tensor>
      //execution::internal::enable_if_execution_policy<ExecutionPolicy,Tensor>
      //Status cpd_f(ExecutionPolicy &&, Tensor const &tnsX, std::size_t rank);

      /*
       * Includes the implementation of CP Decomposition. Based on the given
       * parameters one of the overloaded operators will be called.
       */
      template <typename Tensor_>
      struct CPD_Base {
        static constexpr std::size_t TnsSize    = TensorTraits<Tensor_>::TnsSize; /**< Tensor Order. */
        static constexpr std::size_t lastFactor = TnsSize - 1;                    /**< ID of the last factor. */

        using DataType   = typename TensorTraits<Tensor_>::DataType;              /**< Tensor Data type. */
        using MatrixType = typename TensorTraits<Tensor_>::MatrixType;            /**< Eigen Matrix with the same Data type with the Tensor. */
        using Dimensions = typename TensorTraits<Tensor_>::Dimensions;            /**< Tensor Dimensions type. */

        using Constraints = std::array<Constraint, TnsSize>;                      /**< Stl array of size TnsSize and containing Constraint type. */
        using MatrixArray = std::array<MatrixType, TnsSize>;                      /**< Stl array of size TnsSize and containing MatrixType type. */
        using DoubleArray = std::array<double, TnsSize>;                          /**< Stl array of size TnsSize and containing double type. */
      };

      template <typename Tensor_, typename ExecutionPolicy = execution::sequenced_policy>
      struct CPD : public CPD_Base<Tensor_>
      {
        using          CPD_Base<Tensor_>::TnsSize;
        using          CPD_Base<Tensor_>::lastFactor;
        using typename CPD_Base<Tensor_>::Dimensions;
        using typename CPD_Base<Tensor_>::MatrixArray;
        using typename CPD_Base<Tensor_>::DataType;
        
        using Options = partensor::Options<Tensor_,execution::sequenced_policy,DefaultValues>;
        using Status  = partensor::Status<Tensor_,execution::sequenced_policy,DefaultValues>;

        // Variables that will be used in cpd implementations. 
        struct Member_Variables {
          MatrixArray  krao;
          MatrixArray  factor_T_factor;
          MatrixArray  mttkrp;
          MatrixArray  tns_mat;
          MatrixArray  norm_factors;
          MatrixArray  old_factors;
          MatrixArray  true_factors;

          Matrix       cwise_factor_product;
          Matrix       tnsX_mat_lastFactor_T;
          Matrix       temp_matrix;

          Tensor_      tnsX;

          bool         all_orthogonal = true;
          int          weight_factor;

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
        void cost_function(Member_Variables const &mv,
                           Status                 &st)
        {
          st.f_value = sqrt( ( mv.tns_mat[lastFactor] - st.factors[lastFactor] * mv.krao[lastFactor].transpose() ).squaredNorm() );
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
          return sqrt( st.frob_tns + (PartialCwiseProd(factors_T_factors, lastFactor) * factors_T_factors[lastFactor]).trace()
                 - 2 * ((PartialKhatriRao(factors, lastFactor).transpose() * mv.tnsX_mat_lastFactor_T) * factors[lastFactor]).trace() );
        }

        void cost_function2(Member_Variables const &mv,
                            Status                 &st)
        {
          st.f_value = sqrt( st.frob_tns -2 * (mv.mttkrp.cwiseProduct(st.factors[lastFactor])).sum() + 
                 (mv.cwise_factor_product.cwiseProduct(mv.factor_T_factor[lastFactor])).sum() );
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
         * Sequential implementation of Alternating Least Squares (ALS) method.
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
          for (std::size_t i=0; i<TnsSize; i++)
          {
            mv.factor_T_factor[i].noalias() = status.factors[i].transpose() * status.factors[i];
            mv.tns_mat[i]                   = Matricization(mv.tnsX, i);
          }

          if(status.options.acceleration)
          {
            mv.tnsX_mat_lastFactor_T = mv.tns_mat[lastFactor].transpose();
          }

          if(status.options.normalization)
          {
            choose_normilization_factor(status, mv.all_orthogonal, mv.weight_factor);
          }
          
          // Normalize(static_cast<int>(R), mv.factor_T_factor, status.factors);
          status.frob_tns         = square_norm(mv.tnsX);
          cost_function_init(mv, status);
          status.rel_costFunction = status.f_value/sqrt(status.frob_tns);
          
          // ---- Loop until ALS converges ----
          while(1)
          {
            status.ao_iter++;
            Partensor()->Logger()->info("iter: {} -- fvalue: {} -- relative_costFunction: {}", status.ao_iter, 
                                                         status.f_value, status.rel_costFunction);
            
            for (std::size_t i=0; i<TnsSize; i++)
            {
              mttkrp(status.factors, mv.tns_mat[i], i, mv.krao[i], mv.mttkrp[i]);
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
        }

        /*
         * Sequential implementation of Alternating Least Squares (ALS) method.
         * Make use of true factors read from files instead of the Tensor.
         * 
         * @param  R   [in]     The rank of decomposition.
         * @param  mv  [in]     Struct where ALS variables are stored and being updated
         *                      until a termination condition is true.
         * @param  st  [in,out] Struct where the returned values of @c Cpd are stored.
         */
        void als_true_factors( std::size_t      const  R,
                               Member_Variables       &mv,
                               Status                 &status )
        {
          for (std::size_t i=0; i<TnsSize; i++)
          {
            mv.factor_T_factor[i].noalias() = status.factors[i].transpose() * status.factors[i];
            mv.tns_mat[i]                   = generateTensor(i, mv.true_factors);
          }

          if(status.options.acceleration)
          {
            mv.tnsX_mat_lastFactor_T = mv.tns_mat[lastFactor].transpose();
          }

          if(status.options.normalization)
          {
            choose_normilization_factor(status, mv.all_orthogonal, mv.weight_factor);
          }
          
          // Normalize(static_cast<int>(R), mv.factor_T_factor, status.factors);
          status.frob_tns         = (mv.tns_mat[lastFactor]).squaredNorm();
          cost_function_init(mv, status);
          status.rel_costFunction = status.f_value/sqrt(status.frob_tns);
          
          // ---- Loop until ALS converges ----
          while(1)
          {
            status.ao_iter++;
            Partensor()->Logger()->info("iter: {} -- fvalue: {} -- relative_costFunction: {}", status.ao_iter, 
                                                         status.f_value, status.rel_costFunction);
            
            for (std::size_t i=0; i<TnsSize; i++)
            {
              mv.krao[i]              = PartialKhatriRao(status.factors,  i);
              mv.cwise_factor_product = PartialCwiseProd(mv.factor_T_factor, i);
              mv.mttkrp[i].noalias()  = mv.tns_mat[i] * mv.krao[i];

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
        }

        /**
         * Implementation of CP Decomposition with default values in Options struct
         * and randomly generated initial factors.
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
          Status           status = MakeStatus<Tensor_>();
          Member_Variables mv;

          // extract dimensions from tensor
          Dimensions const &tnsDims = tnsX.dimensions();
          // produce estimate factors using uniform distribution with entries in [0,1].
          makeFactors(tnsDims, status.options.constraints, R, status.factors);
          mv.tnsX = tnsX;
          als(R, mv, status);

          return status;
        }

        /**
         * Implementation of CP Decomposition with user's changed values in Options struct,
         * but with randomly generated initial factors.
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
          Dimensions const &tnsDims = tnsX.dimensions();
          // produce estimate factors using uniform distribution with entries in [0,1].
          makeFactors(tnsDims, status.options.constraints, R, status.factors);
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
         * struct, but with initialized factors.
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
          Status status = MakeStatus<Tensor_>();
          Member_Variables mv;

          status.factors = factorsInit;
          mv.tnsX        = tnsX;
          als(R, mv, status);

          return status;
        }

        /**
         * Implementation of CP Decomposition with user's changed values in Options struct,
         * and also initialized factors.
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
         * and randomly generated initial factors. In this implementation the Tensor 
         * can be read from a file, given the @c path where the Tensor is located.
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
          using  Tensor = Tensor<static_cast<int>(TnsSize)>;

          Status           status = MakeStatus<Tensor>();
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
          als(R, mv, status);

          return status;
        }

        /**
         * Implementation of CP Decomposition with user's changed values in Options struct
         * and randomly generated initial factors. In this implementation the Tensor can 
         * be read from a file, given the @c path where the Tensor is located.
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
         * and initialized factors. In this implementation the Tensor and the factors can 
         * be read from a file, given the @c paths where the Tensor and the initialized 
         * factors are located.
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
          using  Tensor = Tensor<static_cast<int>(TnsSize)>;
          Status status = MakeStatus<Tensor>();
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

          als(R, mv, status);

          return status;
        }

        /**
         * Implementation of CP Decomposition with user's changed values in Options struct
         * and initialized factors. In this implementation the Tensor and the factors can 
         * be read from a file, given the @c paths where the Tensor and the initialized 
         * factors are located.
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
         * and no initialized factors. In this implementation the TRUE and initialized factors 
         * can be read from files, given the @c paths where the factors are located.
         * The Tensor is computed internally.
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
    }  // namespace internal
  }    // namespace v1
} // end namespace partensor

#if USE_MPI

#include "CpdMpi.hpp"
#endif /* USE_MPI */

#if USE_OPENMP

#include "CpdOpenMP.hpp"
#endif /* USE_OPENMP */

namespace partensor
{
  /**
   * Interface of Canonical Polyadic Decomposition(cpd), with the use of an 
   * @c Execution @c Policy, which can be either @c sequential, @c parallel 
   * with the use of @c MPI, or @c parallel with the use of @c OpenMP. 
   * In order to choose a policy, type @c execution::seq, @c execution::mpi or 
   * @c execution::omp. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * @tparam ExecutionPolicy Type of @c stl @c Execution @c Policy 
   *                         (sequential, parallel-mpi).
   * @tparam Tensor_         Type(data type and order) of input Tensor.
   *                         @c Tensor_ must be @c partensor::Tensor<order>, where 
   *                         @c order must be in range of @c [3-8].
   * @param  tnsX    [in]    The given Tensor to be factorized of @c Tensor_ type, 
   *                         with @c double data.
   * @param  R       [in]    The rank of decomposition.
   *
   * @returns An object of type @c Status, containing the results of the algorithm.
   */
  template <typename Tensor_, typename ExecutionPolicy>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpd( ExecutionPolicy       &&, 
       Tensor_         const &tnsX, 
       std::size_t     const  R )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD<Tensor_>()(tnsX,R);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD<Tensor_,execution::openmpi_policy>()(tnsX,R);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::CPD<Tensor_,execution::openmp_policy>()(tnsX,R);
    }
    else
      return internal::CPD<Tensor_>()(tnsX,R);
  }
  
  /*
   * Interface of Canonical Polyadic Decomposition(cpd). Sequential Policy.
   * 
   * @tparam Tensor_      Type(data type and order) of input Tensor.
   *                      @c Tensor_ must be @c partensor::Tensor<order>, where 
   *                      @c order must be in range of @c [3-8].
   * @param  tnsX    [in] The given Tensor to be factorized of @c Tensor_ type, 
   *                      with @c double data.
   * @param  R       [in] The rank of decomposition.
   *
   * @returns An object of type @c Status, containing the results of the algorithm.
   */
  template<typename Tensor_>
  auto cpd(Tensor_     const &tnsX, 
           std::size_t const  R)
  {
    return internal::CPD<Tensor_,execution::sequenced_policy>()(tnsX,R);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd), with the use of an 
   * @c Execution @c Policy, which can be either @c sequential, @c parallel 
   * with the use of @c MPI, or @c parallel with the use of @c OpenMP. 
   * In order to choose a policy, type @c execution::seq, @c execution::mpi or 
   * @c execution::omp.
   * Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * @tparam ExecutionPolicy Type of @c stl @c Execution @c Policy 
   *                         (sequential, parallel-mpi). 
   * @tparam Tensor_         Type(data type and order) of input Tensor.
   *                         @c Tensor_ must be @c partensor::Tensor<order>, where 
   *                         @c order must be in range of @c [3-8].
   * @param  tnsX    [in]    The given Tensor to be factorized of @c Tensor_ type, 
   *                         with @c double data.
   * @param  R       [in]    The rank of decomposition.
   * @param  options [in]    User's @c options, other than the default. It must be of
   *                         @c partensor::Options<partensor::Tensor<order>> type,
   *                         where @c order must be in range of @c [3-8].
   * 
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template <typename Tensor_, typename ExecutionPolicy>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpd( ExecutionPolicy       &&, 
       Tensor_         const &tnsX, 
       std::size_t     const  R, 
       Options<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues> const &options )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD<Tensor_>()(tnsX,R,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD<Tensor_,execution::openmpi_policy>()(tnsX,R,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::CPD<Tensor_,execution::openmp_policy>()(tnsX,R,options);
    }
    else
      return internal::CPD<Tensor_>()(tnsX,R,options);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd). Sequential Policy.
   * 
   * @tparam Tensor_      Type(data type and order) of input Tensor.
   *                      @c Tensor_ must be @c partensor::Tensor<order>, where 
   *                      @c order must be in range of @c [3-8].
   * @param  tnsX    [in] The given Tensor to be factorized of @c Tensor_ type, 
   *                      with @c double data.
   * @param  R       [in] The rank of decomposition.
   * @param  options [in] User's @c options, other than the default. It must be of
   *                      @c partensor::Options<partensor::Tensor<order>> type,
   *                      where @c order must be in range of @c [3-8].
   * 
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template<typename Tensor_>
  auto cpd(Tensor_          const &tnsX, 
           std::size_t      const  R, 
           Options<Tensor_> const &options)
  {
    return internal::CPD<Tensor_,execution::sequenced_policy>()(tnsX,R,options);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd), with the use of an 
   * @c Execution @c Policy, which can be either @c sequential, @c parallel 
   * with the use of @c MPI, or @c parallel with the use of @c OpenMP. 
   * In order to choose a policy, type @c execution::seq, @c execution::mpi or 
   * @c execution::omp. 
   * Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * @tparam ExecutionPolicy  Type of @c stl @c Execution @c Policy 
   *                          (sequential, parallel-mpi). 
   * @tparam Tensor_          Type(data type and order) of input Tensor.
   *                          @c Tensor_ must be @c partensor::Tensor<order>, where 
   *                          @c order must be in range of @c [3-8].
   * @param  tnsX        [in] The given Tensor to be factorized of @c Tensor_ type, 
   *                          with @c double data.
   * @param  R           [in] The rank of decomposition.
   * @param  factorsInit [in] Uses initialized factors instead of randomly generated. The 
   *                          data must be of @c partensor::Matrix type and stored in an
   *                          @c stl array with size same as the @c order of @c tnsX.
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template <typename ExecutionPolicy, typename Tensor_>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpd( ExecutionPolicy            &&, 
       Tensor_              const &tnsX, 
       std::size_t          const  R, 
       MatrixArray<Tensor_> const &factorsInit )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD<Tensor_>()(tnsX,R,factorsInit);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD<Tensor_,execution::openmpi_policy>()(tnsX,R,factorsInit);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::CPD<Tensor_,execution::openmp_policy>()(tnsX,R,factorsInit);
    }
    else
      return internal::CPD<Tensor_>()(tnsX,R,factorsInit);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd). Sequential Policy.
   * 
   * @tparam Tensor_          Type(data type and order) of input Tensor.
   *                          @c Tensor_ must be @c partensor::Tensor<order>, where 
   *                          @c order must be in range of @c [3-8].
   * @param  tnsX        [in] The given Tensor to be factorized of @c Tensor_ type, 
   *                          with @c double data.
   * @param  R           [in] The rank of decomposition.
   * @param  factorsInit [in] Uses initialized factors instead of randomly generated. The 
   *                          data must be of @c partensor::Matrix type and stored in an
   *                          @c stl array with size same as the @c order of @c tnsX.
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template<typename Tensor_>
  auto cpd(Tensor_              const &tnsX, 
           std::size_t          const  R, 
           MatrixArray<Tensor_> const &factorsInit)
  {
    return internal::CPD<Tensor_,execution::sequenced_policy>()(tnsX,R,factorsInit);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd), with the use of an 
   * @c Execution @c Policy, which can be either @c sequential, @c parallel 
   * with the use of @c MPI, or @c parallel with the use of @c OpenMP. 
   * In order to choose a policy, type @c execution::seq, @c execution::mpi or 
   * @c execution::omp. 
   * Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * @tparam ExecutionPolicy  Type of @c stl @c Execution @c Policy 
   *                          (sequential, parallel-mpi).
   * @tparam Tensor_          Type(data type and order) of input Tensor.
   *                          @c Tensor_ must be @c partensor::Tensor<order>, where 
   *                          @c order must be in range of @c [3-8].
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
  template <typename ExecutionPolicy, typename Tensor_>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpd( ExecutionPolicy            &&, 
       Tensor_              const &tnsX, 
       std::size_t          const  R, 
       Options<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues> const &options,
       MatrixArray<Tensor_> const &factorsInit )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD<Tensor_>()(tnsX,R,options,factorsInit);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD<Tensor_,execution::openmpi_policy>()(tnsX,R,options,factorsInit);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::CPD<Tensor_,execution::openmp_policy>()(tnsX,R,options,factorsInit);
    }
    else
      return internal::CPD<Tensor_>()(tnsX,R,options,factorsInit);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd). Sequential Policy.
   * 
   * @tparam Tensor_          Type(data type and order) of input Tensor.
   *                          @c Tensor_ must be @c partensor::Tensor<order>, where 
   *                          @c order must be in range of @c [3-8].
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
  template<typename Tensor_>
  auto cpd(Tensor_              const &tnsX, 
           std::size_t          const  R, 
           Options<Tensor_>     const &options, 
           MatrixArray<Tensor_> const &factorsInit)
  {
    return internal::CPD<Tensor_,execution::sequenced_policy>()(tnsX,R,options,factorsInit);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd), with the use of an 
   * @c Execution @c Policy, which can be either @c sequential, @c parallel 
   * with the use of @c MPI, or @c parallel with the use of @c OpenMP. 
   * In order to choose a policy, type @c execution::seq, @c execution::mpi or 
   * @c execution::omp. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * With this version of @c cpd the Tensor can be read from a file, specified in 
   * @c path variable.
   * 
   * @tparam ExecutionPolicy Type of @c stl @c Execution @c Policy 
   *                         (sequential, parallel-mpi).
   * @tparam TnsSize         Order of input Tensor.
   * 
   * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
   *                         length must be same as the Tensor order.    
   * @param  R          [in] The rank of decomposition.
   * @param  path       [in] The path where the tensor is located.
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template <typename ExecutionPolicy, std::size_t TnsSize>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpd( ExecutionPolicy                &&, 
       std::array<int, TnsSize> const &tnsDims, 
       std::size_t              const  R,
       std::string              const &path )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;
    using  Tensor_  = Tensor<static_cast<int>(TnsSize)>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD<Tensor_>()(tnsDims,R,path);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD<Tensor_,execution::openmpi_policy>()(tnsDims,R,path);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::CPD<Tensor_,execution::openmp_policy>()(tnsDims,R,path);
    }
    else
      return internal::CPD<Tensor_>()(tnsDims,R,path);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd). Sequential Policy. 
   * With this version of @c cpd, the Tensor can be read from a file, specified in 
   * @c path variable.
   * 
   * @tparam TnsSize         Order of input Tensor.
   * 
   * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
   *                         length must be same as the Tensor order.      
   * @param  R          [in] The rank of decomposition.
   * @param  path       [in] The path where the tensor is located.
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template<std::size_t TnsSize>
  auto cpd(std::array<int, TnsSize> const &tnsDims, 
           std::size_t              const  R, 
           std::string              const &path )
  {
    using Tensor_ = Tensor<static_cast<int>(TnsSize)>;
    return internal::CPD<Tensor_,execution::sequenced_policy>()(tnsDims,R,path);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd), with the use of an 
   * @c Execution @c Policy, which can be either @c sequential, @c parallel 
   * with the use of @c MPI, or @c parallel with the use of @c OpenMP. 
   * In order to choose a policy, type @c execution::seq, @c execution::mpi or 
   * @c execution::omp. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * With this version of @c cpd the Tensor can be read from a file, specified in 
   * @c path variable.
   * 
   * @tparam ExecutionPolicy Type of @c stl @c Execution @c Policy 
   *                         (sequential, parallel-mpi).
   * @tparam TnsSize         Order of input Tensor.
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
  template <typename ExecutionPolicy, std::size_t TnsSize>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpd( ExecutionPolicy                &&, 
       std::array<int,TnsSize>  const &tnsDims, 
       std::size_t              const  R,
       std::string              const &path,
       Options<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues> const &options )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;
    using  Tensor_  = Tensor<static_cast<int>(TnsSize)>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD<Tensor_>()(tnsDims,R,path,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD<Tensor_,execution::openmpi_policy>()(tnsDims,R,path,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::CPD<Tensor_,execution::openmp_policy>()(tnsDims,R,path,options);
    }
    else
      return internal::CPD<Tensor_>()(tnsDims,R,path,options);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd). Sequential Policy. 
   * With this version of @c cpd, the Tensor can be read from a file, specified in 
   * @c path variable.
   * 
   * @tparam TnsSize         Order of input Tensor.
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
  template<std::size_t TnsSize>
  auto cpd(std::array<int, TnsSize>                   const &tnsDims, 
           std::size_t                                const  R, 
           std::string                                const &path, 
           Options<Tensor<static_cast<int>(TnsSize)>> const &options )
  {
    using Tensor_ = Tensor<static_cast<int>(TnsSize)>;
    return internal::CPD<Tensor_,execution::sequenced_policy>()(tnsDims,R,path,options);
  }


  /**
   * Interface of Canonical Polyadic Decomposition(cpd), with the use of an 
   * @c Execution @c Policy, which can be either @c sequential, @c parallel 
   * with the use of @c MPI, or @c parallel with the use of @c OpenMP. 
   * In order to choose a policy, type @c execution::seq, @c execution::mpi or 
   * @c execution::omp. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * With this version of @c cpd, the Tensor and the initialized factors can be read from files, 
   * specified in @c paths variable.
   * 
   * @tparam ExecutionPolicy  Type of @c stl @c Execution @c Policy 
   *                          (sequential, parallel-mpi).
   * @tparam TnsSize          Order of input Tensor.
   * 
   * @param  tnsDims     [in] @c Stl array containing the Tensor dimensions, whose
   *                         length must be same as the Tensor order.     
   * @param  R           [in] The rank of decomposition.
   * @param  paths       [in] An @c stl array containing paths for the Tensor to be 
   *                          factorized and after that the paths for the initialized 
   *                          factors. 
   * 
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template <typename ExecutionPolicy, std::size_t TnsSize>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpd( ExecutionPolicy                          &&, 
       std::array<int, TnsSize>           const &tnsDims, 
       std::size_t                        const  R,
       std::array<std::string, TnsSize+1> const &paths  )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;
    using  Tensor_  = Tensor<static_cast<int>(TnsSize)>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD<Tensor_>()(tnsDims,R,paths);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD<Tensor_,execution::openmpi_policy>()(tnsDims,R,paths);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::CPD<Tensor_,execution::openmp_policy>()(tnsDims,R,paths);
    }
    else
      return internal::CPD<Tensor_>()(tnsDims,R,paths);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd). Sequential Policy. 
   * With this version of @c cpd, the Tensor can be read from a file, specified in 
   * @c path variable.
   * 
   * @tparam TnsSize         Order of input Tensor.
   * 
   * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
   *                         length must be same as the Tensor order.    
   * @param  R          [in] The rank of decomposition.
   * @param  paths      [in] An @c stl array containing paths for the Tensor to be 
   *                         factorized and after that the paths for the initialized 
   *                         factors. 
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template<std::size_t TnsSize>
  auto cpd(std::array<int, TnsSize>           const &tnsDims, 
           std::size_t                        const  R,
           std::array<std::string, TnsSize+1> const &paths  )
  {
    using Tensor_ = Tensor<static_cast<int>(TnsSize)>;
    return internal::CPD<Tensor_,execution::sequenced_policy>()(tnsDims,R,paths);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd), with the use of an 
   * @c Execution @c Policy, which can be either @c sequential, @c parallel 
   * with the use of @c MPI, or @c parallel with the use of @c OpenMP. 
   * In order to choose a policy, type @c execution::seq, @c execution::mpi or 
   * @c execution::omp. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * With this version of @c cpd, the Tensor and the initialized factors can be read from files, 
   * specified in @c paths variable.
   * 
   * @tparam ExecutionPolicy  Type of @c stl @c Execution @c Policy 
   *                          (sequential, parallel-mpi).
   * @tparam TnsSize          Order of input Tensor.
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
  template <std::size_t TnsSize, typename ExecutionPolicy>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpd( ExecutionPolicy                          &&, 
       std::array<int, TnsSize>           const &tnsDims, 
       std::size_t                        const  R,
       std::array<std::string, TnsSize+1> const &paths, 
       Options<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues> const &options )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;
    using  Tensor_  = Tensor<static_cast<int>(TnsSize)>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD<Tensor_>()(tnsDims,R,paths,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD<Tensor_,execution::openmpi_policy>()(tnsDims,R,paths,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::CPD<Tensor_,execution::openmp_policy>()(tnsDims,R,paths,options);
    }
    else
      return internal::CPD<Tensor_>()(tnsDims,R,paths,options);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd). Sequential Policy. 
   * With this version of @c cpd, the Tensor can be read from a file, specified in 
   * @c path variable.
   * 
   * @tparam TnsSize         Order of input Tensor.
   * 
   * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
   *                         length must be same as the Tensor order.   
   * @param  R          [in] The rank of decomposition.   
   * @param  paths      [in] An @c stl array containing paths for the Tensor to be 
   *                         factorized and after that the paths for the initialized 
   *                         factors. 
   * @param  options    [in] User's @c options, other than the default. It must be of
   *                         @c partensor::Options<partensor::Tensor<order>> type,
   *                         where @c order must be in range of @c [3-8].
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template<std::size_t TnsSize>
  auto cpd(std::array<int, TnsSize>                   const &tnsDims, 
           std::size_t                                const  R, 
           std::array<std::string,TnsSize+1>          const &paths, 
           Options<Tensor<static_cast<int>(TnsSize)>> const &options )
  {
    using Tensor_ = Tensor<static_cast<int>(TnsSize)>;
    return internal::CPD<Tensor_,execution::sequenced_policy>()(tnsDims,R,paths,options);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd), with the use of an 
   * @c Execution @c Policy, which can be either @c sequential, @c parallel 
   * with the use of @c MPI, or @c parallel with the use of @c OpenMP. 
   * In order to choose a policy, type @c execution::seq, @c execution::mpi or 
   * @c execution::omp. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * With this version of @c cpd, the Tensor can be read from a file, specified in 
   * @c path variable.
   * 
   * @tparam ExecutionPolicy  Type of @c stl @c Execution @c Policy 
   *                          (sequential, parallel-mpi).
   * @tparam TnsSize          Order of input Tensor.
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
  template <std::size_t TnsSize, typename ExecutionPolicy>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpd( ExecutionPolicy                        &&, 
       std::array<int, TnsSize>         const &tnsDims, 
       std::size_t                      const  R,
       std::array<std::string, TnsSize> const &true_paths, 
       std::array<std::string, TnsSize> const &init_paths, 
       Options<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues> const &options )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;
    using  Tensor_  = Tensor<static_cast<int>(TnsSize)>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD<Tensor_>()(tnsDims,R,true_paths,init_paths,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD<Tensor_,execution::openmpi_policy>()(tnsDims,R,true_paths,init_paths,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::CPD<Tensor_,execution::openmp_policy>()(tnsDims,R,true_paths,init_paths,options);
    }
    else
      return internal::CPD<Tensor_>()(tnsDims,R,true_paths,init_paths,options);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd). Sequential Policy. 
   * With this version of @c cpd, the Tensor can be read from a file, specified in 
   * @c path variable.
   * 
   * @tparam TnsSize         Order of input Tensor.
   * 
   * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
   *                         length must be same as the Tensor order.     
   * @param  R          [in] The rank of decomposition.
   * @param  true_paths [in] An @c stl array containing paths for the true factors.    
   * @param  init_paths [in] An @c stl array containing paths for initialized 
   *                         factors. 
   * @param  options    [in] User's @c options, other than the default. It must be of
   *                         @c partensor::Options<partensor::Tensor<order>> type,
   *                         where @c order must be in range of @c [3-8].
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template<std::size_t TnsSize>
  auto cpd(std::array<int, TnsSize>                   const &tnsDims, 
           std::size_t                                const  R, 
           std::array<std::string, TnsSize>           const &true_paths, 
           std::array<std::string, TnsSize>           const &init_paths,  
           Options<Tensor<static_cast<int>(TnsSize)>> const &options )
  {
    using Tensor_ = Tensor<static_cast<int>(TnsSize)>;
    return internal::CPD<Tensor_,execution::sequenced_policy>()(tnsDims,R,true_paths,init_paths,options);
  }

}

#endif // PARTENSOR_CPD_HPPP