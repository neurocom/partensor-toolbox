#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      11/01/2021
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
* @file      GtcStochasticOpenMP.hpp
* @details
* Implements the Canonical Polyadic Decomposition(gtc) using @c OpenMP.
* Make use of @c spdlog library in order to write output in
* a log file in @c "../log".
********************************************************************/

#if !defined(PARTENSOR_GTC_STOCHASTIC_HPP)
#error "GTC_STOCHASTIC_OMP can only included inside GTC_STOCHASTIC"
#endif /* PARTENSOR_GTC_STOCHASTIC_HPP */

namespace partensor
{
  inline namespace v1
  {
    namespace internal
    {
      template <std::size_t TnsSize_>
      struct GTC_STOCHASTIC<TnsSize_, execution::openmp_policy> : public GTC_STOCHASTIC_Base<TnsSize_>
      {
        using          GTC_STOCHASTIC_Base<TnsSize_>::TnsSize;
        using          GTC_STOCHASTIC_Base<TnsSize_>::lastFactor;
        using typename GTC_STOCHASTIC_Base<TnsSize_>::Dimensions;
        using typename GTC_STOCHASTIC_Base<TnsSize_>::MatrixArray;
        using typename GTC_STOCHASTIC_Base<TnsSize_>::DataType;
        using typename GTC_STOCHASTIC_Base<TnsSize_>::SparseTensor;
        using typename GTC_STOCHASTIC_Base<TnsSize_>::IntArray;
        using typename GTC_STOCHASTIC_Base<TnsSize_>::LongMatrix;
        
        using Options = partensor::SparseOptions<TnsSize_,execution::openmp_policy,SparseDefaultValues>;
        using Status  = partensor::SparseStatus<TnsSize_,execution::openmp_policy,SparseDefaultValues>;

        // Variables that will be used in gtc implementations. 
        struct Member_Variables 
        {
          MatrixArray  factors_T;        
          MatrixArray  factor_T_factor;  
          MatrixArray  mttkrp_T;        
          IntArray     tnsDims;         
          std::array<std::array<int, TnsSize_ - 1>, TnsSize_> offsets;         
          
          MatrixArray  norm_factors_T;
          MatrixArray  old_factors_T;

          Matrix       cwise_factor_product;
          Matrix       Ratings_Base_T;
          SparseTensor tnsX;

          // bool         all_orthogonal = true;
          // int          weight_factor;
          int          rank;
          double       c_stochastic_perc;

          MatrixArray grad;
          MatrixArray Y;
          MatrixArray invL;

          Member_Variables() = default;

          Member_Variables(int R, IntArray dims) :  tnsDims(dims),
                                                    rank(R)
          {}

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
         * @param  st  [in] Struct where the returned values of @c Gtc are stored.
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
         * Compute the cost function value at the end of each outer iteration
         * based on the last factor.
         * 
         * @param  mv  [in]     Struct where ALS variables are stored.
         * @param  st  [in,out] Struct where the returned values of @c Gtc are stored.
         *                      In this case the cost function value is updated.
         */
        void cost_function(Member_Variables const &mv,
                           Status                 &st)
        {
          Matrix temp_R_1(mv.rank, 1);
          double temp_1_1 = 0;
          double f_value_loc = 0;
          
          #pragma omp master
          st.f_value = 0;

          #pragma omp barrier
         
          std::array<int,TnsSize-1> offsets;
          offsets[0] = 1;
          for (int j = 1; j < static_cast<int>(TnsSize) - 1; j++)
          {
            offsets[j] = offsets[j - 1] * mv.tnsDims[j-1];
          }

          #pragma omp for schedule(static)
          for (long int i = 0; i < mv.tnsX[lastFactor].outerSize(); ++i)
          {
              int row = 0;
              for (SparseMatrix::InnerIterator it(mv.tnsX[lastFactor], i); it; ++it)
              {
                  temp_R_1 = mv.factors_T[lastFactor].col(it.col());
                  // Select rows of each factor an compute the Hadamard product of the respective row of the Khatri-Rao product, and the row of factor A_N.
                  for (int mode_i = static_cast<int>(TnsSize) - 2; mode_i >= 0; mode_i--)
                  {
                      row                = ((it.row()) / offsets[mode_i]) % (mv.tnsDims[mode_i]);
                      temp_R_1.noalias() = temp_R_1.cwiseProduct(mv.factors_T[mode_i].col(row));
                  }
                  temp_1_1 = it.value() - temp_R_1.sum();
                  f_value_loc += temp_1_1 * temp_1_1;
              }
          }
          #pragma omp atomic
          st.f_value += f_value_loc;

        }
        
        /*
         * Compute the cost function value at the end of each outer iteration
         * based on the last accelerated factor.
         * 
         * @param  mv                [in] Struct where ALS variables are stored.
         * @param  accel_factors     [in] Accelerated factors.
         * 
         * @returns The cost function calculated with the accelerated factors.
         */
        double accel_cost_function(Member_Variables       const &mv,
                                   MatrixArray            const &accel_factors)
        {
          Matrix temp_R_1(mv.rank, 1);
          double temp_1_1 = 0;
          double f_value = 0;

          std::array<int,TnsSize-1> offsets;
          offsets[0] = 1;
          for (int j = 1; j < static_cast<int>(TnsSize) - 1; j++)
          {
            offsets[j] = offsets[j - 1] * mv.tnsDims[j-1];
          }

          #pragma omp for schedule(static)
          for (long int i = 0; i < mv.tnsX[lastFactor].outerSize(); ++i)
          {
            int row = 0;
            for (SparseMatrix::InnerIterator it(mv.tnsX[lastFactor], i); it; ++it)
            {
                temp_R_1 = accel_factors[lastFactor].col(it.col());
                // Select rows of each factor an compute the Hadamard product of the respective row of the Khatri-Rao product, and the row of factor A_N.
                // temp_R_1  = A_N(i_N,:) .* ... .* A_2(i_2,:) .* A_1(i_1,:)
                for (int mode_i = static_cast<int>(TnsSize) - 2; mode_i >= 0; mode_i--)
                {
                    row                = ((it.row()) / offsets[mode_i]) % (mv.tnsDims[mode_i]);
                    temp_R_1.noalias() = temp_R_1.cwiseProduct(accel_factors[mode_i].col(row));
                }
                temp_1_1 = it.value() - temp_R_1.sum();
                f_value += temp_1_1 * temp_1_1;
            }
          }
          return f_value;
        }

        void calculate_offsets(Member_Variables &mv)
        {
          for (int idx = 0; idx < static_cast<int>(TnsSize); idx++)
          {
            mv.offsets[idx][0] = 1;
            for (int j = 1, mode = 0; j < static_cast<int>(TnsSize) - 1; j++, mode++)
            {
              if (idx == mode)
              {
                mode++;
              }
              mv.offsets[idx][j] = mv.offsets[idx][j - 1] * mv.tnsDims[mode];
            }
          }
        }

        void unconstraint_update(int              const  idx,
                                 Member_Variables       &mv,
                                 Status                 &st)
        {
            int r = mv.rank;

            Matrix eye = st.options.lambdas[idx] * Matrix::Identity(r, r);

            int last_mode = (idx == static_cast<int>(TnsSize) - 1) ? static_cast<int>(TnsSize) - 2 : static_cast<int>(TnsSize) - 1;

            Matrix MTTKRP_col(r, 1);
            Matrix temp_RxR(r, r);
            Matrix temp_R_1(r, 1);

            // Compute MTTKRP
            #pragma omp for schedule(guided) nowait
            for (long int i = 0; i < mv.tnsX[idx].outerSize(); ++i)
            {
                MTTKRP_col.setZero();
                temp_RxR.setZero(); // is the Hadamard product of Grammians of the Factors, that correspond to the nnz elements of the Tensor.
                for (SparseMatrix::InnerIterator it(mv.tnsX[idx], i); it; ++it)
                {
                    temp_R_1 = Matrix::Ones(r, 1);
                    int row;
                    // Select rows of each factor an compute the respective row of the Khatri-Rao product.
                    for (int mode_i = last_mode, kr_counter = static_cast<int>(TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
                    {
                        if (mode_i == idx)
                        {
                            continue;
                        }
                        row      = ((it.row()) / mv.offsets[idx][kr_counter]) % (mv.tnsDims[mode_i]);
                        temp_R_1 = temp_R_1.cwiseProduct(mv.factors_T[mode_i].col(row));
                        kr_counter--;
                    }
                    // Subtract from the previous row the respective row of W, according to relation (9).
                    MTTKRP_col.noalias() += it.value() * temp_R_1;
                    temp_RxR.noalias()   += temp_R_1 * temp_R_1.transpose();
                }
                mv.factors_T[idx].col(i) = (temp_RxR + eye).inverse() * MTTKRP_col;
            }
        }
        
        /*
         * Based on each factor's constraint, a different
         * update function is used at every outer iteration.
         * 
         * Computes also factor^T * factor at the end.
         *
         * @param  idx [in]     Factor to be updated.
         * @param  mv  [in]     Struct where ALS variables are stored.
         * @param  st  [in,out] Struct where the returned values of @c Gtc are stored.
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
              // unconstraint_update(idx, mv, st);
              break;
            }
            case Constraint::nonnegativity:
            {
              dynamic_blocksize::local_L::StochasticNesterovMNLS(mv.factors_T, mv.tnsDims, mv.tnsX[idx], mv.offsets[idx], mv.c_stochastic_perc, mv.Y[idx], 
                           st.options.max_nesterov_iter, st.options.lambdas[idx], idx);
              break;
            }
            default: // in case of Constraint::constant
              break;
          }

          // Compute A^T * A + B^T * B + ...
          #pragma omp master
          {
            st.factors[idx] = mv.factors_T[idx].transpose();
            mv.factor_T_factor[idx].noalias() = mv.factors_T[idx] * st.factors[idx];
          }
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
         * @param  st  [in,out] Struct where the returned values of @c Gtc are stored.
         *                      If the acceleration succeeds updates @c factors 
         *                      and cost function value.  
         * 
         */
        void line_search_accel(Member_Variables &mv,
                               Status           &st,
                               double           &f_accel,
                               double           &accel_step,
                               MatrixArray      &accel_factors_T,
                               MatrixArray      &accel_gramians)
        {
          #pragma omp master
          {
            for(std::size_t i=0; i<TnsSize; ++i)
            {
                accel_factors_T[i] = mv.old_factors_T[i] + accel_step * (mv.factors_T[i] - mv.old_factors_T[i]); 
                accel_gramians[i]  = accel_factors_T[i] * accel_factors_T[i].transpose();
            }
                    
            f_accel = 0;
          }

          #pragma omp barrier

          double f_accel_loc = accel_cost_function(mv, accel_factors_T);

          #pragma omp atomic
          f_accel += f_accel_loc;

          #pragma omp barrier

          #pragma omp master
          {
            if (st.f_value > f_accel)
            {
                mv.factors_T       = accel_factors_T;
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
        }

        /*
         * Sequential implementation of Alternating Least Squares (ALS) method.
         * 
         * @param  R   [in]     The rank of decomposition.
         * @param  mv  [in]     Struct where ALS variables are stored and being updated
         *                      until a termination condition is true.
         * @param  st  [in,out] Struct where the returned values of @c Gtc are stored.
         */
        void aogtc_stochastic(Member_Variables       &mv,
                   Status                 &status)
        {
          double f_accel    = 0.0; // Objective Value after the acceleration step
          double accel_step = 0.0;
          
          MatrixArray  accel_factors_T;
          MatrixArray  accel_gramians;

          for (std::size_t i=0; i<TnsSize; i++)
          {
            mv.Y[i]            = Matrix::Zero(mv.rank, mv.tnsDims[i]);
            mv.factors_T[i]    = status.factors[i].transpose();
            mv.mttkrp_T[i]     = Matrix(mv.rank, mv.tnsDims[i]);
            mv.factor_T_factor[i].noalias() = mv.factors_T[i] * status.factors[i];
            accel_factors_T[i] = mv.factors_T[i];
            accel_gramians[i]  = Matrix::Zero(mv.rank, mv.rank);
          }

          // if(status.options.normalization)
          // {
          //   choose_normilization_factor(status, mv.all_orthogonal, mv.weight_factor);
          // }
          // Normalize(static_cast<int>(R), mv.factor_T_factor, status.factors);

          std::size_t epoch = static_cast<std::size_t> (1/mv.c_stochastic_perc);
          std::size_t epoch_counter = 0;

          const int total_num_threads = get_num_threads();
          omp_set_nested(0);
          
          status.frob_tns         = (mv.tnsX[0]).squaredNorm();

          #pragma omp parallel \
                  num_threads(total_num_threads) \
                  proc_bind(spread)\
                  default(shared)
          {
            cost_function(mv, status);
            #pragma omp barrier

            #pragma omp master
            {
              status.rel_costFunction = status.f_value / status.frob_tns;
            }
            #pragma omp barrier
            
            // ---- Loop until ALS converges ----
            while(1)
            {
                #pragma omp master
                {
                    status.ao_iter++;
                    Partensor()->Logger()->info("iter: {} -- fvalue: {} -- relative_costFunction: {}", status.ao_iter, 
                                                            status.f_value, status.rel_costFunction);
                }
                
                for (std::size_t i=0; i<TnsSize; i++)
                {
                    // Update factor
                    update_factor(i, mv, status);
                }

                //	<---------------------	End loop for every mode		------------------------->	//
                // Cost function computation.
                if (status.ao_iter % epoch == 0)
                {
                  #pragma omp master
                  {
                    epoch_counter++;
                  }

                  #pragma omp barrier
                  
                  cost_function(mv, status);
                  
                  #pragma omp barrier

                  #pragma omp master
                  {
                    status.rel_costFunction = status.f_value/status.frob_tns;
                  }

                }
                    
                #pragma omp barrier

                // ---- Terminating condition ----
                if (status.rel_costFunction < status.options.threshold_error || epoch_counter >= status.options.max_iter)
                {
                    #pragma omp master
                    {
                        if (status.options.writeToFile)
                            writeFactorsToFile(status);  
                    }
                    break;
                }

                // if (status.options.acceleration)
                // {
                //     // ---- Acceleration Step ----
                //     if (status.ao_iter > 1)
                //     {
                //         #pragma omp master
                //         accel_step = pow(status.ao_iter+1,(1.0/(status.options.accel_coeff)));

                //         line_search_accel(mv, status, f_accel, accel_step, accel_factors_T, accel_gramians);
                //         #pragma omp barrier
                //     }
                //     #pragma omp master
                //     {
                //         for (int i = 0; i < static_cast<int>(TnsSize); i++)
                //             mv.old_factors_T[i] = mv.factors_T[i];
                //     }
                // }  
            } // end of while
          } // end of pragma
        }

        /**
         * Initialization of factors.
         */
        void initialize_factors(Member_Variables &mv,
                                Status           &status)
        {
          if(status.options.initialized_factors)
          {
            if(status.options.read_factors_from_file)
            {
              for(std::size_t i=0; i<TnsSize; ++i)
              {   
                status.factors[i] = Matrix(mv.tnsDims[i], static_cast<int>(mv.rank));
                read( status.options.initial_factors_paths[i], 
                      mv.tnsDims[i] * mv.rank,
                      0, 
                      status.factors[i] );    
              } 
            }
            else
              status.factors = status.options.factorsInit;
          }
          else // produce estimate factors using uniform distribution with entries in [0,1].
            makeFactors(mv.tnsDims, status.options.constraints, mv.rank, status.factors);
        }

        /**
         * Implementation of General Tensor Completion with default values in Options struct
         * and randomly generated initial factors.
         * 
         * @param  tnsX    [in] The given Tensor to be factorized of @c Tensor_ type, 
         *                      with @c double data.
         * @param  R       [in] The rank of decomposition.
         *
         * @returns An object of type @c Status with the results of the algorithm.
         */
        Status operator()(Options const &options)
        {
          Status           status(options);
          Member_Variables mv(options.rank, options.tnsDims);

          long long int fileSize = (TnsSize + 1) * options.nonZeros;

          Matrix Ratings_Base_T = Matrix(static_cast<int>(TnsSize+1), options.nonZeros);
          // Read the whole Tensor from a file
          read( options.ratings_path, 
                fileSize, 
                0, 
                Ratings_Base_T );

          // GTC_STOCHASTIC_Base<TnsSize>::sort_ratings_base(mv.Ratings_Base_T, options.nonZeros);
          // ReserveSparseTensor<TnsSize>(mv.tnsX, options.tnsDims, options.nonZeros);
          // FillSparseTensor<TnsSize>(mv.tnsX, options.nonZeros, mv.Ratings_Base_T, options.tnsDims);
          // mv.Ratings_Base_T.resize(0,0);
          GTC_STOCHASTIC_Base<TnsSize>::sort_ratings_base(Ratings_Base_T, mv.tnsX, options.tnsDims, options.nonZeros);
          Ratings_Base_T.resize(0,0);
          
          for (std::size_t i=0; i<TnsSize; i++)
          {
            mv.tnsX[i].makeCompressed();
          }
                  
          mv.c_stochastic_perc = options.c_stochastic_perc;
          
          calculate_offsets(mv);

          initialize_factors(mv, status);

          aogtc_stochastic(mv, status);

          return status;
        }

        /**
         * Implementation of General Tensor Completion with default values in Options struct
         * and randomly generated initial factors.
         * 
         * @param  tnsX    [in] The given Tensor to be factorized of @c Tensor_ type, 
         *                      with @c double data.
         * @param  R       [in] The rank of decomposition.
         *
         * @returns An object of type @c Status with the results of the algorithm.
         */
        Status operator()(Matrix       const &Ratings_Base_T, 
                          Options      const &options)
        {
          Status           status(options);
          Member_Variables mv(options.rank, options.tnsDims);

          // ReserveSparseTensor<TnsSize>(mv.tnsX, options.tnsDims, options.nonZeros);
          // FillSparseTensor<TnsSize>(mv.tnsX, options.nonZeros, Ratings_Base_T, options.tnsDims);
          // Ratings_Base_T.resize(0,0);
          GTC_STOCHASTIC_Base<TnsSize>::sort_ratings_base(Ratings_Base_T, mv.tnsX, options.tnsDims, options.nonZeros);
          // Ratings_Base_T.resize(0,0);
          
          for (std::size_t i=0; i<TnsSize; i++)
          {
            mv.tnsX[i].makeCompressed();
          }
                  
          mv.c_stochastic_perc = options.c_stochastic_perc;
          
          calculate_offsets(mv);

          // produce estimate factors using uniform distribution with entries in [0,1].
          initialize_factors(mv, status);

          partensor::timer.startChronoHighTimer();
          aogtc_stochastic(mv, status);
          double end_gtc_time_omp = partensor::timer.endChronoHighTimer();
          std::cout << "GtcStochasticOpenMP took " << end_gtc_time_omp << " sec." << std::endl;
          return status;
        }
      };
    }  // namespace internal
  }    // namespace v1
} // end namespace partensor