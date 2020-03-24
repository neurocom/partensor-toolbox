#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      14/11/2019
* @author    Christos Tsalidis
* @author    Yiorgos Lourakis
* @author    George Lykoudis
* @copyright 2019 Neurocom All Rights Reserved.
*/
#endif // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      CpdMpi.hpp
* @details
* Implements the Canonical Polyadic Decomposition(cpd) using @c MPI.
* Make use of @c spdlog library in order to write output in
* a log file in @c "../log".
********************************************************************/

#if !defined(PARTENSOR_CPD_HPP)
#error "CPD_MPI can only included inside CPD"
#endif /* PARTENSOR_CPD_HPP */

namespace partensor
{

  inline namespace v1 {

    namespace internal {
      /**
       * Includes the implementation of CPDMPI factorization. Based on the given
       * parameters one of the four overloaded operators will be called.
       * @tparam Tensor_ The Type of The given @c Tensor to be factorized.
       */
      template<typename Tensor_>
      struct CPD<Tensor_,execution::openmpi_policy> : public CPD_Base<Tensor_>
      {
        using          CPD_Base<Tensor_>::TnsSize;
        using          CPD_Base<Tensor_>::lastFactor;
        using typename CPD_Base<Tensor_>::Dimensions;
        using typename CPD_Base<Tensor_>::MatrixArray;
        using typename CPD_Base<Tensor_>::DataType;

        using IntArray         = typename TensorTraits<Tensor_>::IntArray; /**< Stl array of size TnsSize and containing int type. */

        using CartCommunicator = partensor::cartesian_communicator; // From ParallelWrapper.hpp
        using CartCommVector   = std::vector<CartCommunicator>;
        using IntVector        = std::vector<int>;
        using Int2DVector      = std::vector <std::vector<int>>;

        using Options = partensor::Options<Tensor_,execution::openmpi_policy,DefaultValues>;
        using Status  = partensor::Status<Tensor_,execution::openmpi_policy,DefaultValues>;
        
        // Variables that will be used in cpd implementations. 
        struct Member_Variables 
        {
          MPI_Communicator &world = Partensor()->MpiCommunicator(); // MPI_COMM_WORLD

          double          local_f_value;
          int             RxR;
          int             world_size;

          Int2DVector     displs_subTns;       // skipping dimension "rows" for each subtensor
          Int2DVector     displs_subTns_R;     // skipping dimension "rows" for each subtensor times R ( for MPI communication purposes )
          Int2DVector     subTnsDims;          // dimensions of subtensor
          Int2DVector     subTnsDims_R;        // dimensions of subtensor times R ( for MPI communication purposes )
          Int2DVector     displs_local_update; // displacement in the local factor for update rows         
          Int2DVector     send_recv_counts;    // rows to be communicated after update times R

          CartCommVector  layer_comm;
          CartCommVector  fiber_comm;

          IntArray        layer_rank;
          IntArray        fiber_rank;
          IntArray        rows_for_update;
          IntArray        subTns_offsets;
          IntArray        subTns_extents;

          MatrixArray     proc_krao;
          MatrixArray     layer_factors;
          MatrixArray     layer_factors_T;
          MatrixArray     factors_T;
          MatrixArray     factor_T_factor;
          MatrixArray     local_mttkrp;
          MatrixArray     layer_mttkrp;
          MatrixArray     local_mttkrp_T;
          MatrixArray     layer_mttkrp_T;
          MatrixArray     subTns_mat;
          MatrixArray     local_factors;
          MatrixArray     local_factors_T;
          MatrixArray     norm_factors;
          MatrixArray     old_factors;
          MatrixArray     true_factors;

          Matrix          cwise_factor_product;
          Matrix          tnsX_mat_lastFactor_T;
          Matrix          temp_matrix;
          Matrix          nesterov_old_layer_factor;

          Tensor_         subTns;

          bool            all_orthogonal = true;
          int             weight_factor;
          
          /*
           * Calculates if the number of processors given from terminal 
           * are equal to the processors in the implementation.
           * 
           * @param procs [in] @c stl array with the number of processors per
           *                   dimension of the tensor.
           */
          void check_processor_avaliability(std::array<int, TnsSize> const &procs)
          {          
            // MPI_Environment &env = Partensor()->MpiEnvironment();
            world_size = world.size();
            //  numprocs must be product of options.proc_per_mode
            if (std::accumulate(procs.begin(), procs.end(), 1,
                                std::multiplies<int>()) != world_size && world.rank() == 0) {
              Partensor()->Logger()->error("The product of the processors per mode must be equal to {}\n", world_size);
              // env.abort(-1);
            }
          }  

          Member_Variables() = default;
          Member_Variables(int R, std::array<int, TnsSize> &procs) :  local_f_value(0.0),
                                                                      RxR(R*R),
                                                                      displs_subTns(TnsSize),
                                                                      displs_subTns_R(TnsSize),
                                                                      subTnsDims(TnsSize),
                                                                      subTnsDims_R(TnsSize),
                                                                      displs_local_update(TnsSize),
                                                                      send_recv_counts(TnsSize)
          {
            check_processor_avaliability(procs);
            layer_comm.reserve(TnsSize);
            fiber_comm.reserve(TnsSize);
          }

          Member_Variables(Member_Variables const &) = default;
          Member_Variables(Member_Variables      &&) = default;

          Member_Variables &operator=(Member_Variables const &) = default;
          Member_Variables &operator=(Member_Variables      &&) = default; 
        };

        /*
         * In case option variable @c writeToFile is enabled then, before the end
         * of the algorithm writes the resulted factors in files, where their
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
         * Compute the cost function value at the end of each outer iteration
         * based on the last factor.
         * 
         * @param  grid_comm [in]     MPI communicator where the new cost function value
         *                            will be communicated and computed.
         * @param  mv        [in]     Struct where ALS variables are stored.
         * @param  st        [in,out] Struct where the returned values of @c Cpd are stored.
         *                            In this case the cost function value is updated.
         */
        void cost_function( CartCommunicator const &grid_comm,
                            Member_Variables       &mv,
                            Status                 &st )
        {
            mv.local_f_value =
                ((mv.proc_krao[lastFactor].transpose() * mv.tnsX_mat_lastFactor_T) * mv.layer_factors[lastFactor]).trace();  
            all_reduce( grid_comm, 
                        inplace(&mv.local_f_value),
                        1, 
                        std::plus<double>() );
                        
            Matrix cwiseFactor_prod = PartialCwiseProd(mv.factor_T_factor, lastFactor) * mv.factor_T_factor[lastFactor];
            st.f_value = 
                sqrt(st.frob_tns - 2 * mv.local_f_value + cwiseFactor_prod.trace());
        }

        /*
         * Compute the cost function value at the end of each outer iteration
         * based on the last accelerated factor.
         * 
         * @param  grid_comm         [in] MPI communicator where the new cost function value
         *                                will be communicated and computed.
         * @param  mv                [in] Struct where ALS variables are stored.
         * @param  st                [in] Struct where the returned values of @c Cpd are stored.
         *                                In this case the cost function value is updated.
         * @param  factors           [in] Accelerated factors.
         * @param  factors_T_factors [in] Gramian matrices of factors.
         * 
         * @returns The cost function calculated with the accelerated factors.
         */
        double accel_cost_function(CartCommunicator const &grid_comm,
                                   Member_Variables const &mv,
                                   Status           const &st,
                                   MatrixArray      const &factors,
                                   MatrixArray      const &factors_T_factors)
        {
          double local_f_value = 
              ((PartialKhatriRao(factors, lastFactor).transpose() * mv.tnsX_mat_lastFactor_T) * factors[lastFactor]).trace();
          all_reduce( grid_comm, 
                      inplace(&local_f_value),
                      1, 
                      std::plus<double>() );
          Matrix cwiseFactor_prod = PartialCwiseProd(factors_T_factors, lastFactor) * factors_T_factors[lastFactor];
          return sqrt(st.frob_tns - 2 * local_f_value + cwiseFactor_prod.trace());
        }

        /*
         * Make use of the dimensions and the number of processors per dimension
         * and then calculates the dimensions of the subtensor and subfactor for 
         * each processor.
         * 
         * @tparam Dimensions          Array type containing the length of Tensor's dimensions.
         * 
         * @param  tnsDims    [in]     Tensor Dimensions. Each index contains the corresponding 
         *                             factor's rows length.
         * @param  st         [in]     Struct where the returned values of @c Cpd are stored.
         * @param  R          [in]     The rank of decomposition.
         * @param  mv         [in,out] Struct where ALS variables are stored. 
         *                             Updates @c stl arrays with dimensions for subtensors and
         *                             subfactors.
         */
        template<typename Dimensions>
        void compute_sub_dimensions(Dimensions       const &tnsDims,
                                    Status           const &st,
                                    std::size_t      const  R,
                                    Member_Variables       &mv)
        {
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.factor_T_factor[i].noalias() = st.factors[i].transpose() * st.factors[i];

            DisCount(mv.displs_subTns[i], mv.subTnsDims[i], st.options.proc_per_mode[i], tnsDims[i], 1);
            // for fiber communication and Gatherv
            DisCount(mv.displs_subTns_R[i], mv.subTnsDims_R[i], st.options.proc_per_mode[i], tnsDims[i], static_cast<int>(R));
            // information per layer
            DisCount(mv.displs_local_update[i], mv.send_recv_counts[i], mv.world_size / st.options.proc_per_mode[i], 
                                                  mv.subTnsDims[i][mv.fiber_rank[i]], static_cast<int>(R));

            mv.rows_for_update[i] = mv.send_recv_counts[i][mv.layer_rank[i]] / static_cast<int>(R);
            mv.subTns_offsets[i]  = mv.displs_subTns[i][mv.fiber_rank[i]];
            mv.subTns_extents[i]  = mv.subTnsDims[i][mv.fiber_rank[i]];
          }
        }        

        /*
         * Based on each factor's constraint, a different
         * update function is used at every outer iteration.
         * 
         * Computes also factor^T * factor at the end.
         *
         * @param  idx [in]     Factor to be updated.
         * @param  R   [in]     The rank of decomposition.
         * @param  st  [in]     Struct where the returned values of @c Cpd are stored.
         *                      Here constraints and options variables are needed.
         * @param  mv  [in,out] Struct where ALS variables are stored.
         *                      Updates the factors of each layer.
         */
        void update_factor(int              const  idx,
                           std::size_t      const  R, 
                           Status           const &st, 
                           Member_Variables       &mv  )
        {
          switch ( st.options.constraints[idx] ) 
          {
            case Constraint::unconstrained:
            {
              v2::reduce_scatter( mv.layer_comm[idx], 
                                  mv.layer_mttkrp_T[idx], 
                                  mv.send_recv_counts[idx][0],
                                  mv.local_mttkrp_T[idx] );

              mv.local_mttkrp[idx] = mv.local_mttkrp_T[idx].transpose();
              if (mv.rows_for_update[idx] != 0)
                mv.local_factors[idx].noalias() = mv.local_mttkrp[idx] * mv.cwise_factor_product.inverse();
              break;
            }
            case Constraint::nonnegativity:
            {
              v2::reduce_scatter( mv.layer_comm[idx], 
                                  mv.layer_mttkrp_T[idx], 
                                  mv.send_recv_counts[idx][0],
                                  mv.local_mttkrp_T[idx] );

              mv.local_mttkrp[idx]         = mv.local_mttkrp_T[idx].transpose();
              mv.nesterov_old_layer_factor = mv.layer_factors[idx];
              if (mv.rows_for_update[idx] != 0) 
              {
                NesterovMNLS(mv.cwise_factor_product, mv.local_mttkrp[idx], st.options.nesterov_delta_1, 
                                st.options.nesterov_delta_2, mv.local_factors[idx]);
              }
              break;
            }
            case Constraint::orthogonality:
            {
              all_reduce( mv.layer_comm[idx],
                          inplace(mv.layer_mttkrp[idx].data()),
                          mv.subTnsDims_R[idx][mv.fiber_rank[idx]],
                          std::plus<double>() );

              if (mv.rows_for_update[idx] != 0)
              {
                mv.local_mttkrp[idx]     = mv.layer_mttkrp[idx].block(mv.displs_local_update[idx][mv.layer_rank[idx]] / static_cast<int>(R), 0, 
                                                                     mv.rows_for_update[idx],                             static_cast<int>(R)); 
                mv.temp_matrix.noalias() = mv.layer_mttkrp[idx].transpose() * mv.layer_mttkrp[idx];
              }
              all_reduce( mv.fiber_comm[idx],
                          inplace(mv.temp_matrix.data()),
                          mv.RxR,
                          std::plus<double>() );

              Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(mv.temp_matrix);
              mv.temp_matrix.noalias() = (eigensolver.eigenvectors()) 
                                          * (eigensolver.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal()) 
                                          * (eigensolver.eigenvectors().transpose());
              
              if(mv.rows_for_update[idx] != 0)
                mv.local_factors[idx].noalias() = mv.local_mttkrp[idx] * mv.temp_matrix;
              break;
            }
            case Constraint::sparsity:
              break;
            default: // in case of Constraint::constant
              break;
          } // end of constraints switch

          if (st.options.constraints[idx] != Constraint::constant)
          {
            mv.local_factors_T[idx] = mv.local_factors[idx].transpose();                 
            v2::all_gatherv( mv.layer_comm[idx], 
                             mv.local_factors_T[idx],
                             mv.send_recv_counts[idx][mv.layer_rank[idx]], 
                             mv.send_recv_counts[idx][0],
                             mv.displs_local_update[idx][0],
                             mv.layer_factors_T[idx] );
            
            mv.layer_factors[idx]             = mv.layer_factors_T[idx].transpose();
            mv.factor_T_factor[idx].noalias() = mv.layer_factors_T[idx] * mv.layer_factors[idx];
          }

          all_reduce( mv.fiber_comm[idx], 
                      inplace(mv.factor_T_factor[idx].data()), 
                      mv.RxR,
                      std::plus<double>() ); 
                      
          if(st.options.constraints[idx] == Constraint::nonnegativity)
          {
            if ((mv.factor_T_factor[idx].diagonal()).minCoeff()==0)
            {
              mv.layer_factors[idx] = 0.9 * mv.layer_factors[idx] + 0.1 * mv.nesterov_old_layer_factor;
              all_reduce( mv.fiber_comm[idx], 
                          inplace(mv.factor_T_factor[idx].data()), 
                          mv.RxR,
                          std::plus<double>() ); 
            }
          }
        }

        /*
         * At the end of the algorithm processor 0
         * collects each part of the factor that each
         * processor holds and return them in status.factors.
         *
         * @tparam Dimensions       Array type containing the Tensor dimensions.
         * 
         * @param  tnsDims [in]     Tensor Dimensions. Each index contains the corresponding 
         *                          factor's rows length.
         * @param  R       [in]     The rank of decomposition.
         * @param  mv      [in]     Struct where ALS variables are stored.
         *                          Use variables to compute result factors by gathering each 
         *                          part of the factor from processors. 
         * @param  st      [in,out] Struct where the returned values of @c Cpd are stored.
         *                          Stores the resulted factors.
         */
        template<typename Dimensions>
        void gather_final_factors(Dimensions       const &tnsDims,
                                  std::size_t      const  R,
                                  Member_Variables       &mv,
                                  Status                 &st)
        {
          for(std::size_t i=0; i<TnsSize; ++i)
            mv.layer_factors_T[i] = mv.layer_factors[i].transpose();

          for(std::size_t i=0; i<TnsSize; ++i)
          {
            mv.temp_matrix.resize(static_cast<int>(R), tnsDims[i]);
            // Gatherv from all processors to processor with rank 0 the final factors
            v2::gatherv( mv.fiber_comm[i],
                         mv.layer_factors_T[i], 
                         mv.subTnsDims_R[i][mv.fiber_rank[i]],
                         mv.subTnsDims_R[i][0], 
                         mv.displs_subTns_R[i][0],  
                         0,
                         mv.temp_matrix ); 
            
            st.factors[i] = mv.temp_matrix.transpose(); 
          }
        }

        /*
         * @brief Line Search Acceleration
         * 
         * Performs an acceleration step in the updated factors, and keeps the accelerated factors when 
         * the step succeeds. Otherwise, the acceleration step is ignored.
         * Line Search Acceleration reduces the number outer iterations in the ALS algorithm.
         * 
         * @note This implementation ONLY, if factors are of @c Matrix type.
         * 
         * @param  grid_comm [in]     MPI communicator where the new cost function value
         *                            will be communicated and computed.
         * @param  mv        [in,out] Struct where ALS variables are stored.
         *                            In case the acceration is successful layer factor^T * factor 
         *                            and layer factor variables are updated.
         * @param  st        [in,out] Struct where the returned values of @c Cpd are stored.
         *                            If the acceleration succeeds updates cost function value. 
         * 
         */
        void line_search_accel(CartCommunicator const &grid_comm,
                               Member_Variables       &mv,
                               Status                 &st)
        {
          double       f_accel    = 0.0; // Objective Value after the acceleration step
          double const accel_step = pow(st.ao_iter+1,(1.0/(st.options.accel_coeff)));
          
          MatrixArray   accel_factors;
          MatrixArray   accel_gramians;

          for(std::size_t i=0; i<TnsSize; ++i)
          {
            accel_factors[i]  = mv.old_factors[i] + accel_step * (mv.layer_factors[i] - mv.old_factors[i]); 
            accel_gramians[i] = accel_factors[i].transpose() * accel_factors[i];
            all_reduce( mv.fiber_comm[i], 
                        inplace(accel_gramians[i].data()),
                        mv.RxR, 
                        std::plus<double>() );    
          }

          f_accel = accel_cost_function(grid_comm, mv, st, accel_factors, accel_gramians);
          if (st.f_value > f_accel)
          {
            mv.layer_factors   = accel_factors;
            mv.factor_T_factor = accel_gramians;
            st.f_value         = f_accel;
            if(grid_comm.rank() == 0)
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
         * Parallel implementation of als method with MPI.
         * 
         * @tparam Dimensions         Array type containing the Tensor dimensions.
         * 
         * @param  grid_comm [in]     The communication grid, where the processors
         *                            communicate their cost function.
         * @param  tnsDims   [in]     Tensor Dimensions. Each index contains the corresponding 
         *                            factor's rows length.
         * @param  R         [in]     The rank of decomposition.
         * @param  mv        [in]     Struct where ALS variables are stored and being updated
         *                            until a termination condition is true.
         * @param  status    [in,out] Struct where the returned values of @c Cpd are stored.
         */
        template<typename Dimensions>
        void als(CartCommunicator const &grid_comm,
                 Dimensions       const &tnsDims,
                 std::size_t      const  R,
                 Member_Variables       &mv,
                 Status                 &status)
        {
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.proc_krao[i]  = PartialKhatriRao(mv.layer_factors, i);
            mv.subTns_mat[i] = Matricization(mv.subTns, i);

            mv.factor_T_factor[i].noalias() = mv.layer_factors[i].transpose() * mv.layer_factors[i];
            all_reduce( mv.fiber_comm[i], 
                        inplace(mv.factor_T_factor[i].data()),
                        mv.RxR, 
                        std::plus<double>() );

            mv.local_mttkrp_T[i].resize(R, mv.rows_for_update[i]);
            mv.layer_factors_T[i].resize(R, mv.subTnsDims[i][mv.fiber_rank[i]]);
          }

          mv.tnsX_mat_lastFactor_T = mv.subTns_mat[lastFactor].transpose();
          if(status.options.normalization)
          {
            choose_normilization_factor(status, mv.all_orthogonal, mv.weight_factor);
          }

          all_reduce( grid_comm, 
                      square_norm(mv.subTns), 
                      status.frob_tns, 
                      std::plus<double>());
          cost_function( grid_comm, mv, status );
          status.rel_costFunction = status.f_value / sqrt(status.frob_tns);

          // Wait for all processors to reach here
          grid_comm.barrier();

          // ---- Loop until ALS converges ----
          while(1) 
          {
              status.ao_iter++;
              if (!grid_comm.rank())
                  Partensor()->Logger()->info("iter: {} -- fvalue: {} -- relative_costFunction: {}", status.ao_iter, 
                                                  status.f_value, status.rel_costFunction);
              
              for (std::size_t i = 0; i < TnsSize; i++) 
              {
                mttkrp(mv.layer_factors, mv.subTns_mat[i], i, mv.proc_krao[i], mv.layer_mttkrp[i]);
                mv.cwise_factor_product = PartialCwiseProd(mv.factor_T_factor, i);
                mv.layer_mttkrp_T[i]    = mv.layer_mttkrp[i].transpose();
                mv.local_factors[i]     = mv.layer_factors[i].block(mv.displs_local_update[i][mv.layer_rank[i]] / static_cast<int>(R), 0, 
                                                                     mv.rows_for_update[i],                           static_cast<int>(R));
                
                update_factor(i, R, status, mv);
              }

              // ---- Cost function Computation ----
              cost_function(grid_comm, mv, status);
              status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
              if(status.options.normalization && !mv.all_orthogonal)
                Normalize(mv.weight_factor, R, mv.factor_T_factor, mv.layer_factors);
              
              // ---- Terminating condition ----
              if (status.rel_costFunction < status.options.threshold_error || status.ao_iter >= status.options.max_iter)
              {
                  gather_final_factors(tnsDims, R, mv, status);
                  if(grid_comm.rank() == 0)
                  {
                    Partensor()->Logger()->info("Processor 0 collected all {} factors.\n", TnsSize);
                    if(status.options.writeToFile)
                      writeFactorsToFile(status);
                  }
                  break;
              }

              if (status.options.acceleration)
              {
                mv.norm_factors = mv.layer_factors;
                // ---- Acceleration Step ----
                if (status.ao_iter > 1)
                  line_search_accel(grid_comm, mv, status);

                mv.old_factors = mv.norm_factors;
              }  

          } // end of outer while loop
        }

        /*
         * Parallel implementation of als method with MPI.
         * 
         * @tparam Dimensions         Array type containing the Tensor dimensions.
         * 
         * @param  grid_comm [in]     The communication grid, where the processors
         *                            communicate their cost function.
         * @param  tnsDims   [in]     Tensor Dimensions. Each index contains the corresponding 
         *                            factor's rows length.
         * @param  R         [in]     The rank of decomposition.
         * @param  mv        [in]     Struct where ALS variables are stored and being updated
         *                            until a termination condition is true.
         * @param  status    [in,out] Struct where the returned values of @c Cpd are stored.
         */
        template<typename Dimensions>
        void als_true_factors( CartCommunicator const &grid_comm,
                               Dimensions       const &tnsDims,
                               std::size_t      const  R,
                               Member_Variables       &mv,
                               Status                 &status )
        {
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.proc_krao[i]  = PartialKhatriRao(mv.layer_factors, i);
            mv.subTns_mat[i] = generateTensor(i, mv.true_factors);

            mv.factor_T_factor[i].noalias() = mv.layer_factors[i].transpose() * mv.layer_factors[i];
            all_reduce( mv.fiber_comm[i], 
                        inplace(mv.factor_T_factor[i].data()),
                        mv.RxR, 
                        std::plus<double>() );

            mv.local_mttkrp_T[i].resize(R, mv.rows_for_update[i]);
            mv.layer_factors_T[i].resize(R, mv.subTnsDims[i][mv.fiber_rank[i]]);
          }

          mv.tnsX_mat_lastFactor_T = mv.subTns_mat[lastFactor].transpose();
          if(status.options.normalization)
          {
            choose_normilization_factor(status, mv.all_orthogonal, mv.weight_factor);
          }

          all_reduce( grid_comm, 
                      (mv.subTns_mat[lastFactor]).squaredNorm(), 
                      status.frob_tns, 
                      std::plus<double>());
          cost_function( grid_comm, mv, status );
          status.rel_costFunction = status.f_value / sqrt(status.frob_tns);

          // Wait for all processors to reach here
          grid_comm.barrier();

          // ---- Loop until ALS converges ----
          while(1) 
          {
              status.ao_iter++;
              if (!grid_comm.rank())
                  Partensor()->Logger()->info("iter: {} -- fvalue: {} -- relative_costFunction: {}", status.ao_iter, 
                                                  status.f_value, status.rel_costFunction);
              
              for (std::size_t i = 0; i < TnsSize; i++) 
              {
                mv.proc_krao[i]              = PartialKhatriRao(mv.layer_factors, i);
                mv.cwise_factor_product      = PartialCwiseProd(mv.factor_T_factor, i);
                mv.layer_mttkrp[i].noalias() = mv.subTns_mat[i] * mv.proc_krao[i];
                mv.layer_mttkrp_T[i]         = mv.layer_mttkrp[i].transpose();
                mv.local_factors[i]          = mv.layer_factors[i].block(mv.displs_local_update[i][mv.layer_rank[i]] / static_cast<int>(R), 0, 
                                                                        mv.rows_for_update[i],                           static_cast<int>(R));
                
                update_factor(i, R, status, mv);
                all_reduce( mv.fiber_comm[i], 
                            inplace(mv.factor_T_factor[i].data()), 
                            mv.RxR,
                            std::plus<double>() );
              }

              // ---- Cost function Computation ----
              cost_function(grid_comm, mv, status);
              status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
              if(status.options.normalization && !mv.all_orthogonal)
                Normalize(mv.weight_factor, R, mv.factor_T_factor, mv.layer_factors);
              
              // ---- Terminating condition ----
              if (status.rel_costFunction < status.options.threshold_error || status.ao_iter >= status.options.max_iter)
              {
                  gather_final_factors(tnsDims, R, mv, status);
                  if(grid_comm.rank() == 0)
                  {
                    Partensor()->Logger()->info("Processor 0 collected all {} factors.\n", TnsSize);
                    if(status.options.writeToFile)
                      writeFactorsToFile(status);
                  }
                  break;
              }

              if (status.options.acceleration)
              {
                mv.norm_factors = mv.layer_factors;
                // ---- Acceleration Step ----
                if (status.ao_iter > 1)
                  line_search_accel(grid_comm, mv, status);

                mv.old_factors = mv.norm_factors;
              }  

          } // end of outer while loop
        }

        /**
         * Implementation of CPDMPI factorization with default values in Options
         * and no initialized factors.
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
          Options          options = MakeOptions<Tensor_>(execution::openmpi_policy());
          Status           status(options);
          Member_Variables mv(R, status.options.proc_per_mode);
          
          // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);

          // extract dimensions from tensor
          Dimensions const &tnsDims = tnsX.dimensions();
          // produce estimate factors using uniform distribution with entries in [0,1].
          makeFactors(tnsDims, status.options.constraints, R, status.factors);
          
          compute_sub_dimensions(tnsDims, status, R, mv);
          // Normalize each layer_factor, compute status.frob_tns and status.f_value
          // Normalize(R, factor_T_factor, status.factors);
          // After factor normalization scatter to each processor a part of each factor.
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i] = status.factors[i].block(mv.displs_subTns[i][mv.fiber_rank[i]], 0,
                                                          mv.subTnsDims[i][mv.fiber_rank[i]],  static_cast<int>(R));
          }

          // Each processor takes a subtensor from tnsX
          mv.subTns = tnsX.slice(mv.subTns_offsets, mv.subTns_extents);
          als(grid_comm, tnsDims, R, mv, status);

          return status;
        }

        /**
         * Implementation of CPDMPI factorization with default values in Options
         * and no initialized factors.
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
                          std::size_t const R, 
                          Options     const &options)
        {
          Status           status(options);
          Member_Variables mv(R, status.options.proc_per_mode);
          
          // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);
          
          // extract dimensions from tensor
          Dimensions const &tnsDims = tnsX.dimensions();
          // produce estimate factors using uniform distribution with entries in [0,1].
          makeFactors(tnsDims, status.options.constraints, R, status.factors);
          
          compute_sub_dimensions(tnsDims, status, R, mv);
          // Normalize each layer_factor, compute status.frob_tns and status.f_value
          // Normalize(R, factor_T_factor, status.factors);
          // After factor normalization scatter to each processor a part of each factor.
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i] = status.factors[i].block(mv.displs_subTns[i][mv.fiber_rank[i]], 0,
                                                          mv.subTnsDims[i][mv.fiber_rank[i]],  static_cast<int>(R));
          }

          // Each processor takes a subtensor from tnsX
          mv.subTns = tnsX.slice(mv.subTns_offsets, mv.subTns_extents);

          switch ( status.options.method )
          {
            case Method::als:
            {
              als(grid_comm, tnsDims, R, mv, status);
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
         * Implementation of CPDMPI factorization with default values in Options
         * and no initialized factors.
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
          Options          options = MakeOptions<Tensor_>(execution::openmpi_policy());
          Status           status(options);
          Member_Variables mv(R, status.options.proc_per_mode);
          
          // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);

          // extract dimensions from tensor
          Dimensions const &tnsDims = tnsX.dimensions();
          status.factors            = factorsInit;
          
          compute_sub_dimensions(tnsDims, status, R, mv);
          // Normalize each layer_factor, compute status.frob_tns and status.f_value
          // Normalize(R, factor_T_factor, status.factors);
          // After factor normalization scatter to each processor a part of each factor.
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i] = status.factors[i].block(mv.displs_subTns[i][mv.fiber_rank[i]], 0,
                                                          mv.subTnsDims[i][mv.fiber_rank[i]],  static_cast<int>(R));
          }

          // Each processor takes a subtensor from tnsX
          mv.subTns = tnsX.slice(mv.subTns_offsets, mv.subTns_extents);

          als(grid_comm, tnsDims, R, mv, status);

          return status;
        }

        /**
         * Implementation of CPDMPI factorization with default values in Options
         * and no initialized factors.
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
                          std::size_t const  R, 
                          Options     const &options, 
                          MatrixArray const &factorsInit)
        {
          Status           status(options);
          Member_Variables mv(R, status.options.proc_per_mode);
          
          // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);
          
          // extract dimensions from tensor
          Dimensions const &tnsDims = tnsX.dimensions();
          status.factors            = factorsInit; 
          
          compute_sub_dimensions(tnsDims, status, R, mv);
          // Normalize each layer_factor, compute status.frob_tns and status.f_value
          // Normalize(R, factor_T_factor, status.factors);
          // After factor normalization scatter to each processor a part of each factor.
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i] = status.factors[i].block(mv.displs_subTns[i][mv.fiber_rank[i]], 0,
                                                          mv.subTnsDims[i][mv.fiber_rank[i]],  static_cast<int>(R));
          }

          // Each processor takes a subtensor from tnsX
          mv.subTns.resize(mv.subTns_extents);
          mv.subTns = tnsX.slice(mv.subTns_offsets, mv.subTns_extents);

          switch ( status.options.method )
          {
            case Method::als:
            {
              als(grid_comm, tnsDims, R, mv, status);
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
         * Implementation of CPDMPI factorization with default values in Options
         * and no initialized factors.
         * 
         * @tparam TnsSize   Order of input Tensor.
         * 
         * @param  tnsDims [in] @c Stl array containing the Tensor dimensions, whose
         *                      length must be same as the Tensor order.
         * @param  R       [in] The rank of decomposition.
         * @param  path    [in] The path where the tensor is located.
         * 
         * @returns An object of type @c Status with the results of the algorithm.
         */
        template <std::size_t TnsSize>
        Status operator()(std::array<int, TnsSize> const &tnsDims, 
                          std::size_t              const  R, 
                          std::string              const &path)
        {
          Options          options = MakeOptions<Tensor_>(execution::openmpi_policy());
          Status           status(options);
          Member_Variables mv(R, status.options.proc_per_mode);
          
          // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);

          // produce estimate factors using uniform distribution with entries in [0,1].
          makeFactors(tnsDims, status.options.constraints, R, status.factors);
          
          compute_sub_dimensions(tnsDims, status, R, mv);
          // Normalize each layer_factor, compute status.frob_tns and status.f_value
          // Normalize(R, factor_T_factor, status.factors);
          // After factor normalization scatter to each processor a part of each factor.
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i] = status.factors[i].block(mv.displs_subTns[i][mv.fiber_rank[i]], 0,
                                                          mv.subTnsDims[i][mv.fiber_rank[i]],  static_cast<int>(R));
          }

          // Each processor takes a subtensor from tnsX
          mv.subTns.resize(mv.subTns_extents);
          readTensor( path, tnsDims, mv.subTns_extents, mv.subTns_offsets, mv.subTns );
          als(grid_comm, tnsDims, R, mv, status);

          return status;
        }

        /**
         * Implementation of CPDMPI factorization with default values in Options
         * and no initialized factors.
         * 
         * @tparam TnsSize      Order of input Tensor.
         * 
         * @param  tnsDims [in] @c Stl array containing the Tensor dimensions, whose
         *                      length must be same as the Tensor order.     
         * @param  R       [in] The rank of decomposition.
         * @param  path    [in] The path where the tensor is located.
         * @param  options [in] The options that the user wishes to use.
         * 
         * @returns An object of type @c Status with the results of the algorithm.
         */
        template <std::size_t TnsSize>
        Status operator()(std::array<int, TnsSize> const &tnsDims, 
                          std::size_t              const  R, 
                          std::string              const &path,
                          Options                  const &options)
        {
          Status           status(options);
          Member_Variables mv(R, status.options.proc_per_mode);
          
          // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);

          // produce estimate factors using uniform distribution with entries in [0,1].
          makeFactors(tnsDims, status.options.constraints, R, status.factors);
          
          compute_sub_dimensions(tnsDims, status, R, mv);
          // Normalize each layer_factor, compute status.frob_tns and status.f_value
          // Normalize(R, factor_T_factor, status.factors);
          // After factor normalization scatter to each processor a part of each factor.
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i] = status.factors[i].block(mv.displs_subTns[i][mv.fiber_rank[i]], 0,
                                                          mv.subTnsDims[i][mv.fiber_rank[i]],  static_cast<int>(R));
          }

          mv.subTns.resize(mv.subTns_extents);
          readTensor( path, tnsDims, mv.subTns_extents, mv.subTns_offsets, mv.subTns );

          switch ( status.options.method )
          {
            case Method::als:
            {
              als(grid_comm, tnsDims, R, mv, status);
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
         * Implementation of CPDMPI factorization with default values in Options
         * and no initialized factors.
         * 
         * @tparam TnsSize      Order of input Tensor.
         * 
         * @param  tnsDims [in] @c Stl array containing the Tensor dimensions, whose
         *                      length must be same as the Tensor order.      
         * @param  R       [in] The rank of decomposition.
         * @param  paths   [in] An @c stl array containing paths for the Tensor to be 
         *                      factorized and after that the paths for the initialized 
         *                      factors. 
         * 
         * @returns An object of type @c Status with the results of the algorithm.
         */
        template <std::size_t TnsSize>
        Status operator()(std::array<int, TnsSize>           const &tnsDims, 
                          std::size_t                        const  R, 
                          std::array<std::string, TnsSize+1> const &paths)
        {
          Options          options = MakeOptions<Tensor_>(execution::openmpi_policy());
          Status           status(options);
          Member_Variables mv(R, status.options.proc_per_mode);
          
          // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);
          
          compute_sub_dimensions(tnsDims, status, R, mv);
          // Read initialized factors from files
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i] = Matrix(mv.subTnsDims[i][mv.fiber_rank[i]],static_cast<int>(R));
          
            read( paths[i+1], 
                  mv.subTnsDims[i][mv.fiber_rank[i]]*static_cast<int>(R),
                  mv.displs_subTns_R[i][mv.fiber_rank[i]], 
                  mv.layer_factors[i] );
          }

          // Normalize each layer_factor, compute status.frob_tns and status.f_value
          // Normalize(R, factor_T_factor, status.factors);
          // Each processor takes a subtensor from tnsX
          mv.subTns.resize(mv.subTns_extents);
          readTensor( paths[0], tnsDims, mv.subTns_extents, mv.subTns_offsets, mv.subTns );
          als(grid_comm, tnsDims, R, mv, status);

          return status;
        }

        /**
         * Implementation of CPDMPI factorization with default values in Options
         * and initialized factors.
         * 
         * @tparam TnsSize       Order of input Tensor.
         * 
         * @param  tnsDims  [in] @c Stl array containing the Tensor dimensions, whose
         *                       length must be same as the Tensor order.   
         * @param  R        [in] The rank of decomposition.
         * @param  paths    [in] An @c stl array containing paths for the Tensor to be 
         *                       factorized and after that the paths for the initialized 
         *                       factors.
         *  @param  options [in] The options that the user wishes to use.
         * 
         * @returns An object of type @c Status with the results of the algorithm.
         */
        template <std::size_t TnsSize>
        Status operator()(std::array<int, TnsSize>           const &tnsDims, 
                          std::size_t                        const  R, 
                          std::array<std::string, TnsSize+1> const &paths,
                          Options                            const &options)
        {
          Status           status(options);          
          Member_Variables mv(R, status.options.proc_per_mode);
          
          // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);
          
          compute_sub_dimensions(tnsDims, status, R, mv);
          // Read initialized factors from files
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i] = Matrix(mv.subTnsDims[i][mv.fiber_rank[i]],static_cast<int>(R));
          
            read( paths[i+1], 
                  mv.subTnsDims[i][mv.fiber_rank[i]]*static_cast<int>(R),
                  mv.displs_subTns_R[i][mv.fiber_rank[i]], 
                  mv.layer_factors[i] );
          }

          // Normalize each layer_factor, compute status.frob_tns and status.f_value
          // Normalize(R, factor_T_factor, status.factors);
          // Each processor takes a subtensor from tnsX
          mv.subTns.resize(mv.subTns_extents);
          readTensor( paths[0], tnsDims, mv.subTns_extents, mv.subTns_offsets, mv.subTns );
          
          switch ( status.options.method )
          {
            case Method::als:
            {
              als(grid_comm, tnsDims, R, mv, status);
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
         * Implementation of CPDMPI factorization with default values in Options
         * and initialized factors. 
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
        template <std::size_t TnsSize>
        Status operator()(std::array<int, TnsSize>         const &tnsDims, 
                          std::size_t                      const  R, 
                          std::array<std::string, TnsSize> const &true_paths, 
                          std::array<std::string, TnsSize> const &init_paths, 
                          Options                          const &options)
        {
          Status           status(options);          
          Member_Variables mv(R, status.options.proc_per_mode);
          
          // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);
          
          compute_sub_dimensions(tnsDims, status, R, mv);
          // Read initialized factors from files
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i] = Matrix(mv.subTnsDims[i][mv.fiber_rank[i]],static_cast<int>(R));
            mv.true_factors[i]  = Matrix(mv.subTnsDims[i][mv.fiber_rank[i]],static_cast<int>(R));
          
            read( init_paths[i], 
                  mv.subTnsDims[i][mv.fiber_rank[i]]*static_cast<int>(R),
                  mv.displs_subTns_R[i][mv.fiber_rank[i]], 
                  mv.layer_factors[i] );

            read( true_paths[i], 
                  mv.subTnsDims[i][mv.fiber_rank[i]]*static_cast<int>(R),
                  mv.displs_subTns_R[i][mv.fiber_rank[i]], 
                  mv.true_factors[i] );
          }

          // Normalize each layer_factor, compute status.frob_tns and status.f_value
          // Normalize(R, factor_T_factor, status.factors);
          switch ( status.options.method )
          {
            case Method::als:
            {
              als_true_factors(grid_comm, tnsDims, R, mv, status);
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
