#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      14/11/2019
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
* @file      GtcMpi.hpp
* @details
* Implements the Canonical Polyadic Decomposition(gtc) using @c MPI.
* Make use of @c spdlog library in order to write output in
* a log file in @c "../log".
********************************************************************/

#if !defined(PARTENSOR_GTC_HPP)
#error "GTC_MPI can only included inside GTC"
#endif /* PARTENSOR_GTC_HPP */

namespace partensor
{

  inline namespace v1 {

    namespace internal {
      /**
       * Includes the implementation of GTCMPI factorization. Based on the given
       * parameters one of the four overloaded operators will be called.
       * @tparam Tensor_ The Type of The given @c Tensor to be factorized.
       */
      template<std::size_t TnsSize_>
      struct GTC<TnsSize_,execution::openmpi_policy> : public GTC_Base<TnsSize_>
      {
        using          GTC_Base<TnsSize_>::TnsSize;
        using          GTC_Base<TnsSize_>::lastFactor;
        using typename GTC_Base<TnsSize_>::Dimensions;
        using typename GTC_Base<TnsSize_>::MatrixArray;
        using typename GTC_Base<TnsSize_>::DataType;
        using typename GTC_Base<TnsSize_>::SparseTensor;
        using typename GTC_Base<TnsSize_>::LongMatrix;

        using IntArray         = typename SparseTensorTraits<SparseTensor>::IntArray; /**< Stl array of size TnsSize and containing int type. */

        using CartCommunicator = partensor::cartesian_communicator; // From ParallelWrapper.hpp
        using CartCommVector   = std::vector<CartCommunicator>;
        using IntVector        = std::vector<int>;
        using Int2DVector      = std::vector <std::vector<int>>;
        
        using Options = partensor::SparseOptions<TnsSize_,execution::openmpi_policy,SparseDefaultValues>;
        using Status  = partensor::SparseStatus<TnsSize_,execution::openmpi_policy,SparseDefaultValues>;
        
        // Variables that will be used in gtc implementations. 
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
          IntArray        tnsDims;         

          MatrixArray     layer_factors;
          MatrixArray     layer_factors_T;
          MatrixArray     factors;
          MatrixArray     factors_T;
          MatrixArray     factor_T_factor;
          MatrixArray     local_factors_T;
          MatrixArray     norm_factors_T;
          MatrixArray     old_factors_T;

          Matrix          cwise_factor_product;
          Matrix          temp_matrix;
          Matrix          Ratings_Base_T;
          SparseTensor    subTns;    

          int             rank;
          std::array<std::array<int, TnsSize-1>, TnsSize> offsets;  
          
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
          Member_Variables(int R, IntArray dims, std::array<int, TnsSize> &procs) :  local_f_value(0.0),
                                                                                     RxR(R*R),
                                                                                     displs_subTns(TnsSize),
                                                                                     displs_subTns_R(TnsSize),
                                                                                     subTnsDims(TnsSize),
                                                                                     subTnsDims_R(TnsSize),
                                                                                     displs_local_update(TnsSize),
                                                                                     send_recv_counts(TnsSize),
                                                                                     tnsDims(dims),
                                                                                     rank(R)
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

        template<int mode>
        void sort_ratings_base_util(Matrix           const &Ratings_Base_T,
                                    long int         const  nnz,
                                    Member_Variables       &mv)
        {
          Matrix ratings_base_temp = Ratings_Base_T;
          std::vector<std::vector<double>> vectorized_ratings_base;
          vectorized_ratings_base.resize(nnz, std::vector<double>(TnsSize + 1));

          for (int rows = 0; rows < static_cast<int>(TnsSize) + 1; rows++)
          {
            for (int cols = 0; cols < nnz; cols++)
            {
              vectorized_ratings_base[cols][rows] = Ratings_Base_T(rows, cols);
            }
          }

          // Sort
          std::sort(vectorized_ratings_base.begin(), vectorized_ratings_base.end(), SortRows<TnsSize_, mode, double>);
          
          for (int rows = 0; rows < static_cast<int>(TnsSize) + 1; rows++)
          {
            for (int cols = 0; cols < nnz; cols++)
            {
              ratings_base_temp(rows, cols) = vectorized_ratings_base[cols][rows];
            }
          }
      
          Dist_NNZ_sorted<TnsSize>(mv.subTns, nnz, mv.displs_subTns, mv.fiber_rank, ratings_base_temp, mv.subTnsDims, mode);
          mv.subTns[mode].makeCompressed();
          
          if constexpr (mode+1 < TnsSize)
            sort_ratings_base_util<mode+1>(Ratings_Base_T, nnz, mv);
        }

        void sort_ratings_base(Matrix           const &Ratings_Base_T,
                               long int         const  nnz,
                               Member_Variables       &mv)
        {
          ReserveSparseTensor<TnsSize>(mv.subTns, mv.subTnsDims, mv.fiber_rank, mv.world_size, nnz);
          sort_ratings_base_util<0>(Ratings_Base_T, nnz, mv);
        }

        void NesterovMNLS(Member_Variables                   &mv, 
                          Status                      const  &st,
                          int    const                        idx, 
                          Matrix                             &MTTKRP_T)
        {
          double L, mu, q, alpha, new_alpha, beta, lambda;
          int iter = 0;
          
          Matrix grad_Y_T(mv.rank, mv.subTnsDims[idx][mv.fiber_rank[idx]]);
          Matrix grad_Y_local_T(mv.rank, mv.rows_for_update[idx]);

          Matrix Y_T(mv.rank, mv.subTnsDims[idx][mv.fiber_rank[idx]]);
          Matrix Y_local_T(mv.rank, mv.rows_for_update[idx]);

          Matrix new_A(mv.rank, mv.rows_for_update[idx]);
          Matrix A(mv.rank, mv.rows_for_update[idx]);
          Matrix Zero_Matrix = Matrix::Zero(mv.rank, mv.rows_for_update[idx]);

          ComputeEIG(mv.cwise_factor_product, L, mu);

          lambda = st.options.lambdas[idx];
          L      = L + lambda;
          q      = lambda / L;
          alpha  = 1;

          A         = mv.local_factors_T[idx];
          Y_T       = mv.layer_factors_T[idx]; // layer_factor
          Y_local_T = A;

          int last_mode = (idx == static_cast<int>(TnsSize) - 1) ? static_cast<int>(TnsSize) - 2 : static_cast<int>(TnsSize) - 1;

          Matrix temp_R_1(mv.rank, 1);
          while (1)
          {
            grad_Y_T.setZero();

            if (iter >= st.options.max_nesterov_iter)
            {
              break;
            }
            
            // Compute grad_Y
            Matrix temp_col = Matrix::Zero(mv.rank, 1);
            for (long int i = 0; i < mv.subTns[idx].outerSize(); ++i)
            {
              temp_col.setZero();
              for (SparseMatrix::InnerIterator it(mv.subTns[idx], i); it; ++it)
              {
                temp_R_1 = Matrix::Ones(mv.rank, 1);
                // Select rows of each factor an compute the respective row of the Khatri-Rao product.
                for (int mode_i = last_mode, kr_counter = static_cast<int>(TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
                {
                  if (mode_i == idx)
                  {
                    continue;
                  }
                  int row;
                  row      = ((it.row()) / mv.offsets[idx][kr_counter]) % mv.subTnsDims[mode_i][mv.fiber_rank[mode_i]];
                  temp_R_1 = temp_R_1.cwiseProduct(mv.layer_factors_T[mode_i].col(row));
                  kr_counter--;
                }
                // Computation of row of Z according the relation (10) of the paper.
                temp_col += (temp_R_1.transpose() * Y_T.col(i))(0) * temp_R_1;
              }
              grad_Y_T.col(i) = temp_col;
            }

            // Add each process' results and scatter the block rows among the processes in the layer.
            // MPI_Reduce_scatter(grad_Y_T.data(), grad_Y_local_T.data(), send_recv_counts, MPI_DOUBLE, MPI_SUM, mode_layer_comm);
            v2::reduce_scatter( mv.layer_comm[idx], 
                                grad_Y_T, 
                                mv.send_recv_counts[idx][0],
                                grad_Y_local_T );

            // Add proximal term.
            grad_Y_local_T += MTTKRP_T + lambda * Y_local_T;
            new_A          = (Y_local_T - grad_Y_local_T / L).cwiseMax(Zero_Matrix);
            
            new_alpha = UpdateAlpha(alpha, q);
            beta      = alpha * (1 - alpha) / (alpha * alpha + new_alpha);

            Y_local_T = (1 + beta) * new_A - beta * A;

            // The updated block rows of Y are all gathered, and we have the whole updated Y of the layer.
            // MPI_Allgatherv(Y_local_T.data(), send_recv_counts_layer, MPI_DOUBLE, Y_T.data(), send_recv_counts, displs, MPI_DOUBLE, mode_layer_comm); // Communication through layer
            v2::all_gatherv( mv.layer_comm[idx], 
                            Y_local_T,
                            mv.send_recv_counts[idx][mv.layer_rank[idx]], 
                            mv.send_recv_counts[idx][0],
                            mv.displs_local_update[idx][0],
                            Y_T );

            A     = new_A;
            alpha = new_alpha;
            iter++;
          }
          mv.local_factors_T[idx] = A;
          
        }

        void NesterovMNLS_localL(Member_Variables                   &mv, 
                                 Status                      const  &st,
                                 int    const                        idx, 
                                 Matrix                             &MTTKRP_T)
        {
          int iter = 0;
				  double L2;
				  double sqrt_q = 0, beta = 0;
          double lambda = st.options.lambdas[idx];

			  	Matrix inv_L2(mv.subTns[idx].outerSize(), 1); // rows_layer
          
          Matrix grad_Y_T(mv.rank, mv.subTnsDims[idx][mv.fiber_rank[idx]]);
          Matrix grad_Y_local_T(mv.rank, mv.rows_for_update[idx]);

          Matrix Y_T(mv.rank, mv.subTnsDims[idx][mv.fiber_rank[idx]]);
          Matrix Y_local_T(mv.rank, mv.rows_for_update[idx]);

          Matrix new_A(mv.rank, mv.rows_for_update[idx]);
          Matrix A(mv.rank, mv.rows_for_update[idx]);

				 const Matrix zero_vec = Matrix::Zero(mv.rank, 1);

          A         = mv.local_factors_T[idx];
          Y_T       = mv.layer_factors_T[idx]; // layer_factor
          Y_local_T = A;

          int last_mode = (idx == static_cast<int>(TnsSize) - 1) ? static_cast<int>(TnsSize) - 2 : static_cast<int>(TnsSize) - 1;

          Matrix temp_R_1(mv.rank, 1);
				  Matrix temp_RxR(mv.rank, mv.rank);

          while (1)
          {
            grad_Y_T.setZero();

            if (iter >= st.options.max_nesterov_iter)
            {
              break;
            }
            
            // Compute grad_Y
            Matrix temp_col = Matrix::Zero(mv.rank, 1);
            for (long int i = 0; i < mv.subTns[idx].outerSize(); ++i)
            {
              temp_col.setZero();
              if (iter < 1)
              {
                temp_RxR.setZero();
              }
              for (SparseMatrix::InnerIterator it(mv.subTns[idx], i); it; ++it)
              {
                temp_R_1 = Matrix::Ones(mv.rank, 1);
                // Select rows of each factor an compute the respective row of the Khatri-Rao product.
                for (int mode_i = last_mode, kr_counter = static_cast<int>(TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
                {
                  if (mode_i == idx)
                  {
                    continue;
                  }
                  long long int row;
                  row      = ((it.row()) / mv.offsets[idx][kr_counter]) % mv.subTnsDims[mode_i][mv.fiber_rank[mode_i]];
                  temp_R_1 = temp_R_1.cwiseProduct(mv.layer_factors_T[mode_i].col(row));
                  kr_counter--;
                }
                // Computation of row of Z according the relation (10) of the paper.
                temp_col += (temp_R_1.transpose() * Y_T.col(i))(0) * temp_R_1;

                // Compute only once!
                if (iter < 1)
                {
                  temp_RxR.noalias() += (temp_R_1 * temp_R_1.transpose());
                }
              }
              grad_Y_T.col(i) = temp_col;

              if (iter < 1)
              {
                // Communicate only once!
                all_reduce( mv.layer_comm[idx], 
                            inplace(temp_RxR.data()), 
                            mv.RxR,
                            std::plus<double>() ); 

                L2 = PowerMethod(temp_RxR, 1e-3);
                L2 += lambda;
                inv_L2(i) = 1 / L2;
              }
            }

            // Add each process' results and scatter the block rows among the processes in the layer.
            // MPI_Reduce_scatter(grad_Y_T.data(), grad_Y_local_T.data(), send_recv_counts, MPI_DOUBLE, MPI_SUM, mode_layer_comm);
            v2::reduce_scatter( mv.layer_comm[idx], 
                                grad_Y_T, 
                                mv.send_recv_counts[idx][0],
                                grad_Y_local_T );

            // Add proximal term.
            grad_Y_local_T += MTTKRP_T + lambda * Y_local_T;

            for (long int i=0; i<mv.rows_for_update[idx]; i++)
            {
              long int translate_i = i + mv.displs_local_update[idx][mv.layer_rank[idx]]/mv.rank;
              
              new_A.col(i) = (Y_local_T.col(i) - grad_Y_local_T.col(i) * inv_L2(translate_i)).cwiseMax(zero_vec);
              
              sqrt_q = sqrt( lambda * inv_L2(translate_i) );
              beta = (1 - sqrt_q) / (1 + sqrt_q);

              // Update Y
              Y_local_T.col(i) = (1 + beta) * new_A.col(i) - beta * A.col(i);
            }

            // The updated block rows of Y are all gathered, and we have the whole updated Y of the layer.
            v2::all_gatherv( mv.layer_comm[idx], 
                            Y_local_T,
                            mv.send_recv_counts[idx][mv.layer_rank[idx]], 
                            mv.send_recv_counts[idx][0],
                            mv.displs_local_update[idx][0],
                            Y_T );

            A = new_A;
            
            iter++;
          }
          mv.local_factors_T[idx] = A;
        }

        /*
         * In case option variable @c writeToFile is enabled then, before the end
         * of the algorithm writes the resulted factors in files, where their
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
         * @param  grid_comm [in]     MPI communicator where the new cost function value
         *                            will be communicated and computed.
         * @param  mv        [in]     Struct where ALS variables are stored.
         * @param  st        [in,out] Struct where the returned values of @c Gtc are stored.
         *                            In this case the cost function value is updated.
         */
        void cost_function( CartCommunicator const &grid_comm,
                            Member_Variables       &mv,
                            Status                 &st )
        {
            Matrix temp_R_1(mv.rank, 1);
            double temp_1_1  = 0;
            mv.local_f_value = 0;
            std::array<int, TnsSize-1> offsets;
            offsets[0] = 1;
            for (int j = 1; j < static_cast<int>(TnsSize) - 1; j++)
            {
              offsets[j] = offsets[j - 1] * mv.subTnsDims[j-1][mv.fiber_rank[j-1]]; // mv.layer_factors_T[j - 1].cols()
            }

            for (long int i = 0; i < mv.subTns[lastFactor].outerSize(); ++i)
            {
                int row;
                for (SparseMatrix::InnerIterator it(mv.subTns[lastFactor], i); it; ++it)
                {
                    temp_R_1 = mv.layer_factors_T[lastFactor].col(it.col());
                    // Select rows of each factor an compute the Hadamard product of the respective row of the Khatri-Rao product, and the row of factor A_N.
                    // temp_R_1  = A_N(i_N,:) .* ... .* A_2(i_2,:) .* A_1(i_1,:) 
                    for (int mode_i = static_cast<int>(TnsSize) - 2; mode_i >= 0; mode_i--)
                    {	
                        row = ((it.row()) / offsets[mode_i]) % (mv.subTnsDims[mode_i][mv.fiber_rank[mode_i]]);
                        temp_R_1.noalias() = temp_R_1.cwiseProduct(mv.layer_factors_T[mode_i].col(row));
                    }
                    temp_1_1 = it.value() - temp_R_1.sum();
                    mv.local_f_value += temp_1_1 * temp_1_1;
                }
            }

            all_reduce( grid_comm, 
                        mv.local_f_value,
                        st.f_value, 
                        std::plus<double>() );
        }

        /*
         * Compute the cost function value at the end of each outer iteration
         * based on the last accelerated factor.
         * 
         * @param  grid_comm         [in] MPI communicator where the new cost function value
         *                                will be communicated and computed.
         * @param  mv                [in] Struct where ALS variables are stored.
         * @param  st                [in] Struct where the returned values of @c Gtc are stored.
         *                                In this case the cost function value is updated.
         * @param  factors           [in] Accelerated factors.
         * @param  factors_T_factors [in] Gramian matrices of factors.
         * 
         * @returns The cost function calculated with the accelerated factors.
         */
        double accel_cost_function(CartCommunicator const &grid_comm,
                                   Member_Variables const &mv,
                                   MatrixArray      const &layer_factors_T)
        {
          Matrix temp_R_1(mv.rank, 1);
          double temp_1_1 = 0;
          double f_value = 0;

          std::array<int, TnsSize-1> offsets;
          offsets[0] = 1;
          for (int j = 1; j < static_cast<int>(TnsSize) - 1; j++)
          {
            offsets[j] = offsets[j - 1] * mv.subTnsDims[j-1][mv.fiber_rank[j-1]];
          }

          for (long int i = 0; i < mv.subTns[lastFactor].outerSize(); ++i)
          {
              int row;
              for (SparseMatrix::InnerIterator it(mv.subTns[lastFactor], i); it; ++it)
              {
                  temp_R_1 = layer_factors_T[lastFactor].col(it.col());
                  // Select rows of each factor an compute the Hadamard product of the respective row of the Khatri-Rao product, and the row of factor A_N.
                  // temp_R_1  = A_N(i_N,:) .* ... .* A_2(i_2,:) .* A_1(i_1,:) 
                  for (int mode_i = static_cast<int>(TnsSize) - 2; mode_i >= 0; mode_i--)
                  {	
                      row = ((it.row()) / offsets[mode_i]) % (mv.subTnsDims[mode_i][mv.fiber_rank[mode_i]]);
                      temp_R_1.noalias() = temp_R_1.cwiseProduct(layer_factors_T[mode_i].col(row));
                  }
                  temp_1_1 = it.value() - temp_R_1.sum();
                  f_value += temp_1_1 * temp_1_1;
              }
          }

          all_reduce( grid_comm, 
                      inplace(&f_value),
                      1, 
                      std::plus<double>() );

          return f_value;
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
         * @param  st         [in]     Struct where the returned values of @c Gtc are stored.
         * @param  R          [in]     The rank of decomposition.
         * @param  mv         [in,out] Struct where ALS variables are stored. 
         *                             Updates @c stl arrays with dimensions for subtensors and
         *                             subfactors.
         */
        void compute_sub_dimensions(Status           const &st,
                                    Member_Variables       &mv)
        {
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.factor_T_factor[i].noalias() = st.factors[i].transpose() * st.factors[i];

            DisCount(mv.displs_subTns[i], mv.subTnsDims[i], st.options.proc_per_mode[i], mv.tnsDims[i], 1);
            // for fiber communication and Gatherv
            DisCount(mv.displs_subTns_R[i], mv.subTnsDims_R[i], st.options.proc_per_mode[i], mv.tnsDims[i], static_cast<int>(mv.rank));
            // information per layer
            DisCount(mv.displs_local_update[i], mv.send_recv_counts[i], mv.world_size / st.options.proc_per_mode[i], 
                                                  mv.subTnsDims[i][mv.fiber_rank[i]], static_cast<int>(mv.rank));

            mv.rows_for_update[i] = mv.send_recv_counts[i][mv.layer_rank[i]] / static_cast<int>(mv.rank);
          }

          calculate_offsets(mv);
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
              mv.offsets[idx][j] = mv.offsets[idx][j - 1] * mv.subTnsDims[mode][mv.fiber_rank[mode]];
            }
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
         * @param  st  [in]     Struct where the returned values of @c Gtc are stored.
         *                      Here constraints and options variables are needed.
         * @param  mv  [in,out] Struct where ALS variables are stored.
         *                      Updates the factors of each layer.
         */
        void update_factor(int              const  idx,
                           Status           const &st, 
                           Member_Variables       &mv  )
        {
          switch ( st.options.constraints[idx] ) 
          {
            case Constraint::unconstrained:
            case Constraint::symmetric:
            {
              // std::cout << " Inside symmetric update factor ... " << std::endl;
              Matrix A(mv.rank, mv.subTnsDims[idx][mv.fiber_rank[idx]]);
              Matrix A_local(mv.rank, mv.rows_for_update[idx]);
              Matrix eye = st.options.lambdas[idx] * Matrix::Identity(mv.rank, mv.rank);

              // int first_mode = (idx == 0) ? 1 : 0;
              int last_mode = (idx == static_cast<int>(TnsSize) - 1) ? static_cast<int>(TnsSize) - 2 : static_cast<int>(TnsSize) - 1;

              Matrix MTTKRP_col(mv.rank, 1);
              Matrix temp_RxR(mv.rank, mv.rank);
              Matrix temp_R_1(mv.rank, 1);

              // Compute MTTKRP
              for (long long int i = 0; i < mv.subTns[idx].outerSize(); ++i)
              {
                MTTKRP_col.setZero();
                temp_RxR.setZero(); // temp_RxR : is the Hadamard product of Grammians of the Factors, that correspond to the nnz elements of the Tensor.
                for (SparseMatrix::InnerIterator it(mv.subTns[idx], i); it; ++it)
                {
                  temp_R_1 = Matrix::Ones(mv.rank, 1);
                  long long int row;
                  // Select rows of each factor an compute the respective row of the Khatri-Rao product.
                  for (int mode_i = last_mode, kr_counter = static_cast<int>(TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
                  {
                    if (mode_i == idx)
                    {
                      continue;
                    }
                    row      = ((it.row()) / mv.offsets[idx][kr_counter]) % (mv.subTnsDims[mode_i][mv.fiber_rank[mode_i]]);

                    temp_R_1 = temp_R_1.cwiseProduct(mv.layer_factors_T[mode_i].col(row));
                    kr_counter--;
                  }
                  // Subtract from the previous row the respective row of W, according to relation (9).
                  MTTKRP_col.noalias() += it.value() * temp_R_1;                   

                  temp_RxR.noalias()   += temp_R_1 * temp_R_1.transpose();
                }
                
                all_reduce( mv.layer_comm[idx], 
                            inplace(MTTKRP_col.data()), 
                            1 * mv.rank,
                            std::plus<double>() ); 
                            
                all_reduce( mv.layer_comm[idx], 
                            inplace(temp_RxR.data()), 
                            mv.RxR,
                            std::plus<double>() ); 

                A.col(i) = ((temp_RxR + eye).inverse()) * MTTKRP_col;
              }
              // std::cout << " Inside symmetric update factor After loop  ... " << std::endl;

              mv.local_factors_T[idx] = A.block(0, mv.displs_local_update[idx][mv.layer_rank[idx]] / mv.rank, mv.rank,  mv.rows_for_update[idx]);

              break;
            }
            case Constraint::nonnegativity:
            case Constraint::symmetric_nonnegativity:
            {
              int last_mode = (idx == static_cast<int>(TnsSize) - 1) ? static_cast<int>(TnsSize) - 2 : static_cast<int>(TnsSize) - 1;

              Matrix local_MTTKRP_T = Matrix::Zero(mv.rank, mv.rows_for_update[idx]);
              Matrix MTTKRP_T       = Matrix::Zero(mv.rank, mv.subTnsDims[idx][mv.fiber_rank[idx]]);

              // Compute MTTKRP
              SparseMTTKRP<TnsSize>(mv.subTnsDims, mv.fiber_rank, mv.subTns[idx], mv.layer_factors_T, mv.rank, mv.offsets[idx], last_mode, idx, MTTKRP_T);

              // Add each process' results and scatter the block rows among the processes in the layer.
              // MPI_Reduce_scatter(MTTKRP_T.data(), local_MTTKRP_T.data(), send_recv_counts, MPI_DOUBLE, MPI_SUM, mode_layer_comm);
              v2::reduce_scatter( mv.layer_comm[idx], 
                                  MTTKRP_T, 
                                  mv.send_recv_counts[idx][0],
                                  local_MTTKRP_T );

              // NesterovMNLS(mv, st, idx, local_MTTKRP_T);
              NesterovMNLS_localL(mv, st, idx, local_MTTKRP_T);

              break;
            }
            case Constraint::sparsity:
              break;
            default: // in case of Constraint::constant
              break;
          } // end of constraints switch

          // std::cout << " Inside update factor After switch  ... " << std::endl;
          if (st.options.constraints[idx] != Constraint::constant)
          {               
            v2::all_gatherv( mv.layer_comm[idx], 
                             mv.local_factors_T[idx],
                             mv.send_recv_counts[idx][mv.layer_rank[idx]], 
                             mv.send_recv_counts[idx][0],
                             mv.displs_local_update[idx][0],
                             mv.layer_factors_T[idx] );
            
            mv.layer_factors[idx]             = mv.layer_factors_T[idx].transpose();

            if (st.options.constraints[idx] == Constraint::symmetric_nonnegativity || st.options.constraints[idx] == Constraint::symmetric)
            {
              for (std::size_t i=0; i<TnsSize; i++)
              {
                if (i != static_cast<std::size_t>(idx))
                {
                  mv.layer_factors_T[i] = mv.layer_factors_T[idx];
                }
              }
            }

            mv.factor_T_factor[idx].noalias() = mv.layer_factors_T[idx] * mv.layer_factors[idx];
          }


          // std::cout << " Inside update factor .... switch  ... " << std::endl;

          all_reduce( mv.fiber_comm[idx], 
                      inplace(mv.factor_T_factor[idx].data()), 
                      mv.RxR,
                      std::plus<double>() ); 
        }

        /*
         * At the end of the algorithm processor 0
         * collects each part of the factor that each
         * processor holds and return them in status.factors.
         *
         * @param  mv      [in]     Struct where ALS variables are stored.
         *                          Use variables to compute result factors by gathering each 
         *                          part of the factor from processors. 
         * @param  st      [in,out] Struct where the returned values of @c Gtc are stored.
         *                          Stores the resulted factors.
         */
        void gather_final_factors(Member_Variables       &mv,
                                  Status                 &st)
        {
          for(std::size_t i=0; i<TnsSize; ++i)
          {
            mv.temp_matrix.resize(static_cast<int>(mv.rank), mv.tnsDims[i]);
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
         * @param  st        [in,out] Struct where the returned values of @c Gtc are stored.
         *                            If the acceleration succeeds updates cost function value. 
         * 
         */
        void line_search_accel(CartCommunicator const &grid_comm,
                               Member_Variables       &mv,
                               Status                 &st)
        {
          double       f_accel    = 0.0; // Objective Value after the acceleration step
          double const accel_step = pow(st.ao_iter+1,(1.0/(st.options.accel_coeff)));
          
          MatrixArray   accel_factors_T;
          MatrixArray   accel_gramians;

          for(std::size_t i=0; i<TnsSize; ++i)
          {
            accel_factors_T[i] = mv.old_factors_T[i] + accel_step * (mv.layer_factors_T[i] - mv.old_factors_T[i]); 
            accel_gramians[i]  = accel_factors_T[i] * accel_factors_T[i].transpose();
            all_reduce( mv.fiber_comm[i], 
                        inplace(accel_gramians[i].data()),
                        mv.RxR, 
                        std::plus<double>() );    
          }

          f_accel = accel_cost_function(grid_comm, mv, accel_factors_T);
          if (st.f_value > f_accel)
          {
            mv.layer_factors_T = accel_factors_T;
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
         * @param  status    [in,out] Struct where the returned values of @c Gtc are stored.
         */
        void aogtc(CartCommunicator const &grid_comm,
                   Member_Variables       &mv,
                   Status                 &status)
        {
          status.frob_tns = (mv.subTns[0]).squaredNorm();
	        all_reduce( grid_comm, 
                      inplace(&status.frob_tns),
                      1, 
                      std::plus<double>() );
          
          cost_function(grid_comm, mv, status);
          status.rel_costFunction = status.f_value / status.frob_tns;

          for (int i=0; i< static_cast<int>(TnsSize); i++)
          {
            mv.factor_T_factor[i].noalias() = mv.layer_factors_T[i] * mv.layer_factors[i];
            all_reduce( mv.fiber_comm[i], 
                        inplace(mv.factor_T_factor[i].data()), 
                        mv.RxR,
                        std::plus<double>() ); 	
          }

          // Wait for all processors to reach here
          grid_comm.barrier();

          // ---- Loop until ALS converges ----
          while(1) 
          {
              status.ao_iter++;
              if (!grid_comm.rank())
              {
                  Partensor()->Logger()->info("iter: {} -- fvalue: {} -- relative_costFunction: {}", status.ao_iter, 
                                                  status.f_value, status.rel_costFunction);

                  std::cout << "iter: " << status.ao_iter << " - status.f_value: " << status.f_value << " - status.rel_costFunction: " << status.rel_costFunction << std::endl;
              }
                                                  
              //	<-----------------------	loop for every mode		--------------------------->	//              
              for (std::size_t i = 0; i < TnsSize; i++) 
              {
                // Compute hadamard of grammians to compute L.
                mv.cwise_factor_product = partensor::PartialCwiseProd(mv.factor_T_factor, i);


                // Partition rows of subfactor to the processes in the respective layer.
                mv.local_factors_T[i] = mv.layer_factors_T[i].block(0, mv.displs_local_update[i][mv.layer_rank[i]] / mv.rank, mv.rank, mv.rows_for_update[i]);
                

                update_factor(i, status, mv);
                
              }

              // ---- Cost function Computation ----
              cost_function(grid_comm, mv, status);

              status.rel_costFunction = status.f_value / status.frob_tns;

              // if(status.options.normalization && !mv.all_orthogonal)
              //   Normalize(mv.weight_factor, mv.rank, mv.factor_T_factor, mv.layer_factors);
              
              // ---- Terminating condition ----
              if (status.ao_iter >= status.options.max_iter)
              {
                  gather_final_factors(mv, status);
                  if(grid_comm.rank() == 0)
                  {
                    std::cout << "status.rel_costFunction : " << status.rel_costFunction << std::endl;
                    Partensor()->Logger()->info("Processor 0 collected all {} factors.\n", TnsSize);
                    if(status.options.writeToFile)
                      writeFactorsToFile(status);
                  }
                  break;
              }

              if (status.options.acceleration)
              {
                mv.norm_factors_T = mv.layer_factors_T;
                // ---- Acceleration Step ----
                if (status.ao_iter > 1)
                  line_search_accel(grid_comm, mv, status);

                mv.old_factors_T = mv.norm_factors_T;
              }  

          } // end of outer while loop
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
          else 
            makeFactors(mv.tnsDims, status.options.constraints, mv.rank, status.factors);
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
        Status operator()(Options const &options)
        {
          Status           status(options);
          Member_Variables mv(options.rank, options.tnsDims, status.options.proc_per_mode);
         
         // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 
          
          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);          

          // produce estimate factors using uniform distribution with entries in [0,1].
          initialize_factors(mv, status);

          compute_sub_dimensions(status, mv);

          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i]             = status.factors[i].block(mv.displs_subTns[i][mv.fiber_rank[i]], 0,
                                                          mv.subTnsDims[i][mv.fiber_rank[i]],  mv.rank);
            mv.layer_factors_T[i]           = mv.layer_factors[i].transpose();
            mv.factor_T_factor[i].noalias() = mv.layer_factors_T[i] * mv.layer_factors[i];
          }

          long long int fileSize = (TnsSize + 1) * options.nonZeros;

          // Matrix Ratings_Base = Matrix(options.nonZeros, static_cast<int>(TnsSize+1));
          Matrix Ratings_Base_T = Matrix(static_cast<int>(TnsSize+1), options.nonZeros);
         
          // Read the whole Tensor from a file
          read( options.ratings_path, 
                fileSize, 
                0, 
                Ratings_Base_T );
          
          // Matrix Ratings_Base_T = Ratings_Base.transpose();

          sort_ratings_base(Ratings_Base_T, options.nonZeros, mv);
	        Ratings_Base_T.resize(0,0);
          // Ratings_Base.resize(0,0);

          aogtc(grid_comm, mv, status);

          return status;
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
        Status operator()(Matrix       const &Ratings_Base_T, 
                          Options      const &options)
        {
          Status           status(options);
          Member_Variables mv(options.rank, options.tnsDims, status.options.proc_per_mode);
         
         // Communicator with cartesian topology
          CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 
          
          // Functions that create layer and fiber grids.
          create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
          create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);          

          // produce estimate factors using uniform distribution with entries in [0,1].
          initialize_factors(mv, status);

          compute_sub_dimensions(status, mv);

          // Begin Load Balancing
	        // Matrix                                     Balanced_Ratings_Base_T(TnsSize + 1, options.nonZeros);
	        // std::array<std::vector<long int>, TnsSize> perm_tns_indices;

          // BalanceDataset<TnsSize>(options.nonZeros, options.tnsDims, Ratings_Base_T, perm_tns_indices, Balanced_Ratings_Base_T);

          // PermuteFactors<TnsSize>(status.factors, perm_tns_indices, mv.factors_T);
          
          // std::cout << "After Balance........" << std::endl;
          for (std::size_t i = 0; i < TnsSize; ++i) 
          {
            mv.layer_factors[i]             = status.factors[i].block(mv.displs_subTns[i][mv.fiber_rank[i]], 0,
                                                          mv.subTnsDims[i][mv.fiber_rank[i]],  mv.rank);
            mv.layer_factors_T[i]           = mv.layer_factors[i].transpose();
            mv.factor_T_factor[i].noalias() = mv.layer_factors_T[i] * mv.layer_factors[i];

            // mv.layer_factors_T[i] = mv.factors_T[i].block(0, mv.displs_subTns[i][mv.fiber_rank[i]],
            //                                               mv.rank, mv.subTnsDims[i][mv.fiber_rank[i]]);
            // mv.layer_factors[i]           = mv.layer_factors_T[i].transpose();
            // mv.factor_T_factor[i].noalias() = mv.layer_factors_T[i] * mv.layer_factors[i];
          }

          // Each processor takes a subtensor from tnsX          
          ReserveSparseTensor<TnsSize>(mv.subTns, mv.subTnsDims, mv.fiber_rank, mv.world_size, options.nonZeros);
          
          Dist_NNZ<TnsSize>(mv.subTns, options.nonZeros, mv.displs_subTns, mv.fiber_rank, Ratings_Base_T, mv.subTnsDims);
          // Dist_NNZ<TnsSize>(mv.subTns, options.nonZeros, mv.displs_subTns, mv.fiber_rank, Balanced_Ratings_Base_T, mv.subTnsDims);
	        // Ratings_Base_T.resize(0,0);
	        // Balanced_Ratings_Base_T.resize(0,0);

          for(int mode_i = 0; mode_i < static_cast<int>(TnsSize); mode_i++)
          {
            mv.subTns[mode_i].makeCompressed();
          }

          aogtc(grid_comm, mv, status);

          return status;
        }

      };
    }  // namespace internal
  }    // namespace v1

} // end namespace partensor
