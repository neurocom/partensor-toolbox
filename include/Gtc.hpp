#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      07/10/2019
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
* @file      Gtc.hpp
* @details
* Implements the General Tensor Completion(gtc).
* Make use of @c spdlog library in order to write output in
* a log file in @c "../log". In case of using parallelism
* with mpi, then the functions from @c GtcMpi.hpp will be called.
********************************************************************/

#ifndef PARTENSOR_GTC_HPP
#define PARTENSOR_GTC_HPP

#include "PARTENSOR_basic.hpp"
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
      //Status gtc_f(ExecutionPolicy &&, Tensor const &tnsX, std::size_t rank);

      /*
       * Includes the implementation of General Tensor Completion. Based on the given
       * parameters one of the overloaded operators will be called.
       */
      template <std::size_t TnsSize_>
      struct GTC_Base 
      {
        static constexpr std::size_t TnsSize    = TnsSize_;                             /**< Sparse Tensor Order. */
        static constexpr std::size_t lastFactor = TnsSize - 1;                          /**< ID of the last factor. */
        
        using SparseTensor = typename partensor::SparseTensor<TnsSize_>;
        using DataType     = typename SparseTensorTraits<SparseTensor>::DataType;            /**< Sparse Tensor Data type. */
        using MatrixType   = typename SparseTensorTraits<SparseTensor>::MatrixType;          /**< Eigen Matrix with the same Data type with the Sparse Tensor. */
        using Dimensions   = typename SparseTensorTraits<SparseTensor>::Dimensions;          /**< Sparse Tensor Dimensions type. */
        using SparseMatrix = typename SparseTensorTraits<SparseTensor>::SparseMatrixType;    /**< Sparse Matrix type. */
        using LongMatrix   = typename SparseTensorTraits<SparseTensor>::LongMatrixType;      /**< Long Matrix type. */

        using Constraints  = typename SparseTensorTraits<SparseTensor>::Constraints;         /**< Stl array of size TnsSize and containing Constraint type. */
        using MatrixArray  = typename SparseTensorTraits<SparseTensor>::MatrixArray;         /**< Stl array of size TnsSize and containing MatrixType type. */
        using DoubleArray  = typename SparseTensorTraits<SparseTensor>::DoubleArray;         /**< Stl array of size TnsSize and containing double type. */
        using IntArray     = typename SparseTensorTraits<SparseTensor>::IntArray;            /**< Stl array of size TnsSize and containing double type. */

        template<int mode>
        void sort_ratings_base_util(Matrix       const &Ratings_Base_T, 
                                    SparseTensor       &tnsX,
                                    IntArray     const &tnsDims,
                                    long int     const  nnz)
        {
          Matrix ratings_base_temp = Ratings_Base_T;
          std::vector<std::vector<double>> vectorized_ratings_base;
          vectorized_ratings_base.resize(nnz,std::vector<double>(TnsSize + 1));

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

          FillSparseMatricization<TnsSize>(tnsX, nnz, ratings_base_temp, tnsDims, mode);

          if constexpr (mode+1 < TnsSize)
            sort_ratings_base_util<mode+1>(Ratings_Base_T, tnsX, tnsDims, nnz);
        }

        void sort_ratings_base(Matrix       const &Ratings_Base_T, 
                               SparseTensor       &tnsX,
                               IntArray     const &tnsDims,
                               long int     const  nnz)
        {
          ReserveSparseTensor<TnsSize>(tnsX, tnsDims, nnz);
          sort_ratings_base_util<0>(Ratings_Base_T, tnsX, tnsDims, nnz);
        }
      };

      template <std::size_t TnsSize_, typename ExecutionPolicy = execution::sequenced_policy>
      struct GTC : public GTC_Base<TnsSize_>
      {
        using          GTC_Base<TnsSize_>::TnsSize;
        using          GTC_Base<TnsSize_>::lastFactor;
        using typename GTC_Base<TnsSize_>::Dimensions;
        using typename GTC_Base<TnsSize_>::MatrixArray;
        using typename GTC_Base<TnsSize_>::DataType;
        using typename GTC_Base<TnsSize_>::SparseTensor;
        using typename GTC_Base<TnsSize_>::IntArray;
        using typename GTC_Base<TnsSize_>::LongMatrix;
        
        using Options = partensor::SparseOptions<TnsSize_,execution::sequenced_policy,SparseDefaultValues>;
        using Status  = partensor::SparseStatus<TnsSize_,execution::sequenced_policy,SparseDefaultValues>;

        // Variables that will be used in gtc implementations. 
        struct Member_Variables 
        {
          MatrixArray  factors_T;        
          MatrixArray  factor_T_factor;  
          MatrixArray  mttkrp_T;        
          IntArray     tnsDims;         
          std::array<std::array<int, TnsSize_ -1>, TnsSize_> offsets;         
          
          MatrixArray  norm_factors_T;
          MatrixArray  old_factors_T;

          Matrix       cwise_factor_product;
          SparseTensor tnsX;
          int          rank;

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
          st.f_value = 0;

          std::array<int,TnsSize-1> offsets;
          offsets[0] = 1;
          for (int j = 1; j < static_cast<int>(TnsSize) - 1; j++)
          {
            offsets[j] = offsets[j - 1] * mv.tnsDims[j-1];
          }

          for (long int i = 0; i < mv.tnsX[lastFactor].outerSize(); ++i)
          {
              int row = 0;
              for (SparseMatrix::InnerIterator it(mv.tnsX[lastFactor], i); it; ++it)
              {
                  temp_R_1 = mv.factors_T[lastFactor].col(it.col());
                  // Select rows of each factor an compute the Hadamard product of the respective row of the Khatri-Rao product, and the row of factor A_N.
                  for (int mode_i = static_cast<int>(TnsSize) - 2; mode_i >= 0; mode_i--)
                  {
                      row      = ((it.row()) / offsets[mode_i]) % (mv.tnsDims[mode_i]);
                      temp_R_1 = temp_R_1.cwiseProduct(mv.factors_T[mode_i].col(row));
                  }
                  temp_1_1 = it.value() - temp_R_1.sum();
                  st.f_value += temp_1_1 * temp_1_1;
              }
          }
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
                    row      = ((it.row()) / offsets[mode_i]) % (mv.tnsDims[mode_i]);
                    temp_R_1 = temp_R_1.cwiseProduct(accel_factors[mode_i].col(row));
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
            case Constraint::symmetric:
            {
              unconstraint_update(idx, mv, st);
              break;
            }
            case Constraint::nonnegativity:
            case Constraint::symmetric_nonnegativity:
            { 
              int last_mode = (idx == static_cast<int>(TnsSize) - 1) ? static_cast<int>(TnsSize) - 2 : static_cast<int>(TnsSize) - 1;
              SparseMTTKRP(mv.tnsDims, mv.tnsX[idx], mv.factors_T, mv.rank, mv.offsets[idx], last_mode, idx, mv.mttkrp_T[idx]);

              // NesterovMNLS(mv.cwise_factor_product, mv.factors_T, mv.tnsDims, mv.tnsX[idx], mv.offsets[idx], st.options.max_nesterov_iter, 
              //              st.options.lambdas[idx], idx, st.options.constraints[idx], mv.mttkrp_T[idx]);
              local_L::NesterovMNLS(mv.factors_T, mv.tnsDims, mv.tnsX[idx], mv.offsets[idx], st.options.max_nesterov_iter, 
                           st.options.lambdas[idx], idx, mv.mttkrp_T[idx]);
              break;
            }
            default: // in case of Constraint::constant
              break;
          }

          // Compute A^T * A + B^T * B + ...
          st.factors[idx] = mv.factors_T[idx].transpose();
          if (st.options.constraints[idx] == Constraint::symmetric_nonnegativity || st.options.constraints[idx] == Constraint::symmetric)
          {
            for (std::size_t i=0; i<TnsSize; i++)
            {
              if (i != static_cast<std::size_t>(idx))
              {
                mv.factors_T[i] = mv.factors_T[idx];
              }
            }
          }
          mv.factor_T_factor[idx].noalias() = mv.factors_T[idx] * st.factors[idx];
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
                               Status           &st)
        {
          double       f_accel    = 0.0; // Objective Value after the acceleration step
          double const accel_step = pow(st.ao_iter+1,(1.0/(st.options.accel_coeff)));

          MatrixArray  accel_factors_T;
          MatrixArray  accel_gramians;

          for(std::size_t i=0; i<TnsSize; ++i)
          {
            accel_factors_T[i] = mv.old_factors_T[i] + accel_step * (mv.factors_T[i] - mv.old_factors_T[i]); 
            accel_gramians[i]  = accel_factors_T[i] * accel_factors_T[i].transpose();
          }

          f_accel = accel_cost_function(mv, accel_factors_T);
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

        /*
         * Sequential implementation of Alternating Least Squares (ALS) method.
         * 
         * @param  R   [in]     The rank of decomposition.
         * @param  mv  [in]     Struct where ALS variables are stored and being updated
         *                      until a termination condition is true.
         * @param  st  [in,out] Struct where the returned values of @c Gtc are stored.
         */
        void aogtc(Member_Variables       &mv,
                   Status                 &status)
        {
          for (std::size_t i=0; i<TnsSize; i++)
          {
            // mv.factors_T[i] = status.factors[i].transpose();
            // mv.factor_T_factor[i].noalias() = mv.factors_T[i] * status.factors[i];
            mv.factor_T_factor[i].noalias() = mv.factors_T[i] * mv.factors_T[i].transpose();
            mv.mttkrp_T[i] = Matrix(mv.rank, mv.tnsDims[i]);
          }

          // if(status.options.normalization)
          // {
          //   choose_normilization_factor(status, mv.all_orthogonal, mv.weight_factor);
          // }
          // Normalize(static_cast<int>(R), mv.factor_T_factor, status.factors);
          
          status.frob_tns         = (mv.tnsX[0]).squaredNorm();
          cost_function(mv, status);
          status.rel_costFunction = status.f_value/status.frob_tns;
          
          // ---- Loop until ALS converges ----
          while(1)
          {
            status.ao_iter++;
            Partensor()->Logger()->info("iter: {} -- fvalue: {} -- relative_costFunction: {}", status.ao_iter, 
                                                         status.f_value, status.rel_costFunction);
            
            for (std::size_t i=0; i<TnsSize; i++)
            {
              mv.cwise_factor_product = PartialCwiseProd(mv.factor_T_factor, i);

              // Update factor
              update_factor(i, mv, status);
            }

            cost_function(mv, status);
            status.rel_costFunction = status.f_value/status.frob_tns;
          
            // if(status.options.normalization && !mv.all_orthogonal)
            //   Normalize(mv.weight_factor, static_cast<int>(R), mv.factor_T_factor, status.factors);

            // ---- Terminating condition ----
            if (status.rel_costFunction < status.options.threshold_error || status.ao_iter >= status.options.max_iter)
            {
              if(status.options.writeToFile)
                writeFactorsToFile(status);  
              break;
            }

            if (status.options.acceleration)
            {
              mv.norm_factors_T = mv.factors_T;
              // ---- Acceleration Step ----
              if (status.ao_iter > 1)
                line_search_accel(mv, status);

              mv.old_factors_T = mv.norm_factors_T;
            }  
          } // end of while
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

          // Begin Load Balancing
	        Matrix                                     Balanced_Ratings_Base_T(TnsSize + 1, options.nonZeros);
	        std::array<std::vector<long int>, TnsSize> perm_tns_indices;

          Matrix Ratings_Base_T = Matrix(static_cast<int>(TnsSize+1), options.nonZeros);
          // Read the whole Tensor from a file
          read( options.ratings_path, 
                fileSize, 
                0, 
                Ratings_Base_T );

	        BalanceDataset<TnsSize>(options.nonZeros, options.tnsDims, Ratings_Base_T, perm_tns_indices, Balanced_Ratings_Base_T);

          // GTC_Base<TnsSize>::sort_ratings_base(Ratings_Base_T, mv.tnsX, options.tnsDims, options.nonZeros);
          GTC_Base<TnsSize>::sort_ratings_base(Balanced_Ratings_Base_T, mv.tnsX, options.tnsDims, options.nonZeros);
          Ratings_Base_T.resize(0,0);
          Balanced_Ratings_Base_T.resize(0,0);
          
          for (std::size_t i=0; i<TnsSize; i++)
          {
            mv.tnsX[i].makeCompressed();
          }
          
          calculate_offsets(mv);

          initialize_factors(mv, status);

          PermuteFactors<TnsSize>(status.factors, perm_tns_indices, mv.factors_T);

          aogtc(mv, status);

          // IF Depermute ....

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

          // Begin Load Balancing
	        Matrix                                     Balanced_Ratings_Base_T(TnsSize + 1, options.nonZeros);
	        std::array<std::vector<long int>, TnsSize> perm_tns_indices;

	        BalanceDataset<TnsSize>(options.nonZeros, options.tnsDims, Ratings_Base_T, perm_tns_indices, Balanced_Ratings_Base_T);

          ReserveSparseTensor<TnsSize>(mv.tnsX, options.tnsDims, options.nonZeros);
          // FillSparseTensor<TnsSize>(mv.tnsX, options.nonZeros, Ratings_Base_T, options.tnsDims);
          FillSparseTensor<TnsSize>(mv.tnsX, options.nonZeros, Balanced_Ratings_Base_T, options.tnsDims);
          // Ratings_Base_T.resize(0,0);
          Balanced_Ratings_Base_T.resize(0,0);
          
          for (std::size_t i=0; i<TnsSize; i++)
          {
            mv.tnsX[i].makeCompressed();
          }
          
          calculate_offsets(mv);

          // produce estimate factors using uniform distribution with entries in [0,1].
          initialize_factors(mv, status);
          
          PermuteFactors<TnsSize>(status.factors, perm_tns_indices, mv.factors_T);

          aogtc(mv, status);

          // IF Depermute ....

          return status;
        }
      };
    }  // namespace internal
  }    // namespace v1
} // end namespace partensor

#if USE_MPI

#include "GtcMpi.hpp"
#endif /* USE_MPI */

#if USE_OPENMP

#include "GtcOpenMP.hpp"
#endif /* USE_OPENMP */

namespace partensor
{
  /**
   * Interface of General Tensor Completion(gtc), with the use of an 
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
  template <std::size_t _TnsSize, typename ExecutionPolicy>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,SparseStatus<_TnsSize,execution::execution_policy_t<ExecutionPolicy>,SparseDefaultValues>>
  gtc( ExecutionPolicy                                                                                     &&, 
       SparseOptions<_TnsSize,execution::execution_policy_t<ExecutionPolicy>,SparseDefaultValues>    const &options  )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::GTC<_TnsSize>()(options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::GTC<_TnsSize,execution::openmpi_policy>()(options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::GTC<_TnsSize,execution::openmp_policy>()(options);
    }
    else
      return internal::GTC<_TnsSize>()(options);
  }
  
  /*
   * Interface of General Tensor Completion(gtc). Sequential Policy.
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
  template<std::size_t _TnsSize>
  auto gtc(SparseOptions<_TnsSize> const &options )
  {
    return internal::GTC<_TnsSize,execution::sequenced_policy>()(options);
  }

  /**
   * Interface of General Tensor Completion(gtc), with the use of an 
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
  template <std::size_t _TnsSize, typename ExecutionPolicy>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,SparseStatus<_TnsSize,execution::execution_policy_t<ExecutionPolicy>,SparseDefaultValues>>
  gtc( ExecutionPolicy                                                                                     &&, 
       Matrix                                                                                        const &Ratings_Base_T,
       SparseOptions<_TnsSize,execution::execution_policy_t<ExecutionPolicy>,SparseDefaultValues>    const &options  )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::GTC<_TnsSize>()(Ratings_Base_T,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::GTC<_TnsSize,execution::openmpi_policy>()(Ratings_Base_T,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmp_policy>)
    {
      return internal::GTC<_TnsSize,execution::openmp_policy>()(Ratings_Base_T,options);
    }
    else
      return internal::GTC<_TnsSize>()(Ratings_Base_T,options);
  }
  
  /*
   * Interface of General Tensor Completion(gtc). Sequential Policy.
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
  template<std::size_t _TnsSize>
  auto gtc(Matrix                              const &Ratings_Base_T, 
           SparseOptions<_TnsSize>             const &options )
  {
    return internal::GTC<_TnsSize,execution::sequenced_policy>()(Ratings_Base_T,options);
  }

}

#endif // PARTENSOR_GTC_HPPP