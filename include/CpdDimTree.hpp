#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      30/09/2019
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
* @file      CpdDimTree.hpp
* @details
* Implements the Canonical Polyadic Decomposition using Dimension 
* Trees. Make use of @c spdlog library in order to write output in
* a log file in @c "../log". In case of using parallelism
* with mpi, then the functions from @c CpdDimTreeMpi.hpp will be 
* called.
********************************************************************/

#ifndef PARTENSOR_CPD_DIMTREE_DIM_TREE_HPP
#define PARTENSOR_CPD_DIMTREE_DIM_TREE_HPP

#include "PARTENSOR_basic.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include "execution.hpp"
#include "DataGeneration.hpp"
#include "DimTrees.hpp"
#include "Normalize.hpp"
#include "NesterovMNLS.hpp"
#include "Constants.hpp"
#include "Timers.hpp"
#include "ReadWrite.hpp"
#include "Matricization.hpp"
#include "PartialCwiseProd.hpp"
#include "PartialKhatriRao.hpp"

namespace partensor
{
  inline namespace v1
  {
    namespace internal
    {

      //template <typename ExecutionPolicy, typename Tensor>
      //execution::internal::enable_if_execution_policy<ExecutionPolicy,Tensor>
      //Status cpd_f(ExecutionPolicy &&, Tensor const &tnsX, std::size_t R);

      /*
       * Includes the implementation of CP Decomposition with Dimension Trees. 
       * Based on the given parameters one of the overloaded operators will
       * be called.
       * @tparam Tensor_   Type(data type and order) of input Tensor.
       */
      template <typename Tensor_>
      struct CPD_DIMTREE_Base {
        static constexpr std::size_t TnsSize    = TensorTraits<Tensor_>::TnsSize; /**< Tensor Order. */
        static constexpr std::size_t lastFactor = TnsSize - 1;                    /**< ID of the last factor. */

        using DataType         = typename TensorTraits<Tensor_>::DataType;        /**< Tensor Data type. */
        using MatrixType       = typename TensorTraits<Tensor_>::MatrixType;      /**< Eigen Matrix with the same Data type with the Tensor. */
        using Dimensions       = typename TensorTraits<Tensor_>::Dimensions;      /**< Tensor Dimensions type. */
        using IntArray         = typename TensorTraits<Tensor_>::IntArray;        /**< Type of an @c stl array containing integers. */
        
        using TensorMatrixType = Tensor<2>;                                       /**< Type of 2-dimension Eigen Tensor. */
        using Constraints      = std::array<Constraint, TnsSize>;                 /**< Stl array of size TnsSize and containing Constraint type. */
        using MatrixArray      = std::array<MatrixType, TnsSize>;                 /**< Stl array of size TnsSize and containing MatrixType type. */
        using DoubleArray      = std::array<double, TnsSize>;                     /**< Stl array of size TnsSize and containing double type. */
        using FactorArray      = std::array<FactorDimTree,TnsSize>;               /**< Stl array of size TnsSize and containing FactorDimTree type. */
        
        using IndexPair        = typename std::array<Eigen::IndexPair<int>, 1>;
      };

      template <typename Tensor_, typename ExecutionPolicy = execution::sequenced_policy>
      struct CPD_DIMTREE : public CPD_DIMTREE_Base<Tensor_>
      {
        using          CPD_DIMTREE_Base<Tensor_>::TnsSize;
        using          CPD_DIMTREE_Base<Tensor_>::lastFactor;
        using typename CPD_DIMTREE_Base<Tensor_>::TensorMatrixType;
        using typename CPD_DIMTREE_Base<Tensor_>::Dimensions;
        using typename CPD_DIMTREE_Base<Tensor_>::MatrixArray;
        using typename CPD_DIMTREE_Base<Tensor_>::DataType;
        using typename CPD_DIMTREE_Base<Tensor_>::IntArray;
        using typename CPD_DIMTREE_Base<Tensor_>::FactorArray;
        using typename CPD_DIMTREE_Base<Tensor_>::IndexPair;

        using Options = partensor::Options<Tensor_,execution::sequenced_policy,DefaultValues>;
        using Status  = partensor::Status<Tensor_,execution::sequenced_policy,DefaultValues>;

        // Variables that will be used in cpd with 
        // the Dimension Trees implementations. 
        struct Member_Variables {
          Matrix      last_gramian;
          Matrix      cwise_factor_product;
          Matrix      mttkrp;
          Matrix      currentFactor;
          Matrix      temp_matrix;
          Matrix      tnsX_mat_lastFactor_T;

          FactorArray factors;
          FactorArray norm_factors;
          FactorArray old_factors;

          typename FactorArray::iterator it_factor;
          typename FactorArray::iterator it_old_factor;

          Tensor_          tnsX;
          Tensor_          tnsX_approx;   
          IntArray         labelSet;    // starting label set for root
          const IndexPair  product_dims = { Eigen::IndexPair<int>(0, 0) }; // used for tensor contractions

          bool             all_orthogonal = true;
          int              weight_factor;

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
         * @param  st  [in] Struct where the returned values of @c CpdDimTree are stored.
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
         * @param  st  [in,out] Struct where the returned values of @c CpdDimTree are stored.
         *                      In this case the cost function value is updated.
         */
        void cost_function ( Member_Variables const &mv,
                             Status                 &st)
        {
          st.f_value = sqrt( st.frob_tns -2 * ((mv.mttkrp * mv.currentFactor.transpose()).trace()) + (mv.cwise_factor_product * mv.last_gramian).trace() );
        }

        /*
         * Compute the cost function value at the end of each outer iteration
         * based on the last accelerated factor.
         *  
         * @param  mv                [in] Struct where ALS variables are stored.
         * @param  st                [in] Struct where the returned values of @c CpdDimTree 
         *                                are stored.
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
          return sqrt (st.frob_tns + (PartialCwiseProd(factors_T_factors, lastFactor) * factors_T_factors[lastFactor]).trace()
                 - 2 * ((PartialKhatriRao(factors, lastFactor).transpose() * mv.tnsX_mat_lastFactor_T) * factors[lastFactor]).trace() );
        }

        /*
         * Based on each factor's constraint, a different
         * update function is used at every outer iteration.
         * 
         * Computes also factor^T * factor at the end.
         * 
         * @tparam Dimensions       Array type containing the Tensor dimensions.
         * 
         * @param  idx     [in]     Factor to be updated.
         * @param  R       [in]     The rank of decomposition.
         * @param  tnsDims [in]     Tensor Dimensions. Each index contains the corresponding factor's rows length.
         * @param  st      [in]     Struct where the returned values of @c CpdDimTree are stored. 
         * @param  mv      [in,out] Struct where ALS variables are stored.
         *                          Updates the current factor (@c Matrix type) and then updates 
         *                          the same factor of @c FactorDimTree type.
         */
        template<typename Dimensions>
        void update_factor(int              const  idx,
                           std::size_t      const  R,
                           Dimensions       const &tnsDims,
                           Status           const &st, 
                           Member_Variables       &mv )
        {
          switch ( st.options.constraints[idx] )
          {
            case Constraint::unconstrained:
            {
              mv.currentFactor = mv.mttkrp * mv.cwise_factor_product.inverse();
              break;
            }
            case Constraint::nonnegativity:
            {
              mv.temp_matrix = mv.currentFactor;
              NesterovMNLS(mv.cwise_factor_product, mv.mttkrp, st.options.nesterov_delta_1, 
                                    st.options.nesterov_delta_2, mv.currentFactor);
              if(mv.currentFactor.cwiseAbs().colwise().sum().minCoeff() == 0)
                mv.currentFactor = 0.9 * mv.currentFactor + 0.1 * mv.temp_matrix;
              break;
            }
            case Constraint::orthogonality:
            {
              mv.temp_matrix = mv.mttkrp.transpose() * mv.mttkrp;
              Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(mv.temp_matrix);
              mv.temp_matrix.noalias() = (eigensolver.eigenvectors()) 
                                          * (eigensolver.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal()) 
                                          * (eigensolver.eigenvectors().transpose());
              mv.currentFactor.noalias() = mv.mttkrp * mv.temp_matrix;
              break;
            }
            case Constraint::sparsity:
              break;
            default: // in case of Constraint::constant
              break;
          }
          mv.it_factor->factor     = matrixToTensor(mv.currentFactor, tnsDims[idx], static_cast<int>(R)); // Map factor from Eigen Matrix to Eigen Tensor
          mv.it_factor->gramian = (mv.it_factor->factor).contract(mv.it_factor->factor, mv.product_dims); // Compute Covariance Tensor          
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
         * @tparam Dimensions   Array type containing the Tensor dimensions.
         * 
         * @param  mv  [in,out] Struct where ALS variables are stored.
         *                      In case the acceration is successful factor and 
         *                      factor^T * factor of @c FactorDimTree type are updated.
         * @param  st  [in,out] Struct where the returned values of @c CpdDimTree are stored.
         *                      If the acceleration succeeds updates the cost function value.  
         * 
         */
        template<typename Dimensions>
        void line_search_accel(Dimensions       const &tnsDims,
                               std::size_t      const  R,
                               Member_Variables       &mv,
                               Status                 &st)
        {
          double       f_accel    = 0.0; // Objective Value after the acceleration step
          double const accel_step = pow(st.ao_iter+1,(1.0/(st.options.accel_coeff)));

          Matrix      factor;
          Matrix      old_factor;
          MatrixArray accel_factors;
          MatrixArray accel_gramians;

          for(std::size_t i=0; i<TnsSize; ++i)
          {
            factor            = tensorToMatrix(mv.factors[i].factor, tnsDims[i],static_cast<int>(R));
            old_factor        = tensorToMatrix(mv.old_factors[i].factor, tnsDims[i],static_cast<int>(R));
            accel_factors[i]  = old_factor + accel_step * (factor - old_factor); 
            accel_gramians[i] = accel_factors[i].transpose() * accel_factors[i];
            
            mv.it_factor++;
            mv.it_old_factor++;
          }

          f_accel = accel_cost_function(mv, st, accel_factors, accel_gramians);
          if (st.f_value > f_accel)
          {
            for(std::size_t i=0; i<TnsSize; ++i)
            {
              mv.factors[i].factor     = matrixToTensor(accel_factors[i], tnsDims[i], static_cast<int>(R));
              mv.factors[i].gramian = matrixToTensor(accel_gramians[i], static_cast<int>(R), static_cast<int>(R));
            }
            st.f_value = f_accel;
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
         * Sequential implementation of Alternating Least Squares (ALS) method
         * with Dimension Trees.
         * 
         * @tparam Dimensions       Array type containing the Tensor dimensions.
         * 
         * @param  tnsDims [in]     Tensor Dimensions. Each index contains the corresponding factor's rows length.
         * @param  R       [in]     The rank of decomposition.
         * @param  mv      [in]     Struct where ALS variables are stored  and being updated
         *                          until a termination condition is true.
         * @param  st      [in,out] Struct where the returned values of @c CpdDimTree are stored.
         */
        template<typename Dimensions>
        void als(Dimensions       const &tnsDims,
                 std::size_t      const  R,
                 Member_Variables       &mv,
                 Status                 &status)
        {
          if (status.options.acceleration)
          {
            mv.tnsX_mat_lastFactor_T = (Matricization(mv.tnsX, lastFactor)).transpose();
          }

          if(status.options.normalization)
          {
            choose_normilization_factor(status, mv.all_orthogonal, mv.weight_factor);
          }

          mv.tnsX_approx.resize(tnsDims);
          CpdGen(mv.factors, R, mv.tnsX_approx);
          
          status.frob_tns         = square_norm(mv.tnsX);
          status.f_value          = norm(mv.tnsX - mv.tnsX_approx); // Error_tnsX
          status.rel_costFunction = status.f_value/sqrt(status.frob_tns);
          // increments from 1 to labelSet.size()
          std::iota(mv.labelSet.begin(), mv.labelSet.end(), 1); 

          ExprTree<TnsSize> tree;
          tree.Create(mv.labelSet, tnsDims, R, mv.tnsX);

          mv.it_factor = mv.factors.begin();
          for(std::size_t k = 0; k<TnsSize; k++)
          {
              mv.it_factor->leaf = static_cast<TnsNode<1>*>(search_leaf(k+1, tree));
              mv.it_factor++;
          }
          
          // ---- Loop until ALS converges ----
          while(1)
          {
            status.ao_iter++;
            mv.it_factor = mv.factors.begin();
            Partensor()->Logger()->info("iter: {} -- fvalue: {} -- relative_costFunction: {}", status.ao_iter, 
                                                            status.f_value, status.rel_costFunction);
            
            for(std::size_t i=0; i<TnsSize; i++)
            {
              mv.it_factor->leaf->UpdateTree(TnsSize, i, mv.it_factor);

              // Maps from Eigen Tensor to Eigen Matrix
              mv.temp_matrix          = tensorToMatrix(*reinterpret_cast<TensorMatrixType *>(mv.it_factor->leaf->TensorX()), static_cast<int>(R), tnsDims[i]);
              mv.mttkrp               = mv.temp_matrix.transpose();
              mv.cwise_factor_product = tensorToMatrix(mv.it_factor->leaf->Gramian(), static_cast<int>(R), static_cast<int>(R));
              mv.currentFactor        = tensorToMatrix(mv.it_factor->factor, tnsDims[i], static_cast<int>(R));
              
              update_factor(i, R, tnsDims, status, mv);
              mv.it_factor++;
            }

            mv.it_factor    = mv.factors.end()-1;
            mv.last_gramian = tensorToMatrix(mv.it_factor->gramian, static_cast<int>(R), static_cast<int>(R));
            cost_function(mv, status);
            status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
            if(status.options.normalization && !mv.all_orthogonal)
              Normalize(mv.weight_factor, static_cast<int>(R), tnsDims, mv.factors);

            // ---- Terminating condition ----
            if (status.ao_iter >= status.options.max_iter || status.rel_costFunction < status.options.threshold_error)
            {
              for(std::size_t i=0; i<TnsSize; ++i)
                status.factors[i] = tensorToMatrix(mv.factors[i].factor, tnsDims[i], static_cast<int>(R));

              if(status.options.writeToFile)
                writeFactorsToFile(status);  
              break;
            }

            if (status.options.acceleration)
            {
              mv.norm_factors = mv.factors;
              if (status.ao_iter > 1)
                line_search_accel(tnsDims, R, mv, status);

              mv.old_factors = mv.norm_factors;
            } 
          } // end of while
        }

        /**
         * Implementation of CP Decomposition with Dimension Trees with default values 
         * in Options Struct and randomly generated initial factors.
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
          makeFactors(tnsDims, status.options.constraints, R, mv.factors);
          // Normalize(static_cast<int>(R), tnsDims, factors);

          mv.tnsX = tnsX;
          als(tnsDims, R, mv, status);

          return status;
        }

        /**
         * Implementation of CP Decomposition with Dimension Trees, user's changed values 
         * in Options struct, but with randomly generated initial factors.
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
          makeFactors(tnsDims, status.options.constraints, R, mv.factors);
          // Normalize(static_cast<int>(R), tnsDims, factors);
          mv.tnsX = tnsX;

          switch ( status.options.method )
          {
            case Method::als:
            {
              als(tnsDims, R, mv, status);
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
         * Implementation of CP Decomposition with Dimension Tree, default values in 
         * Options struct, but with initialized factors.
         * 
         * @tparam MatrixArray_     An @c stl array, where the initialized factors will
         *                          be stored. Its size must be equal to the Tensor's @c tnsX
         *                          @c order. The type can be either @c partensor::Matrix,
         *                          or @c partensor::Tensor<2>.
         * 
         * @param  tnsX        [in] The given Tensor to be factorized of @c Tensor_ type, 
         *                          with @c double data.
         * @param  R           [in] The rank of decomposition.
         * @param  factorsInit [in] Uses initialized factors instead of randomly generated. The 
         *                          data can be either @c partensor::Matrix, or 
         *                          @c partensor::Tensor<2> type and stored in an @c stl array 
         *                          with size same as the @c order of @c tnsX.
         *
         * @returns An object of type @c Status with the results of the algorithm.
         */
        template<typename MatrixArray_>
        Status operator()(Tensor_      const &tnsX, 
                          std::size_t  const  R, 
                          MatrixArray_ const &factorsInit)
        {
          Status           status = MakeStatus<Tensor_>();
          Member_Variables mv;

          // extract dimensions from tensor
          Dimensions const &tnsDims = tnsX.dimensions();
          // Copy factorsInit data to factors - FactorDimTree data struct
          fillDimTreeFactors(factorsInit, status.options.constraints, mv.factors);
          // Normalize(static_cast<int>(R), tnsDims, factors);

          mv.tnsX = tnsX;
          als(tnsDims, R, mv, status);
          return status;
        }

        /**
         * Implementation of CP Decomposition with Dimension Trees, user's changed values in 
         * Options struct and also initialized factors.
         * 
         * @tparam MatrixArray_     An @c stl array, where the initialized factors will
         *                          be stored. Its size must be equal to the Tensor's @c tnsX
         *                          @c order. The type can be either @c partensor::Matrix,
         *                          or @c partensor::Tensor<2>.
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
        template<typename MatrixArray_>
        Status operator()(Tensor_      const &tnsX, 
                          std::size_t  const  R, 
                          Options      const &options, 
                          MatrixArray_ const &factorsInit)
        {
          Status status(options);
          Member_Variables mv;

          // extract dimensions from tensor
          Dimensions const &tnsDims = tnsX.dimensions();
          // Copy factorsInit data to factors - FactorDimTree data struct
          fillDimTreeFactors(factorsInit, status.options.constraints, mv.factors);
          // Normalize(static_cast<int>(R), tnsDims, factors);
          mv.tnsX = tnsX;

          switch ( status.options.method )
          {
            case Method::als:
            {
              als(tnsDims, R, mv, status);
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
         * Implementation of CP Decomposition with Dimension Trees, default values in Options
         * Struct and randomly generated initial factors. In this implementation the Tensor 
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
          makeFactors(tnsDims, status.options.constraints, R, mv.factors);
          // Normalize(static_cast<int>(R), tnsDims, factors);

          als(tnsDims, R, mv, status);
          return status;
        }

        /**
         * Implementation of CP Decomposition with Dimension Trees, user's changed values 
         * in Options struct and randomly generated initial factors. In this implementation 
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
          makeFactors(tnsDims, status.options.constraints, R, mv.factors);
          // Normalize(static_cast<int>(R), tnsDims, mv.factors);

          switch ( status.options.method )
          {
            case Method::als:
            {
              als(tnsDims, R, mv, status);
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
         * Implementation of CP Decomposition with Dimension Trees, default values in Options
         * Struct and initialized factors. In this implementation the Tensor and the factors 
         * can be read from a file, given the @c paths where the Tensor and the initialized 
         * factors are located.
         * 
         * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
         *                         length must be same as the Tensor order.     
         * @param  R          [in] The rank of decomposition.
         * @param  paths      [in] An @c stl array containing paths for the Tensor to be 
         *                         factorized and after that the paths for the initialized 
         *
         * @returns An object of type @c Status with the results of the algorithm.
         */
        Status operator()(std::array<int, TnsSize>          const &tnsDims, 
                          std::size_t                       const  R, 
                          std::array<std::string,TnsSize+1> const &paths)
        {
          using  Tensor = Tensor<static_cast<int>(TnsSize)>;

          Status           status = MakeStatus<Tensor>();
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
          // Copy factorsInit data to factors - FactorDimTree data struct
          fillDimTreeFactors(status.factors, status.options.constraints, mv.factors);
          // Normalize(static_cast<int>(R), tnsDims, mv.factors);

          als(tnsDims, R, mv, status);
          return status;
        }

        /**
         * Implementation of CP Decomposition with Dimension Trees, user's changed values in 
         * Options struct and initialized factors. In this implementation the Tensor and the 
         * factors can be read from a file, given the @c paths where the Tensor and the 
         * initialized factors are located.
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
          // Copy factorsInit data to factors - FactorDimTree data struct
          fillDimTreeFactors(status.factors, status.options.constraints, mv.factors);
          // Normalize(static_cast<int>(R), tnsDims, factors);
          switch ( status.options.method )
          {
            case Method::als:
            {
              als(tnsDims, R, mv, status);
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
  }  // namespace v1
} // end namespace partensor

#if USE_MPI

#include "CpdDimTreeMpi.hpp"
#endif /* USE_MPI */

namespace partensor
{

  /**
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Also, an @c Execution @c Policy can be used, which can be either @c sequential
   * or @c parallel with the use of @c MPI. In order to choose a policy, type @c execution::seq  
   * or @c execution::mpi. Default value is @c sequential, in case no @c ExecutionPolicy
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
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template <typename ExecutionPolicy, typename Tensor_>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpdDimTree( ExecutionPolicy       &&, 
              Tensor_         const &tnsX, 
              std::size_t     const  R )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_>()(tnsX,R);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_,execution::openmpi_policy>()(tnsX,R);
    }
    else
      return internal::CPD_DIMTREE<Tensor_>()(tnsX,R);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Sequential Policy.
   * 
   * @tparam Tensor_      Type(data type and order) of input Tensor.
   *                      @c Tensor_ must be @c partensor::Tensor<order>, where 
   *                      @c order must be in range of @c [3-8].
   * @param  tnsX    [in] The given Tensor to be factorized of @c Tensor_ type, 
   *                      with @c double data.
   * @param  R       [in] The rank of decomposition.
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template<typename Tensor_>
  auto cpdDimTree(Tensor_     const &tnsX, 
                  std::size_t const  R)
  {
    return internal::CPD_DIMTREE<Tensor_>()(tnsX,R);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Also, an @c Execution @c Policy can be used, which can be either @c sequential
   * or @c parallel with the use of @c MPI. In order to choose a policy, type @c execution::seq 
   * or  @c execution::mpi. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * @tparam ExecutionPolicy Type of @c stl @c Execution @c Policy 
   *                         (sequential, parallel-mpi). 
   * @tparam Tensor_         Type(data type and order) of input Tensor.
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
  cpdDimTree( ExecutionPolicy       &&, 
              Tensor_         const &tnsX, 
              std::size_t     const  R,
              Options<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues> const &options )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_>()(tnsX,R,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_,execution::openmpi_policy>()(tnsX,R,options);
    }
    else
      return internal::CPD_DIMTREE<Tensor_>()(tnsX,R,options);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Sequential Policy.
   * 
   * @tparam Tensor_      Type(data type and order) of input Tensor.
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
  auto cpdDimTree(Tensor_          const &tnsX, 
                  std::size_t      const  R, 
                  Options<Tensor_> const &options)
  {
    return internal::CPD_DIMTREE<Tensor_,execution::sequenced_policy>()(tnsX,R,options);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Also, an @c Execution @c Policy can be used, which can be either @c sequential
   * or @c parallel with the use of @c MPI. In order to choose a policy, type @c execution::seq 
   * or @c execution::mpi. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * @tparam ExecutionPolicy  Type of @c stl @c Execution @c Policy 
   *                          (sequential, parallel-mpi).
   * @tparam Tensor_          Type(data type and order) of input Tensor.
   * @tparam MatrixArray_     An @c stl array, where the initialized factors will
   *                          be stored. Its size must be equal to the Tensor's @c tnsX
   *                          @c order. The type can be either @c partensor::Matrix,
   *                          or @c partensor::Tensor<2>.
   * 
   * @param  tnsX        [in] The given Tensor to be factorized of @c Tensor_ type, 
   *                          with @c double data.
   * @param  R           [in] The rank of decomposition.
   * @param  factorsInit [in] Uses initialized factors instead of randomly generated. The 
   *                          data can be either @c partensor::Matrix, or 
   *                          @c partensor::Tensor<2> type and stored in an @c stl array 
   *                          with size same as the @c order of @c tnsX.
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template <typename Tensor_, typename MatrixArray_, typename ExecutionPolicy>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpdDimTree( ExecutionPolicy       &&, 
              Tensor_         const &tnsX, 
              std::size_t     const  R, 
              MatrixArray_    const &factorsInit )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_>()(tnsX,R,factorsInit);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_,execution::openmpi_policy>()(tnsX,R,factorsInit);
    }
    else
      return internal::CPD_DIMTREE<Tensor_>()(tnsX,R,factorsInit);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Sequential Policy.
   * 
   * @tparam Tensor_          Type(data type and order) of input Tensor.
   * @tparam MatrixArray_     An @c stl array, where the initialized factors will
   *                          be stored. Its size must be equal to the Tensor's @c tnsX
   *                          @c order. The type can be either @c partensor::Matrix,
   *                          or @c partensor::Tensor<2>.
   * 
   * @param  tnsX        [in] The given Tensor to be factorized of @c Tensor_ type, 
   *                          with @c double data.
   * @param  R           [in] The rank of decomposition.
   * @param  factorsInit [in] Uses initialized factors instead of randomly generated. The 
   *                          data can be either @c partensor::Matrix, or 
   *                          @c partensor::Tensor<2> type and stored in an @c stl array 
   *                          with size same as the @c order of @c tnsX.
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template<typename Tensor_, typename MatrixArray_>
  auto cpdDimTree(Tensor_      const &tnsX, 
                  std::size_t  const  R, 
                  MatrixArray_ const &factorsInit)
  {
    return internal::CPD_DIMTREE<Tensor_, execution::sequenced_policy>()(tnsX,R,factorsInit);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Also, an @c Execution @c Policy can be used, which can be either @c sequential
   * or @c parallel with the use of @c MPI. In order to choose a policy, type @c execution::seq 
   * or @c execution::mpi. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * @tparam ExecutionPolicy  Type of @c stl @c Execution @c Policy 
   *                          (sequential, parallel-mpi).
   * @tparam Tensor_          Type(data type and order) of input Tensor.
   * @tparam MatrixArray_     An @c stl array, where the initialized factors will
   *                          be stored. Its size must be equal to the Tensor's @c tnsX
   *                          @c order. The type can be either @c partensor::Matrix,
   *                          or @c partensor::Tensor<2>.
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
  template <typename Tensor_, typename MatrixArray_, typename ExecutionPolicy>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpdDimTree( ExecutionPolicy       &&, 
              Tensor_         const &tnsX, 
              std::size_t     const  R, 
              Options<Tensor_,execution::execution_policy_t<ExecutionPolicy>,DefaultValues> const &options,
              MatrixArray_    const &factorsInit )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_>()(tnsX,R,options,factorsInit);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_,execution::openmpi_policy>()(tnsX,R,options,factorsInit);
    }
    else
      return internal::CPD_DIMTREE<Tensor_>()(tnsX,R,options,factorsInit);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Sequential Policy.
   * 
   * @tparam Tensor_          Type(data type and order) of input Tensor.
   * @tparam MatrixArray_     An @c stl array, where the initialized factors will
   *                          be stored. Its size must be equal to the Tensor's @c tnsX
   *                          @c order. The type can be either @c partensor::Matrix,
   *                          or @c partensor::Tensor<2>.
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
  template<typename Tensor_, typename MatrixArray_>
  auto cpdDimTree(Tensor_          const &tnsX, 
                  std::size_t      const  R, 
                  Options<Tensor_> const &options, 
                  MatrixArray_     const &factorsInit)
  {
    return internal::CPD_DIMTREE<Tensor_,execution::sequenced_policy>()(tnsX,R,options,factorsInit);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Also, an @c Execution @c Policy can be used, which can be either @c sequential
   * or @c parallel with the use of @c MPI. In order to choose a policy, type @c execution::seq 
   * or  @c execution::mpi. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * With this version of @c cpdDimTree, the Tensor can be read from a file, specified 
   * in @c path variable.
   * 
   * @tparam ExecutionPolicy Type of @c stl @c Execution @c Policy 
   *                         (sequential, parallel-mpi). 
   * @tparam TnsSize         Order of the input Tensor.
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
  cpdDimTree( ExecutionPolicy                &&, 
              std::array<int, TnsSize> const &tnsDims, 
              std::size_t              const  R,
              std::string              const &path )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;
    using  Tensor_  = Tensor<static_cast<int>(TnsSize)>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_>()(tnsDims,R,path);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_,execution::openmpi_policy>()(tnsDims,R,path);
    }
    else
      return internal::CPD_DIMTREE<Tensor_>()(tnsDims,R,path);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Sequential Policy.
   * 
   * With this version of @c cpdDimTree, the Tensor can be read from a file, specified 
   * in @c path variable.
   * 
   * @tparam TnsSize         Order of the input Tensor.
   *
   * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
   *                         length must be same as the Tensor order.       
   * @param  R          [in] The rank of decomposition.
   * @param  path       [in] The path where the tensor is located.
   *
   * @returns An object of type @c Status with the results of the algorithm.
   */
  template<std::size_t TnsSize>
  auto cpdDimTree(std::array<int, TnsSize> const &tnsDims, 
                  std::size_t              const  R, 
                  std::string              const &path)
  {
    using Tensor_ = Tensor<static_cast<int>(TnsSize)>;
    return internal::CPD_DIMTREE<Tensor_,execution::sequenced_policy>()(tnsDims,R,path);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Also, an @c Execution @c Policy can be used, which can be either @c sequential
   * or @c parallel with the use of @c MPI. In order to choose a policy, type @c execution::seq 
   * or @c execution::mpi. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * With this version of @c cpdDimTree, the Tensor can be read from a file, specified 
   * in @c path variable.
   * 
   * @tparam ExecutionPolicy Type of @c stl @c Execution @c Policy 
   *                         (sequential, parallel-mpi). 
   * @tparam TnsSize         Order of the input Tensor.
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
  cpdDimTree( ExecutionPolicy                &&, 
              std::array<int, TnsSize> const &tnsDims, 
              std::size_t              const  R,
              std::string              const &path,
              Options<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues> const &options )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;
    using  Tensor_  = Tensor<static_cast<int>(TnsSize)>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_>()(tnsDims,R,path,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_,execution::openmpi_policy>()(tnsDims,R,path,options);
    }
    else
      return internal::CPD_DIMTREE<Tensor_>()(tnsDims,R,path,options);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Sequential Policy.
   * 
   * With this version of @c cpdDimTree, the Tensor can be read from a file, specified 
   * in @c path variable.
   * 
   * @tparam TnsSize         Order of the input Tensor.
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
  auto cpdDimTree(std::array<int, TnsSize>                   const &tnsDims, 
                  std::size_t                                const  R, 
                  std::string                                const &path, 
                  Options<Tensor<static_cast<int>(TnsSize)>> const &options )
  {
    using Tensor_ = Tensor<static_cast<int>(TnsSize)>;
    return internal::CPD_DIMTREE<Tensor_,execution::sequenced_policy>()(tnsDims,R,path,options);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Also, an @c Execution @c Policy can be used, which can be either @c sequential
   * or @c parallel with the use of @c MPI. In order to choose a policy, type @c execution::seq 
   * or @c execution::mpi. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * With this version of @c cpdDimTree, the Tensor can be read from a file, specified 
   * in @c path variable.
   * 
   * @tparam ExecutionPolicy Type of @c stl @c Execution @c Policy 
   *                         (sequential, parallel-mpi). 
   * @tparam TnsSize         Order of the input Tensor.
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
  template <typename ExecutionPolicy, std::size_t TnsSize>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpdDimTree( ExecutionPolicy                          &&, 
              std::array<int, TnsSize>           const &tnsDims, 
              std::size_t                        const  R,
              std::array<std::string, TnsSize+1> const &paths )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;
    using  Tensor_  = Tensor<static_cast<int>(TnsSize)>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_>()(tnsDims,R,paths);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_,execution::openmpi_policy>()(tnsDims,R,paths);
    }
    else
      return internal::CPD_DIMTREE<Tensor_>()(tnsDims,R,paths);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Sequential Policy.
   * 
   * With this version of @c cpdDimTree, the Tensor can be read from a file, specified 
   * in @c path variable.
   * 
   * @tparam TnsSize         Order of the input Tensor.
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
  auto cpdDimTree(std::array<int, TnsSize>           const &tnsDims, 
                  std::size_t                        const  R,
                  std::array<std::string, TnsSize+1> const &paths  )
  {
    using Tensor_ = Tensor<static_cast<int>(TnsSize)>;
    return internal::CPD_DIMTREE<Tensor_,execution::sequenced_policy>()(tnsDims,R,paths);
  }

  /**
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Also, an @c Execution @c Policy can be used, which can be either @c sequential
   * or @c parallel with the use of @c MPI. In order to choose a policy, type @c execution::seq 
   * or @c execution::mpi. Default value is @c sequential, in case no @c ExecutionPolicy
   * is passed.
   * 
   * With this version of @c cpdDimTree, the Tensor can be read from a file, specified 
   * in @c path variable.
   * 
   * @tparam ExecutionPolicy Type of @c stl @c Execution @c Policy 
   *                         (sequential, parallel-mpi). 
   * @tparam TnsSize         Order of the input Tensor.
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
  template <typename ExecutionPolicy, std::size_t TnsSize>
  execution::internal::enable_if_execution_policy<ExecutionPolicy,Status<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues>>
  cpdDimTree( ExecutionPolicy                          &&, 
              std::array<int, TnsSize>           const &tnsDims, 
              std::size_t                        const  R,
              std::array<std::string, TnsSize+1> const &paths, 
              Options<Tensor<static_cast<int>(TnsSize)>,execution::execution_policy_t<ExecutionPolicy>,DefaultValues> const &options )
  {
    using  ExPolicy = execution::execution_policy_t<ExecutionPolicy>;
    using  Tensor_  = Tensor<static_cast<int>(TnsSize)>;

    if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_>()(tnsDims,R,paths,options);
    }
    else if constexpr (std::is_same_v<ExPolicy,execution::openmpi_policy>)
    {
      return internal::CPD_DIMTREE<Tensor_,execution::openmpi_policy>()(tnsDims,R,paths,options);
    }
    else
      return internal::CPD_DIMTREE<Tensor_>()(tnsDims,R,paths,options);
  }

  /*
   * Interface of Canonical Polyadic Decomposition(cpd) with the use of Dimension 
   * Trees. Sequential Policy.
   * 
   * With this version of @c cpdDimTree, the Tensor can be read from a file, specified 
   * in @c path variable.
   * 
   * @tparam TnsSize         Order of the input Tensor.
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
  auto cpdDimTree(std::array<int, TnsSize>                   const &tnsDims, 
                  std::size_t                                const  R, 
                  std::array<std::string, TnsSize+1>         const &paths, 
                  Options<Tensor<static_cast<int>(TnsSize)>> const &options )
  {
    using Tensor_ = Tensor<static_cast<int>(TnsSize)>;
    return internal::CPD_DIMTREE<Tensor_,execution::sequenced_policy>()(tnsDims,R,paths,options);
  }

} // end namespace partensor

#endif // PARTENSOR_CPD_DIMTREE_DIM_TREE_HPP
