#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      11/12/2019
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
* @file      CpdDimTreeMpi.hpp
* @details
* Implements the Canonical Polyadic Decomposition(cpd) using @c MPI.
* and Dimensional Trees.
* Make use of @c spdlog library in order to write output in
* a log file in @c "../log".
********************************************************************/

#if !defined(PARTENSOR_CPD_DIMTREE_DIM_TREE_HPP)
#error "CpdDimTreeMpi can only included inside CpdDimTree"
#endif /* PARTENSOR_CPD_DIMTREE_DIM_TREE_HPP */

#include <math.h>
#include "PartialCwiseProd.hpp"
#include "TensorOperations.hpp"
#include "unsupported/Eigen/MatrixFunctions"

namespace partensor
{

    inline namespace v1 {
        namespace internal {

            /**
             * Includes the implementation of CPDMPI factorization. Based on the given
             * parameters one of the four overloaded operators will be called.
             * @tparam Tensor_ The Type of the given Eigen Tensor to be factorized.
             */
            template<typename Tensor_>
            struct CPD_DIMTREE<Tensor_,execution::openmpi_policy> : public CPD_DIMTREE_Base<Tensor_>
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

                // For MPI usage
                using CartCommunicator = partensor::cartesian_communicator; // From ParallelWrapper.hpp
                using CartCommVector   = std::vector<CartCommunicator>;
                using IntVector        = std::vector<int>;
                using Int2DVector      = std::vector<std::vector<int>>;

                using Options = partensor::Options<Tensor_,execution::openmpi_policy,DefaultValues>;
                using Status  = partensor::Status<Tensor_,execution::openmpi_policy,DefaultValues>;
                
                // Variables that will be used in cpd with 
                // Dimension Trees implementations. 
                struct Member_Variables 
                {
                    MPI_Communicator &world = Partensor()->MpiCommunicator(); // MPI_COMM_WORLD

                    double          local_f_value;
                    int             RxR;
                    int             world_size;
                    const IndexPair product_dims = { Eigen::IndexPair<int>(0, 0) }; // used for tensor contractions    
                    
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
                    IntArray        labelSet;        // starting label set for root

                    MatrixArray     local_factors;
                    MatrixArray     local_factors_T;
                    MatrixArray     layer_factors;
                    MatrixArray     layer_factors_T;                    
                    MatrixArray     local_mttkrp;
                    MatrixArray     layer_mttkrp;
                    MatrixArray     local_mttkrp_T;
                    MatrixArray     layer_mttkrp_T;

                    Matrix          cwise_factor_product;
                    Matrix          factor_T_factor;
                    Matrix          last_cov;
                    Matrix          temp_matrix;
                    Matrix          tnsX_mat_lastFactor_T;
                    Matrix          nesterov_old_layer_factor;

                    Tensor_         subTns;
                    Tensor_         subTnsX_approx;  // tensor in order to compute starting f_value from random generated factors

                    FactorArray     factors;
                    FactorArray     layer_factors_dimTree;
                    FactorArray     norm_factors;
                    FactorArray     old_factors;

                    std::array<int, 2> subfactor_offsets;
                    std::array<int, 2> subfactor_extents;

                    typename FactorArray::iterator status_factor_it;
                    typename FactorArray::iterator layer_factor_it;

                    bool             all_orthogonal = true;
                    int              weight_factor;
                    
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
                 * Compute the cost function value based on the initial factors.
                 * 
                 * @param  grid_comm [in]     The communication grid, where the processors
                 *                            communicate their cost function.
                 * @param  R         [in]     The rank of decomposition.
                 * @param  mv        [in]     Struct where ALS variables are stored.
                 * @param  st        [in,out] Struct where the returned values of @c CpdDimTree are stored.
                 *                            In this case the cost function value  and the Frobenius 
                 *                            squared norm of the tensor are updated.
                 */
                void cost_function_init(CartCommunicator const &grid_comm,
                                        std::size_t      const  R,
                                        Member_Variables       &mv,
                                        Status                 &st )
                {
                    mv.subTnsX_approx.resize(mv.subTns_extents);
                    CpdGen(mv.layer_factors_dimTree, R, mv.subTnsX_approx);
                    
                    // communicate the squared norm of sub tensor, in order to compute frob_tns
                    all_reduce( grid_comm, 
                                square_norm(mv.subTns), 
                                st.frob_tns, 
                                std::plus<double>());
                    // communication among all processors for f_value
                    all_reduce( grid_comm, 
                                square_norm(mv.subTns - mv.subTnsX_approx),
                                st.f_value, 
                                std::plus<double>());
                    st.f_value = sqrt(st.f_value);
                }

                /*
                 * Compute the cost function value at the end of each outer iteration
                 * based on the last factor.
                 * 
                 * @param  grid_comm [in]     The communication grid, where the processors
                 *                            communicate their cost function.
                 * @param  mv        [in]     Struct where ALS variables are stored.
                 * @param  st        [in,out] Struct where the returned values of @c CpdDimTree are stored.
                 *                            In this case the cost function value is updated.
                 */
                void cost_function( CartCommunicator const &grid_comm,
                                    Member_Variables       &mv,
                                    Status                 &st )
                {
                    mv.local_f_value = ((mv.layer_mttkrp_T[lastFactor] * mv.layer_factors[lastFactor]).trace()); 
                    all_reduce( grid_comm,
                                inplace(&mv.local_f_value),
                                1,
                                std::plus<double>() );
                    st.f_value = sqrt(st.frob_tns - 2 * mv.local_f_value + (mv.cwise_factor_product.cwiseProduct(mv.factor_T_factor).sum()));
                }

                /*
                 * Compute the cost function value at the end of each outer iteration
                 * based on the last accelerated factor.
                 * 
                 * @param  grid_comm         [in] The communication grid, where the processors
                 *                                communicate their cost function.
                 * @param  mv                [in] Struct where ALS variables are stored.
                 * @param  st                [in] Struct where the returned values of @c CpdDimTree 
                 *                                are stored.
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
                 * Also initialize the FactorDimTree struct for each processor with the
                 * data from factors.
                 * 
                 * @tparam Dimensions          Array type containing the Tensor dimensions.
                 * 
                 * @param  tnsDims    [in]     Tensor Dimensions. Each index contains the corresponding 
                 *                             factor's rows length.
                 * @param  st         [in]     Struct where the returned values of @c CpdDimTree are stored.
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
                    mv.status_factor_it = mv.factors.begin(); 
                    mv.layer_factor_it  = mv.layer_factors_dimTree.begin(); 
                    for (std::size_t i = 0; i < TnsSize; ++i) 
                    {
                        DisCount(mv.displs_subTns[i], mv.subTnsDims[i], st.options.proc_per_mode[i], tnsDims[i], 1);
                        // for fiber communication and Gatherv
                        DisCount(mv.displs_subTns_R[i], mv.subTnsDims_R[i], st.options.proc_per_mode[i], tnsDims[i], static_cast<int>(R));
                        // information per layer
                        DisCount(mv.displs_local_update[i], mv.send_recv_counts[i], mv.world_size / st.options.proc_per_mode[i], 
                                                            mv.subTnsDims[i][mv.fiber_rank[i]], static_cast<int>(R));
                        // sizes and skips for sub factor
                        mv.subfactor_offsets = { mv.displs_subTns[i][mv.fiber_rank[i]], 0 };
                        mv.subfactor_extents = { mv.subTnsDims[i][mv.fiber_rank[i]], static_cast<int>(R) };
                        // get sub factor and compute its covariance matrix
                        mv.layer_factor_it->factor     = mv.status_factor_it->factor.slice(mv.subfactor_offsets, mv.subfactor_extents);
                        mv.layer_factor_it->gramian = mv.layer_factor_it->factor.contract(mv.layer_factor_it->factor, mv.product_dims);
                        all_reduce( mv.fiber_comm[i], 
                                    inplace(mv.layer_factor_it->gramian.data()),
                                    mv.RxR, 
                                    std::plus<double>() );
                         
                        mv.rows_for_update[i] = mv.send_recv_counts[i][mv.layer_rank[i]] / static_cast<int>(R);
                        // sizes and skips for sub tensor
                        mv.subTns_offsets[i]  = mv.displs_subTns[i][mv.fiber_rank[i]];
                        mv.subTns_extents[i]  = mv.subTnsDims[i][mv.fiber_rank[i]];
                        mv.local_mttkrp_T[i].resize(R, mv.rows_for_update[i]);
                        mv.layer_factors_T[i].resize(R, mv.subTnsDims[i][mv.fiber_rank[i]]);

                        mv.status_factor_it++;
                        mv.layer_factor_it++;
                    }
                }

                /*
                 * Make use of the dimensions and the number of processors per dimension
                 * and then calculates the dimensions of the subtensor and subfactor for 
                 * each processor.
                 * 
                 * After reading from files the given factors, then initializes the 
                 * FactorDimTree struct for each processor with the data read from files.
                 * 
                 * @tparam Dimensions          Array type containing the Tensor dimensions.
                 * 
                 * @param  tnsDims    [in]     Tensor Dimensions. Each index contains the corresponding 
                 *                             factor's rows length.
                 * @param  st         [in]     Struct where the returned values of @c CpdDimTree are stored.
                 * @param  R          [in]     The rank of decomposition.
                 * 
                 * @param  paths      [in]     Paths where the starting point-factors are located.
                 * @param  mv         [in,out] Struct where ALS variables are stored. 
                 *                             Updates @c stl arrays with dimensions for subtensors and
                 *                             subfactors.
                 */
                template<typename Dimensions>
                void compute_sub_dimensions(Dimensions                         const &tnsDims,
                                            Status                             const &st,
                                            std::size_t                        const  R,
                                            std::array<std::string, TnsSize+1> const &paths, 
                                            Member_Variables                         &mv)
                {
                    mv.layer_factor_it  = mv.layer_factors_dimTree.begin(); 
                    for (std::size_t i = 0; i < TnsSize; ++i) 
                    {
                        DisCount(mv.displs_subTns[i], mv.subTnsDims[i], st.options.proc_per_mode[i], tnsDims[i], 1);
                        // for fiber communication and Gatherv
                        DisCount(mv.displs_subTns_R[i], mv.subTnsDims_R[i], st.options.proc_per_mode[i], tnsDims[i], static_cast<int>(R));
                        // information per layer
                        DisCount(mv.displs_local_update[i], mv.send_recv_counts[i], mv.world_size / st.options.proc_per_mode[i], 
                                                            mv.subTnsDims[i][mv.fiber_rank[i]], static_cast<int>(R));
                        // sizes and skips for sub factor
                        mv.subfactor_offsets = { mv.displs_subTns[i][mv.fiber_rank[i]], 0 };
                        mv.subfactor_extents = { mv.subTnsDims[i][mv.fiber_rank[i]], static_cast<int>(R) };
                        
                        mv.temp_matrix = Matrix(mv.subTnsDims[i][mv.fiber_rank[i]],static_cast<int>(R));
                        read( paths[i+1], 
                              mv.subTnsDims[i][mv.fiber_rank[i]]*static_cast<int>(R),
                              mv.displs_subTns_R[i][mv.fiber_rank[i]], 
                              mv.temp_matrix );

                        // get sub factor and compute its covariance matrix
                        mv.layer_factor_it->factor     = matrixToTensor(mv.temp_matrix, mv.subTnsDims[i][mv.fiber_rank[i]], static_cast<int>(R));
                        mv.layer_factor_it->gramian = mv.layer_factor_it->factor.contract(mv.layer_factor_it->factor, mv.product_dims);
                        all_reduce( mv.fiber_comm[i], 
                                    inplace(mv.layer_factor_it->gramian.data()),
                                    mv.RxR, 
                                    std::plus<double>() );
                            
                        mv.rows_for_update[i] = mv.send_recv_counts[i][mv.layer_rank[i]] / static_cast<int>(R);
                        // sizes and skips for sub tensor
                        mv.subTns_offsets[i]  = mv.displs_subTns[i][mv.fiber_rank[i]];
                        mv.subTns_extents[i]  = mv.subTnsDims[i][mv.fiber_rank[i]];
                        mv.local_mttkrp_T[i].resize(R, mv.rows_for_update[i]);
                        mv.layer_factors_T[i].resize(R, mv.subTnsDims[i][mv.fiber_rank[i]]);

                        mv.layer_factor_it++;
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
                 * @param  st  [in]     Struct where the returned values of @c CpdDimTree 
                 *                      are stored.
                 * @param  mv  [in,out] Struct where ALS variables are stored.
                 *                      Updates the current layer factor (@c Matrix type) and 
                 *                      then updates the same factor of @c FactorDimTree type.
                 */
                void update_factor( int              const  idx,
                                    std::size_t      const  R, 
                                    Status           const &st, 
                                    Member_Variables       &mv  )
                {
                    switch ( st.options.constraints[idx] ) 
                    {
                        case Constraint::unconstrained:
                        {
                            // communicate the local mttkrp
                            v2::reduce_scatter( mv.layer_comm[idx], 
                                                mv.layer_mttkrp_T[idx], 
                                                mv.send_recv_counts[idx][0],
                                                mv.local_mttkrp_T[idx] );
                    
                            mv.local_mttkrp[idx] = mv.local_mttkrp_T[idx].transpose();                    
                            if(mv.rows_for_update[idx] != 0)
                                mv.local_factors[idx] = mv.local_mttkrp[idx] * mv.cwise_factor_product.inverse(); // Compute new factor

                            break;
                        }
                        case Constraint::nonnegativity:
                        {
                            // communicate the local mttkrp
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
                                        
                            if (mv.rows_for_update[idx] != 0) {
                                mv.local_mttkrp[idx]     = mv.layer_mttkrp[idx].block(mv.displs_local_update[idx][mv.layer_rank[idx]] / static_cast<int>(R), 0, 
                                                                                     mv.rows_for_update[idx],                            static_cast<int>(R)); 
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
                                // mv.local_factors[idx].noalias() = mv.local_mttkrp[idx] * (mv.temp_matrix.pow(-0.5));
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
                        v2::all_gatherv(mv.layer_comm[idx], 
                                        mv.local_factors_T[idx],
                                        mv.send_recv_counts[idx][mv.layer_rank[idx]], 
                                        mv.send_recv_counts[idx][0],
                                        mv.displs_local_update[idx][0],
                                        mv.layer_factors_T[idx] );
                        
                        mv.layer_factors[idx] = mv.layer_factors_T[idx].transpose();
                        mv.factor_T_factor    = mv.layer_factors_T[idx] * mv.layer_factors[idx];
                        all_reduce( mv.fiber_comm[idx], 
                                    inplace(mv.factor_T_factor.data()), 
                                    mv.RxR,
                                    std::plus<double>() );

                        if(st.options.constraints[idx] == Constraint::nonnegativity)
                        {
                            if ((mv.factor_T_factor.diagonal()).minCoeff()==0)
                            {
                            mv.layer_factors[idx] = 0.9 * mv.layer_factors[idx] + 0.1 * mv.nesterov_old_layer_factor;
                            all_reduce( mv.fiber_comm[idx], 
                                        inplace(mv.factor_T_factor.data()), 
                                        mv.RxR,
                                        std::plus<double>() ); 
                            }
                        }

                        mv.layer_factor_it->factor     = matrixToTensor(mv.layer_factors[idx], mv.subTnsDims[idx][mv.fiber_rank[idx]], static_cast<int>(R)); // Map factor from Eigen Matrix to Eigen Tensor
                        mv.layer_factor_it->gramian = matrixToTensor(mv.factor_T_factor, static_cast<int>(R), static_cast<int>(R));   // Map Covariance from Eigen Matrix to Eigen Tensor            
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
                 * @param  st      [in,out] Struct where the returned values of @c CpdDimTree are stored.
                 *                          Stores the resulted factors.
                 */
                template<typename Dimensions>
                void gather_final_factors(Dimensions       const &tnsDims,
                                          std::size_t      const  R,
                                          Member_Variables       &mv,
                                          Status                 &st)
                {
                    for(std::size_t i=0; i<TnsSize; ++i)
                    {
                        mv.temp_matrix        = tensorToMatrix(mv.layer_factors_dimTree[i].factor, mv.subTnsDims[i][mv.fiber_rank[i]], static_cast<int>(R));
                        mv.layer_factors_T[i] = mv.temp_matrix.transpose();
                    }

                    for(std::size_t i=0; i<TnsSize; ++i)
                    {
                        mv.temp_matrix.resize(static_cast<int>(R), tnsDims[i]);
                        // Gatherv from all processors to processor with rank 0 the final factors
                        v2::gatherv(mv.fiber_comm[i],
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
                 * @param  R         [in]     Rank of the factorization.
                 * @param  mv        [in,out] Struct where ALS variables are stored.
                 *                            In case the acceration is successful layer factor^T * factor 
                 *                            and layer factor variables are updated.
                 * @param  st        [in,out] Struct of the returned values of @c CpdDimTree are stored.
                 *                            If the acceleration succeeds updates cost function value.  
                 */
                void line_search_accel(CartCommunicator const &grid_comm,
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
                        factor            = tensorToMatrix(mv.layer_factors_dimTree[i].factor, mv.subTnsDims[i][mv.fiber_rank[i]],static_cast<int>(R));
                        old_factor        = tensorToMatrix(mv.old_factors[i].factor, mv.subTnsDims[i][mv.fiber_rank[i]],static_cast<int>(R));
                        accel_factors[i]  = old_factor + accel_step * (factor - old_factor); 
                        accel_gramians[i] = accel_factors[i].transpose() * accel_factors[i];
                        all_reduce( mv.fiber_comm[i], 
                                    inplace(accel_gramians[i].data()),
                                    mv.RxR, 
                                    std::plus<double>() ); 
                    }

                    f_accel = accel_cost_function(grid_comm, mv, st, accel_factors, accel_gramians);
                    if (st.f_value > f_accel)
                    {
                        for(std::size_t i=0; i<TnsSize; ++i)
                        {
                            mv.layer_factors_dimTree[i].factor     = matrixToTensor(accel_factors[i], mv.subTnsDims[i][mv.fiber_rank[i]], static_cast<int>(R));
                            mv.layer_factors_dimTree[i].gramian = matrixToTensor(accel_gramians[i], static_cast<int>(R), static_cast<int>(R));
                        }
                        st.f_value = f_accel;
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
                 * @param  grid_comm [in]     MPI communicator where the new cost function value
                 *                            will be communicated and computed.
                 * @param  tnsDims    [in]     Tensor Dimensions. Each index contains the corresponding 
                 *                             factor's rows length.
                 * @param  R         [in]     The rank of decomposition.
                 * @param  mv        [in]     Struct where ALS variables are stored  and being updated
                 *                            until a termination condition is true.
                 * @param  st        [in,out] Struct where the returned values of @c CpdDimTree are stored.
                 */
                template<typename Dimensions>
                void als(CartCommunicator const &grid_comm,
                         Dimensions       const &tnsDims,
                         std::size_t      const  R,
                         Member_Variables       &mv,
                         Status                 &status)
                {
                    mv.tnsX_mat_lastFactor_T = (Matricization(mv.subTns, lastFactor)).transpose();
                    if(status.options.normalization)
                    {
                      choose_normilization_factor(status, mv.all_orthogonal, mv.weight_factor);
                    }

                    std::iota(mv.labelSet.begin(), mv.labelSet.end(), 1); // increments from 1 to labelSet.size()
                    
                    ExprTree<TnsSize> tree;
                    tree.Create(mv.labelSet, mv.subTns_extents, R, mv.subTns);
                    
                    mv.layer_factor_it = mv.layer_factors_dimTree.begin(); 
                    for(std::size_t k = 0; k<TnsSize; k++)
                    {
                        mv.layer_factor_it->leaf = static_cast<TnsNode<1>*>(search_leaf(k+1, tree));
                        mv.layer_factor_it++;
                    }

                    // Wait for all processors to reach here
                    grid_comm.barrier();

                    // ---- Loop until ALS converges ----
                    while(1)
                    {
                        status.ao_iter++;
                        mv.layer_factor_it = mv.layer_factors_dimTree.begin();
                        if (!grid_comm.rank())
                            Partensor()->Logger()->info("iter: {} -- fvalue: {} -- relative_costFunction: {}", status.ao_iter, 
                                              status.f_value, status.rel_costFunction);
                        
                        for(std::size_t i=0; i<TnsSize; i++)
                        {
                            mv.layer_factor_it->leaf->UpdateTree(TnsSize, i, mv.layer_factor_it);              

                            // Maps from Eigen Tensor to Eigen Matrix
                            mv.layer_mttkrp_T[i]    = tensorToMatrix(*reinterpret_cast<TensorMatrixType *>(mv.layer_factor_it->leaf->TensorX()), 
                                                                static_cast<int>(R), mv.subTnsDims[i][mv.fiber_rank[i]]);
                            mv.layer_mttkrp[i]      = mv.layer_mttkrp_T[i].transpose();
                            mv.cwise_factor_product = tensorToMatrix(mv.layer_factor_it->leaf->Gramian(), static_cast<int>(R), static_cast<int>(R));
                            mv.layer_factors[i]     = tensorToMatrix(mv.layer_factor_it->factor, mv.subTnsDims[i][mv.fiber_rank[i]], static_cast<int>(R));
                            mv.local_factors[i]     = mv.layer_factors[i].block(mv.displs_local_update[i][mv.layer_rank[i]] / static_cast<int>(R), 0, 
                                                                                mv.rows_for_update[i],                           static_cast<int>(R));

                            update_factor(i, R, status, mv);
                            mv.layer_factor_it++;
                        }
                        
                        cost_function( grid_comm, mv, status );
                        status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
                        if(status.options.normalization && !mv.all_orthogonal)
                            Normalize(mv.weight_factor, static_cast<int>(R), tnsDims, mv.factors);
                        
                        // ---- Terminating condition ----
                        if (status.ao_iter >= status.options.max_iter || status.rel_costFunction < status.options.threshold_error)
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
                            mv.norm_factors = mv.layer_factors_dimTree;
                            // ---- Acceleration Step ----
                            if (status.ao_iter > 1)
                                line_search_accel(grid_comm, R, mv, status);

                            mv.old_factors = mv.norm_factors;
                        }
                    }
                }

                /**
                 * Implementation of CPDMPI factorization with default values in Options
                 * and no initialized factors.
                 * @param tnsX [in] The given Eigen Tensor to be factorized.
                 * @param R    [in] The rank of decomposition.
                 *
                 * @returns If @c spdlog provoke no exception, returns an object of type
                 * @c Status with the results of the algorithm.
                 */
                Status operator()(Tensor_     const &tnsX, 
                                  std::size_t const  R)
                {
                    Options options = MakeOptions<Tensor_>(execution::openmpi_policy());
                    Status  status(options);
                    Member_Variables mv(R, status.options.proc_per_mode);

                    // Communicator with cartesian topology
                    CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

                    // Functions that create layer and fiber grids.
                    create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
                    create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);
                    
                    // extract dimensions from tensor
                    Dimensions const &tnsDims = tnsX.dimensions();
                    // produce estimate factors using uniform distribution with entries in [0,1].
                    makeFactors(tnsDims, status.options.constraints, R, mv.factors);
                    
                    compute_sub_dimensions(tnsDims, status, R, mv);
                    // Normalize each layer_factor, compute status.frob_tns and status.f_value
                    // Normalize(R, subTns_extents, layer_factors_dimTree);
                    // Each processor takes a subtensor from tnsX
                    mv.subTns = tnsX.slice(mv.subTns_offsets, mv.subTns_extents);
                    cost_function_init(grid_comm, R, mv, status);           
                    status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
                    
                    als(grid_comm, tnsDims, R, mv, status);
                    
                    return status;
                }

                /**
                 * Implementation of CPDMPI factorization with default values in Options
                 * and no initialized factors.
                 * @param tnsX    [in] The given Eigen Tensor to be factorized.
                 * @param R       [in] The rank of decomposition.
                 * @param options [in] The options that the user wishes to use.
                 *
                 * @returns If @c spdlog provoke no exception, returns an object of type
                 * @c Status with the results of the algorithm.
                 */
                Status operator()(Tensor_     const &tnsX, 
                                  std::size_t const  R, 
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
                    makeFactors(tnsDims, status.options.constraints, R, mv.factors);
                    
                    compute_sub_dimensions(tnsDims, status, R, mv);
                    // Normalize each layer_factor, compute status.frob_tns and status.f_value
                    // Normalize(R, subTns_extents, layer_factors_dimTree);
                    // Each processor takes a subtensor from tnsX
                    mv.subTns = tnsX.slice(mv.subTns_offsets, mv.subTns_extents);
                    cost_function_init(grid_comm, R, mv, status);           
                    status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
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
                 * @param tnsX        [in] The given Eigen Tensor to be factorized.
                 * @param R           [in] The rank of decomposition.
                 * @param factorsInit [in] Uses initialized factors instead of randomly generated.
                 *
                 * @returns If @c spdlog provoke no exception, returns an object of type
                 * @c Status with the results of the algorithm.
                 */
                template <typename MatrixArray_>
                Status operator()(Tensor_      const &tnsX, 
                                  std::size_t  const  R, 
                                  MatrixArray_ const &factorsInit)
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
                    // Copy factorsInit data to status.factors - FactorDimTree data struct
                    fillDimTreeFactors(factorsInit, status.options.constraints, mv.factors);
                    
                     compute_sub_dimensions(tnsDims, status, R, mv);
                    // Normalize each layer_factor, compute status.frob_tns and status.f_value
                    // Normalize(R, subTns_extents, layer_factors_dimTree);
                    // Each processor takes a subtensor from tnsX
                    mv.subTns = tnsX.slice(mv.subTns_offsets, mv.subTns_extents);
                    cost_function_init(grid_comm, R, mv, status);           
                    status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
                    
                    als(grid_comm, tnsDims, R, mv, status);
                    return status;
                }

                /**
                 * Implementation of CPDMPI factorization with default values in Options
                 * and no initialized factors.
                 * @param tnsX        [in] The given Eigen Tensor to be factorized.
                 * @param R           [in] The rank of decomposition.
                 * @param options     [in] The options that the user wishes to use.
                 * @param factorsInit [in] Uses initialized factors instead of randomly generated.
                 *
                 * @returns If @c spdlog provoke no exception, returns an object of type
                 * @c Status with the results of the algorithm.
                 */
                template <typename MatrixArray_>
                Status operator()(Tensor_      const &tnsX, 
                                  std::size_t  const  R, 
                                  Options      const &options, 
                                  MatrixArray_ const &factorsInit)
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
                    // Copy factorsInit data to status.factors - FactorDimTree data struct
                    fillDimTreeFactors(factorsInit, status.options.constraints, mv.factors);
                    
                    compute_sub_dimensions(tnsDims, status, R, mv);
                    // Normalize each layer_factor, compute status.frob_tns and status.f_value
                    // Normalize(R, subTns_extents, layer_factors_dimTree);
                    // Each processor takes a subtensor from tnsX
                    mv.subTns = tnsX.slice(mv.subTns_offsets, mv.subTns_extents);
                    cost_function_init(grid_comm, R, mv, status);           
                    status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
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
                 * @tparam TnsSize         Order of the input Tensor.
                 *
                 * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
                 *                         length must be same as the Tensor order.       
                 * @param  R          [in] The rank of decomposition.
                 * @param  path       [in] The path where the tensor is located.
                 *
                 *
                 * @returns An object of type @c Status with the results of the algorithm.
                 */
                template <std::size_t TnsSize>
                Status operator()(std::array<int, TnsSize> const &tnsDims,
                                  std::size_t              const  R, 
                                  std::string              const &path)
                {
                    using   TensorType = Tensor<static_cast<int>(TnsSize)>;

                    Options          options    = MakeOptions<TensorType>(execution::openmpi_policy());
                    Status           status(options);
                    Member_Variables mv(R, status.options.proc_per_mode);

                    // Communicator with cartesian topology
                    CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

                    // Functions that create layer and fiber grids.
                    create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
                    create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);
                    
                    // produce estimate factors using uniform distribution with entries in [0,1].
                    makeFactors(tnsDims, status.options.constraints, R, mv.factors);

                    compute_sub_dimensions(tnsDims, status, R, mv);
                    // Normalize each layer_factor, compute status.frob_tns and status.f_value
                    // Normalize(R, subTns_extents, layer_factors_dimTree);
                    // Each processor takes a subtensor from tnsX
                    mv.subTns.resize(mv.subTns_extents);
                    readTensor( path, tnsDims, mv.subTns_extents, mv.subTns_offsets, mv.subTns );
                    cost_function_init(grid_comm, R, mv, status);           
                    status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
                    
                    als(grid_comm, tnsDims, R, mv, status);
                    return status;
                }

                /**
                 * Implementation of CPDMPI factorization with default values in Options
                 * and no initialized factors.
                 * 
                 * @tparam TnsSize         Order of the input Tensor.
                 *
                 * @param  tnsDims    [in] @c Stl array containing the Tensor dimensions, whose
                 *                         length must be same as the Tensor order.        
                 * @param  R          [in] The rank of decomposition.
                 * @param  path       [in] The path where the tensor is located.
                 * @param  options    [in] The options that the user wishes to use.
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
                    makeFactors(tnsDims, status.options.constraints, R, mv.factors);
                    
                    compute_sub_dimensions(tnsDims, status, R, mv);
                    // Normalize each layer_factor, compute status.frob_tns and status.f_value
                    // Normalize(R, subTns_extents, layer_factors_dimTree);
                    // Each processor takes a subtensor from tnsX
                    mv.subTns.resize(mv.subTns_extents);
                    readTensor( path, tnsDims, mv.subTns_extents, mv.subTns_offsets, mv.subTns );
                    cost_function_init(grid_comm, R, mv, status);           
                    status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
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
                 * @tparam TnsSize      Order of the input Tensor.
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
                    using   TensorType = Tensor<static_cast<int>(TnsSize)>;
                    
                    Options          options = MakeOptions<TensorType>(execution::openmpi_policy());
                    Status           status(options);
                    Member_Variables mv(R, status.options.proc_per_mode);

                    // Communicator with cartesian topology
                    CartCommunicator grid_comm(mv.world, status.options.proc_per_mode, true); 

                    // Functions that create layer and fiber grids.
                    create_layer_grid(grid_comm, mv.layer_comm, mv.layer_rank);
                    create_fiber_grid(grid_comm, mv.fiber_comm, mv.fiber_rank);
                    
                    compute_sub_dimensions(tnsDims, status, R, paths, mv);
                    // Normalize each layer_factor, compute status.frob_tns and status.f_value
                    // Normalize(R, subTns_extents, layer_factors_dimTree);
                    // Each processor takes a subtensor from tnsX
                    mv.subTns.resize(mv.subTns_extents);
                    readTensor( paths[0], tnsDims, mv.subTns_extents, mv.subTns_offsets, mv.subTns );
                    cost_function_init(grid_comm, R, mv, status);           
                    status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
                    als(grid_comm, tnsDims, R, mv, status);

                    return status;
                }

                /**
                 * Implementation of CPDMPI factorization with default values in Options
                 * and no initialized factors.
                 * 
                 * @tparam TnsSize       Order of the input Tensor.
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
                    
                    compute_sub_dimensions(tnsDims, status, R, paths, mv);
                    // Normalize each layer_factor, compute status.frob_tns and status.f_value
                    // Normalize(R, subTns_extents, layer_factors_dimTree);
                    // Each processor takes a subtensor from tnsX
                    mv.subTns.resize(mv.subTns_extents);
                    readTensor( paths[0], tnsDims, mv.subTns_extents, mv.subTns_offsets, mv.subTns );
                    cost_function_init(grid_comm, R, mv, status);           
                    status.rel_costFunction = status.f_value / sqrt(status.frob_tns);
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

            };
      } // end namespace internal
    } // end namespace v1

} //end namespace partensor