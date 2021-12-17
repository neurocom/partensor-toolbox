#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      21/03/2019
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
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file 	 TensorOperations.hpp
* @details
* Contains functions required in @c DimTrees.hpp, but also implementations
* to be used in case of @c Eigen @c Tensor module. 
********************************************************************/

#ifndef PARTENSOR_TENSOR_OPERATIONS_HPP
#define PARTENSOR_TENSOR_OPERATIONS_HPP

#include <iostream>
#include "PARTENSOR_basic.hpp"

namespace partensor
{

 	/**
 	 * @brief Change from @c Tensor to @c Matrix type.
 	 * 
 	 * Changes the type of an @c Tensor and converts it to 
	 * an @c Matrix type.
	 * 
	 * @tparam Tensor_   Type(data type and order) of input Tensor.
	 * @param  tnsX [in] The @c Tensor to be converted.
	 * @param  rows [in] Number of rows of the resulting @c Matrix.
	 * @param  cols [in] Number of columns of the resulting @c Matrix.
	 * 
	 * @returns A @c Matrix ( @c rows, @c cols ), which type same as
	 *          of @c tnsX.
	 */ 
	template<typename Tensor_>
	typename TensorTraits<Tensor_>::MatrixType tensorToMatrix( Tensor_ const &tnsX, 
	                                                           int 	   const  rows, 
															   int 	   const  cols  )
	{
		using  Matrix = typename TensorTraits<Tensor_>::MatrixType;
		return Eigen::Map<const Matrix> (tnsX.data(), rows, cols);
	}

 	/**
	 * @brief Change from @c Matrix to 2D @c Tensor type.
	 * 
	 * Changes the type of a @c Matrix and converts it to 
	 * a 2-dimension @c Tensor type.
	 * 
	 * @param  mtx  [in] The @c Matrix to be converted.
	 * @param  dim0 [in] Number of rows of the resulting @c Tensor.
	 * @param  dim1 [in] Number of columns of the resulting @c Tensor.
	 * 
	 * @returns A 2-dimension @c Tensor ( @c rows, @c cols ), which type same 
	 *          as of @c mtx.
	 */ 
	auto matrixToTensor( Matrix const &mtx, 
	                     int 	const  dim0, 
						 int 	const  dim1 )
	{
	  return Eigen::TensorMap<Eigen::Tensor<const DefaultDataType,2>>(mtx.data(), {dim0,dim1});
	}

 	/**
 	 * @brief Change from @c Matrix to an @c Tensor type.
 	 * 
 	 * Changes the type of an @c Matrix and converts it to 
	 * an N-dimension @c Tensor type.
	 * 
	 * @tparam Dimensions      An array container type.
	 * @param  mtx        [in] The @c Matrix to be converted.
	 * @param  tnsDims    [in] A @c Dimensions array with the lengths of each of 
	 *                         Tensor dimension.
	 * 
	 * @returns An N-dimension @c Tensor ( @c tnsDims[0],tnsDims[1],... ), which type  
	 *          same as of @c mtx.
	 */ 
	template<typename Dimensions>
	auto matrixToTensor( Matrix     const &mtx, 
						 Dimensions const tnsDims )
	{
	  	static constexpr std::size_t TnsSize = tnsDims.size();
		using tensormap = typename Eigen::TensorMap<Eigen::Tensor<const DefaultDataType,TnsSize>>;

    	if  constexpr (TnsSize == 3) {
			return tensormap(mtx.data(), {tnsDims[0],tnsDims[1],tnsDims[2]});
		}
		else if constexpr (TnsSize == 4) {
			return tensormap(mtx.data(), {tnsDims[0],tnsDims[1],tnsDims[2],tnsDims[3]});
		}
		else if constexpr (TnsSize == 5) {
			return tensormap(mtx.data(), {tnsDims[0],tnsDims[1],tnsDims[2],tnsDims[3],tnsDims[4]});
		}
		else if constexpr (TnsSize == 6) {
			return tensormap(mtx.data(), {tnsDims[0],tnsDims[1],tnsDims[2],tnsDims[3],tnsDims[4],tnsDims[5]});
		}
		else if constexpr (TnsSize == 7) {
			return tensormap(mtx.data(), {tnsDims[0],tnsDims[1],tnsDims[2],tnsDims[3],tnsDims[4],tnsDims[5],tnsDims[6]});
		}
		else {
			return tensormap(mtx.data(), {tnsDims[0],tnsDims[1],tnsDims[2],tnsDims[3],tnsDims[4],tnsDims[5],tnsDims[6],tnsDims[7]});
		}
	}

	/**
	 * @brief Frobenius norm of a @c Tensor.
	 * 
	 * Computes the Frobenius Norm of a @c Tensor.
	 * 
	 * @tparam Tensor      Type(data type and order) of input Tensor.
	 * @param  tnsX   [in] The @c Tensor used for this operation.
	 * 
	 * @returns A @c double quantity, with the Frobenius Norm of @c tnsX.
	 */
	template<typename Tensor_>
	double norm( Tensor_ const &tnsX )
	{
		 Tensor<0> frob_norm_tens = tnsX.square().sum().sqrt();
		 return frob_norm_tens.coeff();
	}

	/**
	 * @brief Squared Frobenius norm of a @c Tensor.
	 * 
	 * Computes the Squared Frobenius Norm of a @c Tensor.
	 * 
	 * @tparam Tensor      Type(data type and order) of input Tensor.
	 * @param  tnsX   [in] The @c Tensor used for this operation.
	 * 
	 * @returns A @c double quantity, with the Squared Frobenius Norm of @c tnsX.
	 */
	template<typename Tensor_>
	double square_norm( Tensor_ const &tnsX )
	{
		 Tensor<0> frob_norm_tens = tnsX.square().sum();
		 return frob_norm_tens.coeff();
	}

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Implements the Tensor Contraction and permute the first dimension  
	 * as last in the resulting Tensor.
	 * 
	 * @tparam _TnsSize			       Tensor Order.
	 * @param  parentTensor   [in]     
	 * @param  factor         [in]     
	 * @param  contractDim1   [in]     First dimension of @c Tensor Contraction.
	 * @param  contractDim2   [in]     Second dimension of @c Tensor Contraction.
	 * @param  contractionRes [in,out] The result of Tensor Contraction.
	 */
	template<std::size_t _TnsSize>
	void TensorContraction( Tensor<static_cast<int>(_TnsSize)> const &parentTensor,
							Tensor<2>                          const &factor,
							int                                const  contractDim1,
							int                                const  contractDim2,
							Tensor<static_cast<int>(_TnsSize)>       &contractionRes )
	{
		std::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(contractDim1, contractDim2) };
		std::array<int, _TnsSize>            shuffles;

		Tensor<static_cast<int>(_TnsSize)>   _temp;

		for(std::size_t i=0; i<_TnsSize; i++) { shuffles[i] = i+1; }
		shuffles[_TnsSize-1] = 0;

		_temp          = factor.contract(parentTensor, product_dims);
		contractionRes = _temp.shuffle(shuffles);
	}
	#endif  // DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Implements the Tensor right partial Product for @c Dimension Trees.
	 * 
	 * @tparam ParTnsSize 			   Tensor Order of the Parent @c TnsNode.
	 * @tparam TnsSize 				   Tensor Order.
	 * @param  parentTensor   [in]     Parent @c TnsNode @c Tensor with 
	 *                                 size @c ParTnsSize.
	 * @param  factor 		  [in]     @c TnsNode's 2D - @c Tensor.
	 * @param  chipDim 		  [in]     First dimension of Tensor Contraction.
	 * @param  contractDim 	  [in]     Second dimension of Tensor Contraction.
	 * @param  contractionRes [in,out] The result of Tensor Contraction.
	 */
	template<std::size_t _ParTnsSize, std::size_t _TnsSize>
	void TensorPartialProduct_R(Tensor<static_cast<int>(_ParTnsSize)> const &parentTensor,
								Tensor<2>                             const &factor,
								int                                   const  chipDim,
								int                                   const  contractDim,
                				Tensor<static_cast<int>(_TnsSize)>          *contractionRes )
	{
		static_assert(_ParTnsSize == _TnsSize+1,"Wrong call!");

		const int R = factor.dimension(1);
		
		std::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(contractDim, 0) };

		for (int i=0; i<R; i++)
		{
      		contractionRes->chip(i,chipDim) = (parentTensor.chip(i,chipDim)).contract(factor.chip(i,1),product_dims);
		}
	}
	#endif  // DOXYGEN_SHOULD_SKIP_THIS

	/**
	 * @brief Creates random @c Tensor.
	 * 
	 * In case an @c Tensor is declared, but with no dimensions or there is
	 * a need to change Tensor dimensions, @c RandomTensorGen can be used. In both
	 * cases the Tensor order cannot be changed.
	 * Generates pseudo-random data for @c tnsX, in a uniform distribution.
	 * 
	 * @note The data will be in range of [-1,1].
	 *
	 * @tparam Array_ 		    An array container type.
	 * @tparam Tensor_          Type(data type and order) of input Tensor.
	 * @param  tnsDims [in]  	Contains the lengths of each of @c tnsX dimensions.
	 * @param  tnsX    [in,out] @c Tensor filled with the data.
	 * 
	 * @note   @c tnsX must be initialized before function call.
	 */
	template<typename Array_, typename Tensor_>
	void RandomTensorGen(Array_ const tnsDims, Tensor_ &tnsX)
	{
		std::srand((unsigned int) time(NULL)+std::rand());
		tnsX.resize(tnsDims);
		tnsX.template setRandom<Eigen::internal::UniformRandomGenerator<double>>();
	}

	/**
	 * @brief Creates a zero @c Tensor.
	 * 
	 * In case a @c Tensor is declared, but with no dimensions or there is
	 * a need to change Tensor dimensions, @c ZeroTensorGen can be used. In both
	 * cases the Tensor order cannot be changed. Fills @c tnsX with zero elements.
	 * 
	 * @tparam Array_ 		    An array container type.
	 * @tparam Tensor_          Type(data type and order) of input Tensor.
	 * @param  tnsDims [in]  	Contains the lengths of each of @c tnsX dimensions.
	 * @param  tnsX    [in,out] @c Tensor filled with the data.
	 * 
	 * @note   @c tnsX must be initialized before function call.
	 * 
	 */
	template<typename Array_, typename Tensor_>
	void ZeroTensorGen(Array_ const tnsDims, Tensor_ &tnsX)
	{
		tnsX.resize(tnsDims);
		tnsX.setZero();
	}

	/**
	 * @brief Creates an identity @c Tensor.
	 * 
	 * In case a @c Tensor is declared, but with no dimensions or there is
	 * a need to change Tensor's dimensions, @c IdentityTensorGen can be used. In both
	 * cases the Tensor's order cannot be changed.
	 * Fills @c tnsX as an identity @c Tensor. Meaning that it will have 1-elements 
	 * in the hyperdiagonal and 0-elements in the rest.
	 * 
	 * @note Implementation supports ONLY Tensors with order in range of @c [2,8].
	 *
	 * @tparam Array_ 		    An array container type.
	 * @tparam Tensor_          Type(data type and order) of input Tensor.
	 * @param  tnsDims [in]  	Contains the lengths of each of @c tnsX dimensions.
	 * @param  tnsX    [in,out] @c Eigen Tensor filled with the data.
	 * 
	 * @note   @c tnsX must be initialized before function call.
	 */
	template<typename Array_, typename Tensor_>
	void IdentityTensorGen(Array_ const tnsDims, Tensor_ &tnsX)
	{
		static constexpr std::size_t TnsSize = TensorTraits<Tensor_>::TnsSize;

		const int dim0 = tnsDims[0];
		ZeroTensorGen(tnsDims, tnsX);

		if constexpr(TnsSize == 2) {
			for (int i=0; i<dim0; i++) {
				tnsX(i,i) = 1;
			}
		}
		else if constexpr(TnsSize == 3) {
			for (int i=0; i<dim0; i++) {
				tnsX(i,i,i) = 1;
			}
		}
		else if constexpr(TnsSize == 4) {
			for (int i=0; i<dim0; i++) {
				tnsX(i,i,i,i) = 1;
			}
		}
		else if constexpr(TnsSize == 5) {
			for (int i=0; i<dim0; i++) {
				tnsX(i,i,i,i,i) = 1;
			}
		}
		else if constexpr(TnsSize == 6) {
			for (int i=0; i<dim0; i++) {
				tnsX(i,i,i,i,i,i) = 1;
			}
		}
		else if constexpr(TnsSize == 7) {
			for (int i=0; i<dim0; i++) {
				tnsX(i,i,i,i,i,i,i) = 1;
			}
		}
		else {
			for (int i=0; i<dim0; i++) {
				tnsX(i,i,i,i,i,i,i,i) = 1;
			}
		}
	}

	/* Parallel Version of ReserveSparseTensor */
	template <std::size_t _TnsSize>
	void ReserveSparseTensor(std::array<SparseMatrix, _TnsSize>       &layer_tns_sparse,
							 std::vector<std::vector<int>>      const &local_tns_dimensions,
							 std::array<int,_TnsSize>           const &fiber_rank,
							 int                                const  grid_size,
							 long int                           const  nnz)
	{
		for (std::size_t i = 0; i < _TnsSize; i++)
		{
			long int col = 1;
			for (std::size_t j = 0; j < _TnsSize; j++)
			{
				if (j == i)
					continue;

				col = col * local_tns_dimensions[j][fiber_rank[j]];
			}
			layer_tns_sparse[i].resize(col, local_tns_dimensions[i][fiber_rank[i]]);

			layer_tns_sparse[i].reserve((long int)(nnz / grid_size) + 1);
		}
  	}

	/* Serial Version of ReserveSparseTensor */
	template<std::size_t _TnsSize>
	void ReserveSparseTensor(std::array<SparseMatrix, _TnsSize>       &tns_sparse,
							 std::array<int, _TnsSize>          const &tns_dimensions,
							 const long int                             nnz)
	{
		for (std::size_t i = 0; i < _TnsSize; i++)
		{
			long int col = 1;
			for (std::size_t j = 0; j < _TnsSize; j++)
			{
				if (j == i)
					continue;

				col = col * tns_dimensions[j];
			}

			tns_sparse[i].resize(col, tns_dimensions[i]);
			tns_sparse[i].reserve(nnz);
		}
	}

	// Assign nonzeros to the respective layer_tns_sparse subtensor.
	template <std::size_t _TnsSize>
	void FillSparseTensor(std::array<SparseMatrix, _TnsSize>       &tns_sparse,
						  long int                           const  nnz,
						  Matrix                             const &Ratings_Base_T,
						  std::array<int, _TnsSize>          const &tns_dimensions)
	{
		LongMatrix matr_mapping(static_cast<int>(_TnsSize), static_cast<int>(_TnsSize));
		for (int i = 0; i < static_cast<int>(_TnsSize); i++)
		{
			for (int j = 0, first = 1, prev = 0; j < static_cast<int>(_TnsSize); j++)
			{
				if (j == i)
				{
					matr_mapping(i, j) = 0;
					continue;
				}
				if (first == 1)
				{
					matr_mapping(i, j) = 1;
					first = 0;
					prev = j;
				}
				else
				{
					matr_mapping(i, j) = matr_mapping(i, prev) * tns_dimensions[prev];
					prev = j;
				}
			}
		}

		LongMatrix tuple(1, _TnsSize);

		for (long int nnz_k = 0; nnz_k < nnz; nnz_k++)
		{
			for (int column_idx = 0; column_idx < static_cast<int>(_TnsSize); column_idx++)
			{
				tuple(0, column_idx) = static_cast<long int>(Ratings_Base_T(column_idx, nnz_k));
			}

			for (int mode_i = 0; mode_i < static_cast<int>(_TnsSize); mode_i++)
			{
				long int linear_col = ((matr_mapping.row(mode_i)).cwiseProduct(tuple)).sum();
				long int row        = tuple(0, mode_i);

				if (tns_sparse[mode_i].outerSize() < row || tns_sparse[mode_i].innerSize() < linear_col)
				{
					std::cerr << "error!" << tns_sparse[mode_i].outerSize() << " " << row << " " << tns_sparse[mode_i].innerSize() << " " << linear_col << std::endl;
				}

				if (tns_sparse[mode_i].coeff(linear_col, row) == 0)
				{
					// Using insert to fill sparse matrix
					tns_sparse[mode_i].insert(linear_col, row) = Ratings_Base_T(_TnsSize, nnz_k);
				}
			}
				
		}
	}

	template<std::size_t _TnsSize>
	void FillSparseMatricization(std::array<SparseMatrix, _TnsSize> &tns_sparse,
								 const long int                      nnz,
								 Matrix                             &Ratings_Base_T,
								 std::array<int, _TnsSize>    const &tns_dimensions,
								 const int                           cur_mode)
	{
		LongMatrix matr_mapping(static_cast<int>(_TnsSize), static_cast<int>(_TnsSize));
		for (int i = 0; i < static_cast<int>(_TnsSize); i++)
		{
			for (int j = 0, first = 1, prev = 0; j < static_cast<int>(_TnsSize); j++)
			{
				if (j == i)
				{
					matr_mapping(i, j) = 0;
					continue;
				}
				if (first == 1)
				{
					matr_mapping(i, j) = 1;
					first = 0;
					prev = j;
				}
				else
				{
					matr_mapping(i, j) = matr_mapping(i, prev) * tns_dimensions[prev];
					prev = j;
				}
			}
		}

		LongMatrix tuple(1, static_cast<int>(_TnsSize));

		for (int nnz_k = 0; nnz_k < nnz; nnz_k++)
		{
			for (int column_idx = 0; column_idx < static_cast<int>(_TnsSize); column_idx++)
			{
				tuple(0, column_idx) = static_cast<long int>(Ratings_Base_T(column_idx, nnz_k));
			}

			long linear_col = ((matr_mapping.row(cur_mode)).cwiseProduct(tuple)).sum();

			long row = tuple(0, cur_mode);

			if (tns_sparse[cur_mode].outerSize() < row || tns_sparse[cur_mode].innerSize() < linear_col)
			{
				std::cerr << "error!" << tns_sparse[cur_mode].outerSize() << " " << row << " " << tns_sparse[cur_mode].innerSize() << " " << linear_col << std::endl;
			}

			if (tns_sparse[cur_mode].coeff(linear_col, row) == 0)
			{
				// Using insert to fill sparse matrix
				tns_sparse[cur_mode].insert(linear_col, row) = Ratings_Base_T(static_cast<int>(_TnsSize), nnz_k);
			}
		}
	}	
	
	template <std::size_t _TnsSize>
	void Dist_NNZ(std::array<SparseMatrix, _TnsSize>        &layer_tns_sparse,
				  long int                            const  nnz,
				  std::vector<std::vector<int>>       const &skip_rows,
				  std::array<int,_TnsSize>            const &fiber_rank,
				  Matrix                              const &Ratings_Base_T,
				  std::vector<std::vector<int>>       const &local_tns_dimensions)
	{
		LongMatrix matr_mapping(static_cast<int>(_TnsSize), static_cast<int>(_TnsSize));
		for (int i = 0; i < static_cast<int>(_TnsSize); i++)
		{
			for (int j = 0, first = 1, prev = 0; j < static_cast<int>(_TnsSize); j++)
			{
				if (j == i)
				{
					matr_mapping(i, j) = 0;
					continue;
				}
				if (first == 1)
				{
					matr_mapping(i, j) = 1;
					first = 0;
					prev = j;
				}
				else
				{
					matr_mapping(i, j) = matr_mapping(i, prev) * local_tns_dimensions[prev][fiber_rank[prev]];
					prev = j;
				}
			}
		}

		long int local_nnz_counter = 0;

		LongMatrix tuple(1, static_cast<int>(_TnsSize));

		for (int nnz_k = 0; nnz_k < nnz; nnz_k++)
		{
			for (int column_idx = 0, insert_tuple_flag = 0; column_idx < static_cast<int>(_TnsSize) && insert_tuple_flag == column_idx; column_idx++)
			{
				if((Ratings_Base_T(column_idx, nnz_k) >= skip_rows[column_idx][fiber_rank[column_idx]]) && (Ratings_Base_T(column_idx, nnz_k) < local_tns_dimensions[column_idx][fiber_rank[column_idx]] + skip_rows[column_idx][fiber_rank[column_idx]]))
				{
					tuple(0, column_idx) = (long int)(Ratings_Base_T(column_idx, nnz_k)) - skip_rows[column_idx][fiber_rank[column_idx]];
					insert_tuple_flag++;
				}
				else
				{
					break;
				}
				if (column_idx == static_cast<int>(_TnsSize) - 1)
				{
					local_nnz_counter++;
					for (int mode_i = 0; mode_i < static_cast<int>(_TnsSize); mode_i++)
					{
						long int linear_col = ((matr_mapping.row(mode_i)).cwiseProduct(tuple)).sum();
						long int row = tuple(0, mode_i);

						if (layer_tns_sparse[mode_i].coeff(linear_col, row) == 0)
						{
							layer_tns_sparse[mode_i].insert(linear_col, row) = Ratings_Base_T(static_cast<int>(_TnsSize), nnz_k);
						}
					}
				}
			}
		}
	}

	template <std::size_t _TnsSize>
	void Dist_NNZ_sorted(std::array<SparseMatrix, _TnsSize>        &layer_tns_sparse,
						 long int                            const  nnz,
						 std::vector<std::vector<int>>       const &skip_rows,
						 std::array<int,_TnsSize>            const &fiber_rank,
						 Matrix                              const &Ratings_Base_T,
						 std::vector<std::vector<int>>       const &local_tns_dimensions,
						 int                                 const  cur_mode)
	{
		LongMatrix matr_mapping(static_cast<int>(_TnsSize), static_cast<int>(_TnsSize));
		for (int i = 0; i < static_cast<int>(_TnsSize); i++)
		{
			for (int j = 0, first = 1, prev = 0; j < static_cast<int>(_TnsSize); j++)
			{
				if (j == i)
				{
					matr_mapping(i, j) = 0;
					continue;
				}
				if (first == 1)
				{
					matr_mapping(i, j) = 1;
					first = 0;
					prev = j;
				}
				else
				{
					matr_mapping(i, j) = matr_mapping(i, prev) * local_tns_dimensions[prev][fiber_rank[prev]];
					prev = j;
				}
			}
		}

		long int local_nnz_counter = 0;

		LongMatrix tuple(1, static_cast<int>(_TnsSize));

		for (int nnz_k = 0; nnz_k < nnz; nnz_k++)
		{
			for (int column_idx = 0, insert_tuple_flag = 0; column_idx < static_cast<int>(_TnsSize) && insert_tuple_flag == column_idx; column_idx++)
			{
				if((Ratings_Base_T(column_idx, nnz_k) >= skip_rows[column_idx][fiber_rank[column_idx]]) && (Ratings_Base_T(column_idx, nnz_k) < local_tns_dimensions[column_idx][fiber_rank[column_idx]] + skip_rows[column_idx][fiber_rank[column_idx]]))
				{
					tuple(0, column_idx) = (long int)(Ratings_Base_T(column_idx, nnz_k)) - skip_rows[column_idx][fiber_rank[column_idx]];
					insert_tuple_flag++;
				}
				else
				{
					break;
				}
				if (column_idx == static_cast<int>(_TnsSize) - 1)
				{
					local_nnz_counter++;
					
					long int linear_col = ((matr_mapping.row(cur_mode)).cwiseProduct(tuple)).sum();
					long int row = tuple(0, cur_mode);

					if (layer_tns_sparse[cur_mode].coeff(linear_col, row) == 0)
					{
						layer_tns_sparse[cur_mode].insert(linear_col, row) = Ratings_Base_T(static_cast<int>(_TnsSize), nnz_k);
					}
				}
			}
		}
	}

	template <int TnsSize, int mode, typename Type>
	bool SortRows(const std::vector<Type> &v1, const std::vector<Type> &v2)
	{
		std::array<int, TnsSize> sort_direction;
		bool final_expr{false}; // final criterion for sorting
		bool prev_equal{false}; // keep history of equal comparisons between v1,v2

		sort_direction[0] = mode;
		sort_direction[1] = (mode < TnsSize - 1) ? TnsSize - 1 : TnsSize - 2;
		for (int i = 2; i < TnsSize; i++)
		{
			sort_direction[i] = sort_direction[i - 1] - 1;
			if (sort_direction[i] == mode)
			{
				sort_direction[i]--;
			}
		}
		std::array<bool, TnsSize> expr;
		for (int i = 0; i < TnsSize; i++)
		{
			if (i > 0)
			{
				expr[i] = prev_equal && (v1[sort_direction[i]] < v2[sort_direction[i]]);
				final_expr = final_expr || expr[i];
				prev_equal = prev_equal && (v1[sort_direction[i]] == v2[sort_direction[i]]);

				if (final_expr)
				{
					return final_expr;
				}
			}
			else
			{
				expr[i] = v1[sort_direction[i]] < v2[sort_direction[i]];
				final_expr = final_expr || expr[i];
				prev_equal = (v1[sort_direction[i]] == v2[sort_direction[i]]);
				if (final_expr)
				{
					return final_expr;
				}
			}
		}

		return final_expr;
	}

	/**
     * Shuffles (or permutes) the indices of nonzeros in order to distribute them uniformly.
     * 
     * @param  nnz 						[in]     The nonzeros number.
     * @param  cur_mode 				[in]     The current mode.	 
     * @param  tns_dim					[in]     The current mode tensor dimension. 
     * @param  Ratings_Base_T   		[in]     The input matrix which containts all nonzeros.
     * @param  perm_tns_indices    		[in,out] @c Stl array containing the Tensor indices (vector), 
	 * 											 which will pe permuted.
     * @param  Balanced_Ratings_Base_T  [in/out] The ouput matrix which containts all permuted nonzeros.
     */
	void PermuteModeN(long int              const  nnz,
					  int                   const  cur_mode,
					  int                   const  tns_dim,
					  Matrix                const &Ratings_Base_T,
					  std::vector<long int>       &perm_tns_indices,
					  Matrix                      &Balanced_Ratings_Base_T)

	{
		// Copy values
		Balanced_Ratings_Base_T = Ratings_Base_T;

		// Allocate & Initialize permuted dims
		perm_tns_indices.reserve(tns_dim);

		for (int i_i = 0; i_i < tns_dim; i_i++)
		{
			perm_tns_indices.push_back(i_i);
		}

		// Permute dims
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(perm_tns_indices.begin(), perm_tns_indices.end(), g);
		double prev = -1;
		long int idx;
		std::vector<long int>::iterator it;
		long int perm_idx = 0;
		for (long int nnz_i = 0; nnz_i < nnz; nnz_i++)
		{
			if(Ratings_Base_T(cur_mode, nnz_i) != prev)
			{
				idx = static_cast<long int>(Ratings_Base_T(cur_mode, nnz_i));
				it = std::find(perm_tns_indices.begin(), perm_tns_indices.end(), idx);
				perm_idx = it - perm_tns_indices.begin();
				Balanced_Ratings_Base_T(cur_mode, nnz_i) = static_cast<double>(perm_idx);
				prev = Ratings_Base_T(cur_mode, nnz_i);
			}
			else
			{
				Balanced_Ratings_Base_T(cur_mode, nnz_i) = static_cast<double>(perm_idx);
				prev = Ratings_Base_T(cur_mode, nnz_i);
			}
		}
	}

	/**
     * Shuffles (or permutes) the indices of nonzeros in order to distribute them uniformly.
     * 
     * @tparam TnsSize         Tensor Order. 
     * 
     * @param  nnz 						[in]     The nonzeros number.
     * @param  tns_dimensions			[in]     @c Stl array containing the Tensor dimensions, whose
     *                              			 length must be same as the Tensor order.
     * @param  Ratings_Base_T   		[in]     The input matrix which containts all nonzeros.
     * @param  perm_tns_indices    		[in,out] @c Stl array containing the Tensor indices (vector), 
	 * 											 which will pe permuted.
     * @param  Balanced_Ratings_Base_T  [in/out] The ouput matrix which containts all permuted nonzeros.
     */
	template <std::size_t TnsSize>
	void BalanceDataset(long int  									 const  nnz,
						std::array<int, TnsSize> 					 const  tns_dimensions,
						Matrix 										 const &Ratings_Base_T,
						std::array<std::vector<long int>, TnsSize> 	       &perm_tns_indices,
						Matrix 							   				   &Balanced_Ratings_Base_T)
	{
		for (int i = 0; i < static_cast<int>(TnsSize); i++)
		{
			PermuteModeN(nnz, i, tns_dimensions[i], Ratings_Base_T, perm_tns_indices[i], Balanced_Ratings_Base_T);
		}
	}

	/**
     * Permute rows of factors according to shuffled indices perm_tns_indices.
     * 
     * @tparam TnsSize         Tensor Order. 
     * 
     * @param  depermuted_factors 	[in]     	The input factors.
     * @param  perm_tns_indices  	[in] 	 	@c Stl array containing the Tensor indices (vector), 
	 * 										 	which will pe permuted.
     * @param  permuted_factors_T  	[in/out] 	The output factors, whose rows are permuted.
     */
	template <std::size_t TnsSize>
	void PermuteFactors(std::array<Matrix, TnsSize> 			   const &depermuted_factors,
						std::array<std::vector<long int>, TnsSize> const &perm_tns_indices,
						std::array<Matrix, TnsSize> 				     &permuted_factors_T)
	{
		Matrix temp_permuted_factor;
		for (int i = 0; i < static_cast<int>(TnsSize); i++)
		{
			temp_permuted_factor = Matrix::Zero(depermuted_factors[i].rows(), depermuted_factors[i].cols());
			for (int row = 0; row < temp_permuted_factor.rows(); row++)
			{
				temp_permuted_factor.row(row) = depermuted_factors[i].row(perm_tns_indices[i][row]);
			}
			permuted_factors_T[i] = temp_permuted_factor.transpose();
		}
	}

	/**
     * Depermute rows of factors according to shuffled indices perm_tns_indices.
     * 
     * @tparam TnsSize         Tensor Order. 
     * 
     * @param  depermuted_factors 	[in]     	The input factors, whose rows are permuted.
     * @param  perm_tns_indices  	[in] 	 	@c Stl array containing the Tensor indices (vector), 
	 * 										 	which will pe permuted.
     * @param  permuted_factors  	[in/out] 	The output factors, whose rows are depermuted.
     */
	template <std::size_t TnsSize>
	void DepermuteFactors(std::array<Matrix, TnsSize>                const &permuted_factors,
						  std::array<std::vector<long int>, TnsSize> const &perm_tns_indices,
						  std::array<Matrix, TnsSize>                      &depermuted_factors)
	{
		for (int i = 0; i < static_cast<int>(TnsSize); i++)
		{
			depermuted_factors[i] = Matrix::Zero(permuted_factors[i].rows(), permuted_factors[i].cols());
			for (int row = 0; row < permuted_factors[i].rows(); row++)
			{
				depermuted_factors[i].row(perm_tns_indices[i][row]) = permuted_factors[i].row(row);
			}
		}
	}

} // end namespace partensor

#endif // PARTENSOR_TENSOR_OPERATIONS_HPP
