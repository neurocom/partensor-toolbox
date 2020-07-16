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
						 Dimensions const &tnsDims )
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

} // end namespace partensor

#endif // PARTENSOR_TENSOR_OPERATIONS_HPP
