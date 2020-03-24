#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
 * @date      25/03/2019
 * @author    Christos Tsalidis
 * @author    Yiorgos Lourakis
 * @author    George Lykoudis
 * @copyright 2019 Neurocom All Rights Reserved.
 */
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      DataGeneration.hpp
* @details
* Includes a variety of functions, that either create synthetic
* data or reform them.
*********************************************************************/

#ifndef PARTENSOR_DATA_GENERATION_HPP
#define PARTENSOR_DATA_GENERATION_HPP

#include "PARTENSOR_basic.hpp"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h"
#include "PartialKhatriRao.hpp"
#include "TensorOperations.hpp"
#include "DimTrees.hpp"
#include "Constants.hpp"
// #include "Normalize.hpp"
#include <time.h>
#include <cassert>

// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

namespace partensor
{

	/**
	 * @brief Fills a @c Matrix with pseudo random data.
	 * 
	 * Generate synthetic-random Matrix in uniform distribution with 
	 * numbers in @c [0,1].
	 * 
	 * @param  mtx [in,out] The @c Matrix, where the data 
	 *                      will be stored.
	 * 
	 * @note @c mtx must initiallized before the function call.
	 */
	void generateRandomMatrix(Matrix &mtx)
	{
		int m = mtx.rows();
		int n = mtx.cols();
		assertm(m>0 && n>0, "Rows and columns must be greater than 1 in generateRandomMatrix\n");

		std::srand((unsigned int) time(NULL)+std::rand());

		mtx = (Matrix::Random(m, n) + Matrix::Ones(m, n))/2;
	}

	/**
	 * @brief Fills a @c Tensor with pseudo random data.
	 * 
	 * Generates synthetic-random data for a @c Tensor based on 
	 * a distribution. The distribution can be either uniform or normal.
	 * 
	 * @tparam Tensor_               Type(data type and order) of input Tensor @c tnsX.
     *                               @c Tensor_ must be @c Tensor<order>, where @c order must be in 
	 *                               range of @c [3-8].
	 * @param  tnsX         [in,out] The given Tensor to be filled with data, based on @c distribution.
	 * @param  distribution [in]     If 0 data the data are chosen from a Uniform distribution in
	 *                               range @c [0,1], else in Normal distribution with @c mean=0 and 
	 *                               standard @c deviation(σ)=1. The default is Uniform distribution.
	 * 
	 * @note @c tnsX must be initialized before the function call.
	 */
	template<typename Tensor_>
	void generateRandomTensor( Tensor_        &tnsX,
	                           unsigned const  distribution = 0 )
	{
		using DataType = typename TensorTraits<Tensor_>::DataType;
		std::srand((unsigned int) time(NULL)+std::rand());

		if(distribution == 0)// uniform distribution with numbers in [0,1]
		{
			tnsX.template setRandom<Eigen::internal::UniformRandomGenerator<DataType>>();
		}
		else if (distribution == 1) // normal distribution with mean = 0 and deviation(σ) = 1
		{
			tnsX.template setRandom<Eigen::internal::NormalRandomGenerator<DataType>>();
		}
		else
		{
			throw std::runtime_error("Choose correct distribution in generateRandomTensor().\n");
		}
	}

	/**
	 * @brief Computes the matriced Tensor from an array of factors.
	 * 
	 * If there are factors saved in an @c stl array, then 
	 * @c generateTensor, can be used to produce the matricized Tensor.
	 * It computes the matricized Tensor, from the @c Khatri-Rao product 
	 * of factors, based on the chosen @c mode for the matricization.
	 * 
	 * @tparam MatrixArray      An array container with Matrices of
	 *                          @c Matrix type.
	 * @param  mode 	   [in] The dimension, in which the matricized Tensor 
	 *                          will be returned.
	 * @param  factorArray [in] The array with the factors of 
	 *                          @c Matrix type, where its size
	 *                          must be equal to Tensor order.
	 * 
	 * @returns The Tensor matricization based on the factors.
	 */
	template<std::size_t Size>
  	Matrix generateTensor( std::size_t             const  mode, 
	  					   std::array<Matrix,Size> const &factorArray)
	{
		if (mode > Size-1) { throw std::runtime_error("Mode must be in [0,factorArray.Size) in generateTensor().\n"); }

		Matrix partial_krao = PartialKhatriRao(factorArray, mode);
		return (factorArray[mode] * partial_krao.transpose());
	}

	/**
	 * @brief Computes the Tensor from an array of factors.
	 * 
	 * If there are factors saved in an @c stl array, then 
	 * @c generateTensor, can be used to produce the matricized Tensor.
	 * It computes the matricized Tensor, from contraction of all
	 * factors in @c factorArray.
	 * 
	 * @tparam _TnsSize             Tensor Order of the Tensor.
	 * @param  factorArray [in]     An @c stl array with all Factors of type
	 *                              @c Tensor<2>.
	 * 
	 * @returns The generated Tensor from the @c factorArray.
   	 */
   	template<std::size_t _TnsSize>
	Tensor<static_cast<int>(_TnsSize)> generateTensor( std::array<Tensor<2>,_TnsSize> &factorArray)
	{
		static_assert(_TnsSize>0, "Tensor cannot be scalar in generateTensor()!\n");

		using MatrixArray = std::array<Tensor<2>,_TnsSize>;
		// same for all factors
		const std::size_t R = factorArray[0].dimension(1); 
		// Initialize Core Tensor for PARAFAC
		std::array<int,_TnsSize> dim;
		int                      i = 0;
		constexpr int            w = 1;

		Tensor<static_cast<int>(_TnsSize)>    tnsX;
		Tensor<static_cast<int>(_TnsSize)>    Temp_X;

		std::fill(dim.begin(), dim.end(), R);
		IdentityTensorGen(dim, tnsX);

		std::array<Eigen::IndexPair<int>, 1> product_dims;

		for(typename MatrixArray::reverse_iterator it=factorArray.rbegin(); it != factorArray.rend(); ++it)
		{
			product_dims = { Eigen::IndexPair<int>(w,i) };

			Temp_X = (*it).contract(tnsX, product_dims);
			tnsX   = Temp_X;
			i++;
		}
		return tnsX;
	}

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/** 
	 * Implementation of @c fillDimTreeFactors, in case @c factorArray containts data
	 * in @c Matrix type.
	 * 
	 * @tparam idx                         Indexing for the @c factorDimTreeArray.
	 * @tparam _TnsSize                    Length of the @c stl arrays, based on the number 
	 *                                     of factors.
	 * @param  factorArray        [in]     Contains the factors data. The type of this @c stl
	 *                                     array is @c Matrix and has length @c _TnsSize. 
	 * @param  constraints        [in]     An @c stl array with the constraint that has been
	 *                                     applied to each factor of @c factorArray.
	 * @param  factorDimTreeArray [in,out] The @c stl array that will be returned and contains 
	 *                                     the copied data of each factor in a @c FactorDimTree 
	 *                                     struct.
	 */
	template <std::size_t idx, std::size_t _TnsSize>
	void fillDimTreeFactors_Matrix( std::array<Matrix, _TnsSize>        const &factorArray, 
									std::array<Constraint, _TnsSize>    const &constraints,
									std::array<FactorDimTree, _TnsSize>       &factorDimTreeArray )
	{
		const Matrix                          &_factor        = factorArray[idx];
		FactorDimTree                         &_factorDimTree = factorDimTreeArray[idx];
		std::array<Eigen::IndexPair<int>, 1>   product_dims   = { Eigen::IndexPair<int>(0, 0) };

		const int rows = _factor.rows();
		const int cols = _factor.cols();

		_factorDimTree.factor.resize(rows,cols);
		_factorDimTree.gramian.resize(cols,cols);
		_factorDimTree.constraint = constraints[idx];
		_factorDimTree.factor     = matrixToTensor(_factor, rows, cols);
		_factorDimTree.gramian    = _factorDimTree.factor.contract(_factorDimTree.factor, product_dims);

		if constexpr (idx+1 < _TnsSize)
		{
			fillDimTreeFactors_Matrix<idx + 1, _TnsSize>(factorArray, constraints, factorDimTreeArray);
		}
	}
	#endif  // DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/** 
	 * Implementation of @c fillDimTreeFactors, in case @c factorArray containts data
	 * in @c Tensor<2> type.
	 * 
	 * @tparam idx                         Indexing for the @c factorDimTreeArray.
	 * @tparam _TnsSize                    Length of the @c stl arrays, based on the number 
	 *                                     of factors.
	 * @param  factorArray        [in]     Contains the factors data. The type of this @c stl
	 *                                     array is @c Tensor<2> and has 
	 *                                     length @c _TnsSize. 
	 * @param  constraints        [in]     An @c stl array with the constraint that has been
	 *                                     applied to each factor of @c factorArray.
	 * @param  factorDimTreeArray [in,out] The @c stl array that will be returned and contains 
	 *                                     the copied data of each factor in a @c FactorDimTree 
	 *                                     struct.
	 */
	template <std::size_t idx, std::size_t _TnsSize>
	void fillDimTreeFactors_Tensor( std::array<Tensor<2>, _TnsSize>     const &factorArray, 
									std::array<Constraint, _TnsSize>    const &constraints,
									std::array<FactorDimTree, _TnsSize>       &factorDimTreeArray )
	{
		const Tensor<2>                       &_factor        = factorArray[idx];
		FactorDimTree                         &_factorDimTree = factorDimTreeArray[idx];
		std::array<Eigen::IndexPair<int>, 1>   product_dims   = { Eigen::IndexPair<int>(0, 0) };

		const int rows = _factor.dimension(0); // rows
		const int cols = _factor.dimension(1); // cols

		_factorDimTree.factor.resize(rows,cols);
		_factorDimTree.gramian.resize(cols,cols);
		_factorDimTree.constraint = constraints[idx];
		_factorDimTree.factor     = _factor;
		_factorDimTree.gramian    = _factorDimTree.factor.contract(_factorDimTree.factor, product_dims);

		if constexpr (idx+1 < _TnsSize)
		{
			fillDimTreeFactors_Tensor<idx + 1, _TnsSize>(factorArray, constraints, factorDimTreeArray);
		}
	}
	#endif  // DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * @brief Copy input data to an @c stl array with @c FactorDimTree type.
	 * 
	 * Can be used to fill an @c stl array with length @c _TnsSize and @c FactorDimTree
	 * type. The data come from two @c stl arrays @c factorArray and @c constraints.
	 * The first one has the factors of @c FactorType type. They can be either 
	 * @c Matrix or @c Tensor<2>. Also, the @c constraints is necessary, with values  
	 * of type @c Constraint from @c Constants.hpp.
	 * 
	 * @tparam _TnsSize                    Length of the @c stl arrays, based on the number 
	 *                                     of factors.
	 * @tparam FactorType                  The type of @c stl @c factorArray that can be
	 *                                     either @c Matrix or @c Tensor<2>.
	 * @param  factorArray        [in]     Contains the factors data. The type of this @c stl
	 *                                     array is @c FactorType and has length @c _TnsSize. 
	 * @param  constraints        [in]     An @c stl array with the constraint that has been
	 *                                     applied to each factor of @c factorArray.
	 * @param  factorDimTreeArray [in,out] The @c stl array that will be returned and contains 
	 *                                     the copied data of each factor in a @c FactorDimTree 
	 *                                     struct.
	 */
	template <std::size_t _TnsSize, typename FactorType>
	void fillDimTreeFactors( std::array<FactorType, _TnsSize> const &factorArray, 
	                         std::array<Constraint, _TnsSize> const &constraints,
							 std::array<FactorDimTree, _TnsSize>    &factorDimTreeArray )
	{
		constexpr bool check = (std::is_same_v<FactorType,Matrix> || 
		                        std::is_same_v<FactorType,Tensor<2>>);
		static_assert(check, "Factors must be of type Matrix or Tensor<2>, fillDimTreeFactors()\n");

		if constexpr (std::is_same_v<FactorType,Matrix>)
			fillDimTreeFactors_Matrix<0,_TnsSize>(factorArray, constraints, factorDimTreeArray);
		else
		    fillDimTreeFactors_Tensor<0,_TnsSize>(factorArray, constraints, factorDimTreeArray);
	}
	#endif  // DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * @brief Forms an @c Eigen Tensor from factors.
	 * 
	 * If there are factors saved in an @c stl array, then @c CpdGen, 
	 * can be used to produce an @c Eigen Tensor, from them. It computes 
	 * the Tensor, using @c Eigen @c contract function, among the factors, 
	 * that are @c FactorDimTree type.
	 * 
	 * @tparam _TnsSize             Tensor Order of @c tnsX.
	 * @param  FactorArray [in]     An @c stl array with all Factors of type
	 *                              @c FactorDimTree.
	 * @param  R           [in]     Rank of factorization (Number of columns in each 
	 *                              factor).
	 * @param  tnsX        [in,out] Generated Tensor from the @c FactorArray.
   	 */
   	template<std::size_t _TnsSize>
	void CpdGen( std::array<FactorDimTree,_TnsSize>       &FactorArray, 
				 int 							    const  R, 
				 Tensor<static_cast<int>(_TnsSize)>       &tnsX)
	{
		static_assert(_TnsSize>0, "Tensor cannot be scalar in CpdGen()!\n");

		assertm(R>0, "Variable R - factor column must be greater than one in CpdGen()!\n");

		using MatrixArray = std::array<FactorDimTree,_TnsSize>;
		// Initialize Core Tensor for PARAFAC
		std::array<int,_TnsSize> dim;
		int                      i = 0;
		constexpr int            w = 1;

		Tensor<static_cast<int>(_TnsSize)>   Temp_X;
		std::array<Eigen::IndexPair<int>, 1> product_dims;

		std::fill(dim.begin(), dim.end(), R);
		IdentityTensorGen(dim, tnsX);

		for(typename MatrixArray::reverse_iterator it=FactorArray.rbegin(); it != FactorArray.rend(); ++it)
		{
			product_dims = { Eigen::IndexPair<int>(w,i) };

			Temp_X = (it->factor).contract(tnsX, product_dims);
			tnsX   = Temp_X;
			i++;
		}
	}
	#endif  // DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Implementation for the generation of an Array of Factors of @c Matrix type.
	 * 
	 * @tparam idx                  Indexing for the arrays @c tnsDims, @c constraints 
	 *                              and @c factorArray.
	 * @tparam _TnsSize				Tensor Order.
	 * @tparam Dimensions			Type of an array containing the @c Tensor dimensions.
	 * @param tnsDims     [in]      The row dimension for each Factor based on Tensor dimensions.
	 * @param constraints [in]      The @c Constraint to apply to each Factor of type @c Matrix. 
	 * @param R           [in]      Rank of factorization (Number of columns in each Matrix of 
	 *                              @c factorArray ).
	 * @param factorArray [int,out] An @c stl array containing all Factors of @c Matrix Type.
	 * 
	 */
	template<std::size_t idx, std::size_t _TnsSize, typename Dimensions>
	void makeFactors_Matrix( Dimensions 					 const &tnsDims,
							 std::array<Constraint,_TnsSize> const &constraints,
							 int         					 const  R,
							 std::array<Matrix,_TnsSize>           &factorArray  )
	{
		Matrix  &_factor = factorArray[idx];
		_factor.resize(tnsDims[idx], R);
		generateRandomMatrix(_factor);
		
		switch(constraints[idx])
		{
			case Constraint::unconstrained:
			{
				break;
			}
			case Constraint::nonnegativity:
			{
				Matrix Zeros = Matrix::Zero(_factor.rows(),_factor.cols());
				_factor = _factor.cwiseMax(Zeros);
				break;
			}
			case Constraint::orthogonality:
			{
				Eigen::JacobiSVD<Matrix> svd(_factor, Eigen::ComputeThinU | Eigen::ComputeThinV);
				_factor = svd.matrixU();
				break;
			}
			case Constraint::sparsity:
			{
				break;
			}
			default: // in case of Constraint::constant
			{
				break;
			}
		}

		if constexpr (idx+1 < _TnsSize)
		{
			makeFactors_Matrix<idx + 1, _TnsSize>(tnsDims, constraints, R, factorArray);
		}
	}
	#endif  // DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Implementation for the generation of an Array of Factors of @c Tensor<2> type.
	 * 
	 * @tparam idx                   Indexing for the arrays @c tnsDims, @c constraints 
	 *                               and @c factorArray.
	 * @tparam _TnsSize				 Tensor Order.
	 * @tparam Dimensions			 Type of an array containing the Tensor dimensions.
	 * @param  tnsDims     [in]      The row dimension for each @c Tensor<2> based on 
	 *                               Tensor dimensions.
	 * @param  constraints [in]      The @c Constraint to apply to each Factor of type 
	 *                               @c Tensor<2>.
	 * @param  R           [in]      Rank of factorization (Number of columns in each 
	 *                               @c Tensor<2> of @c factorArray).
	 * @param  factorArray [int,out] An @c stl array containing all Factors of @c Tensor<2> 
	 *                               Type.
	 * 
	 */
	template<std::size_t idx, std::size_t _TnsSize, typename Dimensions>
	void makeFactors_Tensor( Dimensions                       const &tnsDims,
							 std::array<Constraint,_TnsSize>  const &constraints,
							 int                              const  R,
							 std::array<Tensor<2>,_TnsSize>         &factorArray )
	{
		Tensor<2>  &_factor = factorArray[idx];

		_factor.resize(tnsDims[idx],R);
		generateRandomTensor(_factor);

		switch(constraints[idx])
		{
			case Constraint::unconstrained:
			{
				break;
			}
			case Constraint::nonnegativity:
			{
				_factor = (_factor.abs()).eval();
				break;
			}
			case Constraint::orthogonality:
			{
				Matrix _mtx = tensorToMatrix(_factor, tnsDims[idx], R);

				Eigen::JacobiSVD<Matrix> svd(_mtx, Eigen::ComputeThinU | Eigen::ComputeThinV);
				_mtx = svd.matrixU();

				_factor = matrixToTensor(_mtx, tnsDims[idx], R);
				break;
			}
			case Constraint::sparsity:
			{
				break;
			}
			default: // in case of Constraint::constant
			{
				break;
			}
		}

		if constexpr (idx+1 < _TnsSize)
		{
			makeFactors_Tensor<idx + 1, _TnsSize>(tnsDims, constraints, R, factorArray);
		}
	}
	#endif // DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Implementation for the generation of an Array of Factors of @c FactorDimTree type.
	 * 
	 * @tparam idx                  Indexing for the arrays @c tnsDims, @c constraints 
	 *                              and @c factorArray.
	 * @tparam _TnsSize				Tensor Order.
	 * @tparam Dimensions			Type of an array containing the Tensor dimensions.
	 * @param tnsDims     [in]      The row dimension for each @c FactorDimTree based on Tensor dimensions.
	 * @param constraints [in]      The @c Constraint to apply to each Factor of type @c FactorDimTree.
	 * @param R           [in]      Rank of factorization (Number of columns in each @c FactorDimTree 
	 * 								of @c factorArray).
	 * @param factorArray [int,out] An @c stl array containing all Factors of @c FactorDimTree Type.
	 * 
	 */
	template<std::size_t idx, std::size_t _TnsSize, typename Dimensions>
	void makeFactors_DimTree( Dimensions                         const &tnsDims,
							  std::array<Constraint,_TnsSize>    const &constraints,
							  int                                const  R,
							  std::array<FactorDimTree,_TnsSize>       &factorArray )
	{
		FactorDimTree                        &_factor      = factorArray[idx];
		std::array<Eigen::IndexPair<int>, 1>  product_dims = { Eigen::IndexPair<int>(0, 0) };

		_factor.factor.resize(tnsDims[idx],R);
		_factor.gramian.resize(R,R);
		_factor.constraint = constraints[idx];
		
		generateRandomTensor(_factor.factor);

		switch(constraints[idx])
		{
			case Constraint::unconstrained:
			{
				break;
			}
			case Constraint::nonnegativity:
			{
			_factor.factor = (_factor.factor.abs()).eval();
			break;
			}
			case Constraint::orthogonality:
			{
				Matrix _mtx = tensorToMatrix(_factor.factor, tnsDims[idx], R);

				Eigen::JacobiSVD<Matrix> svd(_mtx, Eigen::ComputeThinU | Eigen::ComputeThinV);
				_mtx = svd.matrixU();

				_factor.factor = matrixToTensor(_mtx, tnsDims[idx], R);
			break;
			}
			case Constraint::sparsity:
			{
				break;
			}
			default: // in case of Constraint::constant
			{
				break;
			}
		}
		_factor.gramian = _factor.factor.contract(_factor.factor, product_dims);

		if constexpr (idx+1 < _TnsSize)
		{
			makeFactors_DimTree<idx + 1, _TnsSize>(tnsDims, constraints, R, factorArray);
		}
	}
	#endif  // DOXYGEN_SHOULD_SKIP_THIS

	/**
	 * @brief Creates an @c stl array, where Matrices-Factors are stored.
	 * 
	 * Creates a pseudo-random @c stl array with Matrices-Factors. These
	 * factors can be of type either @c Matrix or @c FactorDimTree.
	 * Also, there can be applied different @c constraints to each factor, 
	 * specified in @c Constraint enumeration from @c Constants.hpp.
	 * 
	 * An array container with the dimension per factor is needed and the 
	 * variable @c R, which indicates the number of columns of each factor.
	 * 
	 * @tparam _TnsSize 			 Essentially tensor order, but also the size of 
	 *                               @c constraints and @c factorArray arrays.
	 * @tparam Dimensions			 Array container for @c tnsDims.
	 * @tparam FactorType			 The type for @c factorArray and the generated Factors, 
	 *                               either @c Eigen Matrix or @c FactorDimTree.
	 * @param  tnsDims     [in]      The row dimension for each factor.
	 * @param  constraints [in]      The @c Constraint to apply to each Factor, check 
	 *                               @c Constants.hpp.
	 * @param  R           [in]      Rank of factorization (Number of columns in each @c Matrix
	 *                               or @c FactorDimTree).
	 * @param  factorArray [int,out] An @c stl array containing all factors with type @c FactorType.
	 */
	template<std::size_t _TnsSize, typename Dimensions, typename FactorType>
	void makeFactors( Dimensions                      const &tnsDims,
					  std::array<Constraint,_TnsSize> const &constraints,
					  int                             const  R,
					  std::array<FactorType,_TnsSize>       &factorArray )
	{
		constexpr bool check = (std::is_same_v<FactorType,Matrix> || 
		                        std::is_same_v<FactorType,Tensor<2>> || 
								std::is_same_v<FactorType,FactorDimTree>);
		static_assert(check, "Factors must be of type Matrix, Tensor<2> or FactorDimTree, makeFactors()\n");

		if constexpr (std::is_same_v<FactorType,Matrix>)
			makeFactors_Matrix<0, _TnsSize>(tnsDims, constraints, R, factorArray);
		else if constexpr (std::is_same_v<FactorType,Tensor<2>>)
			makeFactors_Tensor<0, _TnsSize>(tnsDims, constraints, R, factorArray);
		else
			makeFactors_DimTree<0, _TnsSize>(tnsDims, constraints, R, factorArray);			
	}
	
	/**
	 * @brief Initialize a Tensor with constraints applied.
	 * 
	 * Creates a tensor @c tnsX, with dimensions specified in @c tnsDims.
	 * In order to create @c tnsX, some factors will be created. The
	 * number of factors being created is equal to @c tnsDims size. 
	 * On each factor can be applied a constraint of type @c Constraint. 
	 * Check @c Constants.hpp for the other constraints. Default value
	 * is @c nonnegative constraint.
	 * Also, the rank @c R, of these factors is needed.
	 * 
	 * @tparam _TnsSize             Size of the @c tnsDims, @c constraints arrays
	 *                              and the number of @c tnsX dimensions.
	 * @param  tnsDims     [in]     @c stl array with each dimension for @c tnsX.
	 * @param  constraints [in]     The constraints to be applied in on each factor
	 *                              that will be used to generate @c tnsX.
	 * @param  R           [in]     Essentially, the number of columns for each
	 *                              factor.
	 * @param  tnsX        [in,out] The tensor to be created based on the other
	 *                              parameters. It is a @c Tensor type.
	 */
	template <std::size_t _TnsSize>
	void makeTensor( std::array<int,_TnsSize>           const &tnsDims,
	                 std::array<Constraint,_TnsSize>    const &constraints,
	                 std::size_t                        const  R,
					 Tensor<static_cast<int>(_TnsSize)>       &tnsX  )
	{
		static_assert(_TnsSize>0, "Tensor cannot be scalar in makeTensor()!\n");

		assertm(R>0, "Variable R - factor column must be greater than one in makeTensor()!\n");

		using MatrixArray = typename TensorTraits<Tensor<_TnsSize>>::MatrixArray;
		
		MatrixArray  true_factors;
        MatrixArray  gramians;
		Matrix       matricized_tensor;
		
		tnsX.resize(tnsDims);
		makeFactors(tnsDims, constraints, R, true_factors);
		for (std::size_t i=0; i<_TnsSize; ++i)
        {
			gramians[i].noalias() = true_factors[i].transpose() * true_factors[i];
        }
		// Normalize(R, gramians, true_factors);
		matricized_tensor = generateTensor(0, true_factors);
  		tnsX              = matrixToTensor(matricized_tensor, tnsDims);
	}
	
	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * @brief Initialize a Tensor with constraints applied.
	 * 
	 * Creates a tensor @c tnsX, with dimensions specified in @c tnsDims.
	 * In order to create @c tnsX, some factors will be created. The
	 * number of factors being created is equal to @c tnsDims size. 
	 * On each factor can be applied a constraint of type @c Constraint. 
	 * Check @c Constants.hpp for the other constraints. Default value
	 * is @c nonnegative constraint.
	 * Also, the rank @c R, of these factors is needed.
	 * 
	 * @tparam _TnsSize               Size of the @c tnsDims, @c constraints arrays
	 *                                and the number of @c tnsX dimensions.
	 * @param  tnsDims      [in]      @c stl array with each dimension for @c tnsX.
	 * @param  constraints  [in]      The constraints to be applied in on each factor
	 *                                that will be used to generate @c tnsX.
	 * @param  R            [in]      Essentially, the number of columns for each
	 *                                factor.
	 * @param  true_factors [int,out] An @c stl array containing all factors used to create
	 *                                the factor.
	 * @param  tnsX         [in,out]  The tensor to be created based on the other
	 *                                parameters. It is a @c Tensor type.
	 */
	template <std::size_t _TnsSize>
	void makeTensor( std::array<int,_TnsSize>           const &tnsDims,
	                 std::array<Constraint,_TnsSize>    const &constraints,
	                 std::size_t                        const  R,
					 std::array<Matrix,_TnsSize>              &true_factors,
					 Tensor<static_cast<int>(_TnsSize)>       &tnsX)
	{
		static_assert(_TnsSize>0, "Tensor cannot be scalar in makeTensor()!\n");

		assertm(R>0, "Variable R - factor column must be greater than one in makeTensor()!\n");

		using MatrixArray = typename TensorTraits<Tensor<_TnsSize>>::MatrixArray;
		
        MatrixArray  gramians;
		Matrix       matricized_tensor;
		
		tnsX.resize(tnsDims);
		makeFactors(tnsDims, constraints, R, true_factors);  

		for (std::size_t i=0; i<_TnsSize; ++i)
        {
			gramians[i].noalias() = true_factors[i].transpose() * true_factors[i];
        }
		
		matricized_tensor = generateTensor(0, true_factors);
  		tnsX              = matrixToTensor(matricized_tensor, tnsDims);
	}
	#endif  // DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Generate synthetic-random Eigen Matrices - Factors based on a distribution.
	 * 
	 * @tparam Dimensions   	     An array of dimensions for each FactorDimTree.
	 * @tparam MatrixArray           An array with Type of Eigen Matrices.
	 * @param  rank         [in]     The rank of the Tensor factorization.
	 * @param  tnsDims 	    [in]     Row - Dimension of each FactorDimTree.
	 * @param  distribution [in]
	 * - 0 for data in Uniform [-1,1],
	 * - 1 for data in Uniform [0,1],
	 * - 2 for Normal with mean = 0 and deviation(σ) = 1.
	 * @param factorArray   [in,out] An array with the Eigen Matrices - Factors 
	 * 							     containing pseudorandom data in generated 
	 * 								 based the @c distribution.
	 */
	template<typename Dimensions, typename MatrixArray>
	void generateFactors( std::size_t const  rank, 
	                      Dimensions  const &tnsDims, 
						  unsigned    const  distribution, 
						  MatrixArray       &factorArray)
	{
		using Matrix = typename MatrixArrayTraits<MatrixArray>::value_type;

		constexpr std::size_t TnsSize = MatrixArrayTraits<MatrixArray>::Size;

		std::srand((unsigned int) time(NULL)+std::rand());
		switch(distribution)
		{
			case 0: // uniform distribution with numbers in [-1,1]
			{
				for(std::size_t i=0; i<TnsSize; i++) { factorArray[i] = Matrix::Random(tnsDims[i], rank); }
				break;
			}
			case 1: // uniform distribution with numbers in [0,1]
			{
				for(std::size_t i=0; i<TnsSize; i++) { factorArray[i] = (Matrix::Random(tnsDims[i], rank) + Matrix::Ones(tnsDims[i], rank))/2; }
				break;
			}
			case 2: // normal distribution with mean = 0 and deviation(σ) = 1
			{
				std::random_device rd;
				std::mt19937 e2(rd());
				std::normal_distribution<> dist(0.0, 1.0);
				for(std::size_t i=0; i<TnsSize; i++)
				{
					factorArray[i] = Matrix(tnsDims[i],rank);
	        		for(std::size_t j=0; j<static_cast<std::size_t>(tnsDims[i]); j++)
					{
						for(std::size_t k=0; k<rank; k++)
						{
	            			factorArray[i](j,k) = dist(e2);
						}
	        		}
		    	}
				break;
			}
			default:
			{
				break;
			}
		}
	}
	#endif // DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Creates an @c Eigen Tensor with data starting from 0 until
	 * the product of its dimensions. The implementation
	 * is for Tensor of size 3,4 or 5 (e.g if tensor(2,3,4) then
	 * its values are 0,2,...,(2*3*4 - 1)).
	 * @tparam Dimensions       Array Type.
	 * @tparam Tensor_          Type of @c Eigen Tensor (float, double, etc.).
	 * @param  tns_dim [in]     The Tensor dimensions.
	 * @param  tnsX    [in,out] The @c Eigen Tensor that will contains the data
	 *                          and must be initialized before calling the function.
	 */
	template<typename  Dimensions, typename Tensor_>
	void customTensor( Dimensions const &tns_dim, 
					   Tensor_          &tnsX    )
	{
		static constexpr std::size_t TnsSize = TensorTraits<Tensor_>::TnsSize;

		tnsX.resize(tns_dim);
		int count = 0;

		if constexpr (TnsSize == 3)
		{
			for(int k=0; k<tns_dim[2]; k++)
			{
				for(int j=0; j<tns_dim[1]; j++)
				{
					for(int i=0; i<tns_dim[0]; i++)
					{
							tnsX(i,j,k) = count+1;
							count++;
					}
				}
			}
		}
		else if constexpr (TnsSize == 4)
		{
			for(int l=0; l<tns_dim[3]; l++)
      {
        for(int k=0; k<tns_dim[2]; k++)
        {
          for(int j=0; j<tns_dim[1]; j++)
          {
            for(int i=0; i<tns_dim[0]; i++)
            {
                tnsX(i,j,k,l) = count+1;
                count++;
            }
          }
        }
      }
		}
		else if constexpr (TnsSize == 5)
		{
			for(int m=0; m<tns_dim[4]; m++)
      {
        for(int l=0; l<tns_dim[3]; l++)
        {
          for(int k=0; k<tns_dim[2]; k++)
          {
            for(int j=0; j<tns_dim[1]; j++)
            {
              for(int i=0; i<tns_dim[0]; i++)
              {
                  tnsX(i,j,k,l,m) = count+1;
                  count++;
              }
            }
          }
        }
      }
	}
	#endif // DOXYGEN_SHOULD_SKIP_THIS

	}
} // end namespace partensor

#endif // end of PARTENSOR_DATA_GENERATION_HPP
