#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
 * @date      18/04/2019
 * @author    Christos Tsalidis
 * @author    Yiorgos Lourakis
 * @author    George Lykoudis
 * @copyright 2019 Neurocom All Rights Reserved.
 */
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file 	 PartialKhatriRao.hpp
* @details
* Implementation for partial computation of Khatri-Rao product.
* Makes use of @c KhatriRao function from @c KhatriRao.hpp.
*
* @warning The implementation supports only this operation, when
*          the number of Matrices of @c Matrix type are in range of 
           @c [3-8].
********************************************************************/

#ifndef PARTENSOR_PARTIAL_KHATRI_RAO_HPP
#define PARTENSOR_PARTIAL_KHATRI_RAO_HPP

#include "KhatriRao.hpp"

namespace partensor
{
	inline namespace v1 {
		/**
		 * @brief Partial @c KhatriRao implementation.
		 * 
		 * Computes the Khatri-Rao product of matrices of @c Matrix type
		 * contained in an array container, excluding the one specified in 
		 * @c mode.
		 * 
		 * An @c execution_policy can be applied as in @c KhatriRao function.
		 * Either @c execution::seq (sequential) or @c execution::par (parallel).
		 * The default value is sequential.
		 * 
		 * @tparam Policy           Type of @c Execution @c Policy (sequential, parallel).
		 * @tparam MatrixArray      An array container type.
		 * @param  execution_policy [in] Which @c Policy to execute.
		 * @param  matArray         [in] An array containing matrices to use 
		 *                               in this operation.
		 * @param  mode             [in] Id of matrix to exclude from the Khatri-Rao product.
		 * 
		 * @returns An @c Eigen Matrix with the result.
		 * 
		 * @note The function is supported only, when the size of @c matArray is in range
		 *       of @c [3-8].
		 */
		template <typename Policy, typename MatrixArray_>
		auto PartialKhatriRao( Policy       const &execution_policy, 
							   MatrixArray_ const &matArray, 
							   std::size_t  const  mode            )
		{
			using Matrix = typename MatrixArrayTraits<MatrixArray_>::value_type;   // Type of data of @c Eigen Matrix (e.g double, float, ..).

			constexpr std::size_t TnsSize = MatrixArrayTraits<MatrixArray_>::Size; // Size of matArray.

			if constexpr ( TnsSize == 3 )
			{
				switch ( mode )
				{
					case 0:
						return KhatriRao(execution_policy, matArray[2], matArray[1]);
						break;
					case 1:
						return KhatriRao(execution_policy, matArray[2], matArray[0]);
						break;
					case 2:
						return KhatriRao(execution_policy, matArray[1], matArray[0]);
						break;
					default:
						Matrix resMat;
						return resMat.setZero();
						break;
				}
			}
			else if constexpr ( TnsSize == 4 )
			{
				switch ( mode )
				{
					case 0:
						return KhatriRao(execution_policy, matArray[3], matArray[2], matArray[1]);
						break;
					case 1:
						return KhatriRao(execution_policy, matArray[3], matArray[2], matArray[0]);
						break;
					case 2:
						return KhatriRao(execution_policy, matArray[3], matArray[1], matArray[0]);
						break;
					case 3:
						return KhatriRao(execution_policy, matArray[2], matArray[1], matArray[0]);
						break;
					default:
						Matrix resMat;
						return resMat.setZero();
						break;
				}
			}
			else if constexpr ( TnsSize == 5 )
			{
				switch ( mode )
				{
					case 0:
						return KhatriRao(execution_policy, matArray[4], matArray[3], matArray[2], matArray[1]);
						break;
					case 1:
						return KhatriRao(execution_policy, matArray[4], matArray[3], matArray[2], matArray[0]);
						break;
					case 2:
						return KhatriRao(execution_policy, matArray[4], matArray[3], matArray[1], matArray[0]);
						break;
					case 3:
						return KhatriRao(execution_policy, matArray[4], matArray[2], matArray[1], matArray[0]);
						break;
					case 4:
						return KhatriRao(execution_policy, matArray[3], matArray[2], matArray[1], matArray[0]);
						break;
					default:
						Matrix resMat;
						return resMat.setZero();
						break;
				}
			}
			else if constexpr ( TnsSize == 6 )
			{
				switch ( mode )
				{
					case 0:
						return KhatriRao(execution_policy, matArray[5], matArray[4], matArray[3], matArray[2], matArray[1]);
						break;
					case 1:
						return KhatriRao(execution_policy, matArray[5], matArray[4], matArray[3], matArray[2], matArray[0]);
						break;
					case 2:
						return KhatriRao(execution_policy, matArray[5], matArray[4], matArray[3], matArray[1], matArray[0]);
						break;
					case 3:
						return KhatriRao(execution_policy, matArray[5], matArray[4], matArray[2], matArray[1], matArray[0]);
						break;
					case 4:
						return KhatriRao(execution_policy, matArray[5], matArray[3], matArray[2], matArray[1], matArray[0]);
						break;
					case 5:
						return KhatriRao(execution_policy, matArray[4], matArray[3], matArray[2], matArray[1], matArray[0]);
						break;
					default:
						Matrix resMat;
						return resMat.setZero();
						break;
				}
			}
			else if constexpr ( TnsSize == 7 )
			{
				switch ( mode )
				{
					case 0:
						return KhatriRao(execution_policy, matArray[6], matArray[5], matArray[4], matArray[3], matArray[2], matArray[1]);
						break;
					case 1:
						return KhatriRao(execution_policy, matArray[6], matArray[5], matArray[4], matArray[3], matArray[2], matArray[0]);
						break;
					case 2:
						return KhatriRao(execution_policy, matArray[6], matArray[5], matArray[4], matArray[3], matArray[1], matArray[0]);
						break;
					case 3:
						return KhatriRao(execution_policy, matArray[6], matArray[5], matArray[4], matArray[2], matArray[1], matArray[0]);
						break;
					case 4:
						return KhatriRao(execution_policy, matArray[6], matArray[5], matArray[3], matArray[2], matArray[1], matArray[0]);
						break;
					case 5:
						return KhatriRao(execution_policy, matArray[6], matArray[4], matArray[3], matArray[2], matArray[1], matArray[0]);
						break;
					case 6:
						return KhatriRao(execution_policy, matArray[5], matArray[4], matArray[3], matArray[2], matArray[1], matArray[0]);
						break;
					default:
						Matrix resMat;
						return resMat.setZero();
						break;
				}
			}
			else if constexpr ( TnsSize == 8 )
			{
				switch ( mode )
				{
					case 0:
						return KhatriRao(execution_policy, matArray[7], matArray[6], matArray[5], matArray[4], matArray[3], matArray[2], matArray[1]);
						break;
					case 1:
						return KhatriRao(execution_policy, matArray[7], matArray[6], matArray[5], matArray[4], matArray[3], matArray[2], matArray[0]);
						break;
					case 2:
						return KhatriRao(execution_policy, matArray[7], matArray[6], matArray[5], matArray[4], matArray[3], matArray[1], matArray[0]);
						break;
					case 3:
						return KhatriRao(execution_policy, matArray[7], matArray[6], matArray[5], matArray[4], matArray[2], matArray[1], matArray[0]);
						break;
					case 4:
						return KhatriRao(execution_policy, matArray[7], matArray[6], matArray[5], matArray[3], matArray[2], matArray[1], matArray[0]);
						break;
					case 5:
						return KhatriRao(execution_policy, matArray[7], matArray[6], matArray[4], matArray[3], matArray[2], matArray[1], matArray[0]);
						break;
					case 6:
						return KhatriRao(execution_policy, matArray[7], matArray[5], matArray[4], matArray[3], matArray[2], matArray[1], matArray[0]);
						break;
					case 7:
						return KhatriRao(execution_policy, matArray[6], matArray[5], matArray[4], matArray[3], matArray[2], matArray[1], matArray[0]);
						break;
					default:
						Matrix resMat;
						return resMat.setZero();
						break;
				}
			}
		}

		template <typename MatrixArray_>
		auto PartialKhatriRao(MatrixArray_ const &matArray, std::size_t const mode)
		{
			return PartialKhatriRao(execution::seq, matArray, mode);
		}

	} // end namespace v1

} // end namespace partensor

#endif // end of PARTENSOR_PARTIAL_KHATRI_RAO_HPP
