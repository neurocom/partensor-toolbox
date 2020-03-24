#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
 * @date      12/04/2019
 * @author    Christos Tsalidis
 * @author    Yiorgos Lourakis
 * @author    George Lykoudis
 * @copyright 2019 Neurocom All Rights Reserved.
 */
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      Kronecker.hpp
* @details
* Implementations of the Kronecker Product for two or more matrices
* of @c Matrix type, using the @c kroneckerProduct function from 
* @c Eigen. 
* @see @ref Kronecker
*******************************************************************/

#ifndef PARTENSOR_KRONECKER_HPP
#define PARTENSOR_KRONECKER_HPP

#include "PARTENSOR_basic.hpp"
#include "unsupported/Eigen/KroneckerProduct"

namespace partensor {

  inline namespace v1 {

    /**
     * Computes the @c Kronecker Product among 2 or more Matrices, 
     * with the use of @c Eigen::kroneckerProduct.
     * 
     * @tparam Matrices    A @c variadic type in case of more than 2 matrices.
     * @param  mat1 	[in] A @c Matrix.
     * @param  mat2 	[in] A @c Matrix.
     * @param  mats 	[in] Possible 0 or more Matrices of @c Matrix type.
     * 
     * @returns The result of the @c Kronecker product, stored in a @c Matrix variable.
     */
    template <typename ...Matrices>
    Matrix Kronecker( Matrix   const &mat1, 
                      Matrix   const &mat2, 
                      Matrices const &...mats )
    {
      if constexpr (sizeof... (mats) == 0)
      {
        int I1 	= mat1.rows();
        int I2 	= mat1.cols();
        int I3 	= mat2.rows();
        int I4 	= mat2.cols();

        Matrix  res(I1*I3,I2*I4);
        res = Eigen::kroneckerProduct(mat1,mat2);

        return res;
      }
      else
      {
        auto _temp = Kronecker(mat2,mats...);

        return Kronecker(mat1,_temp);
      }
    }
  
  } // namespace v1
  
}// end namespace partensor

#endif // end of PARTENSOR_KRONECKER_HPP
