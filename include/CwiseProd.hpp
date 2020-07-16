#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      15/04/2019
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
* @file      CwiseProd.hpp
* @details
* Implements the element wise product among two or more Matrices 
* using the @c Eigen function @c cwiseProduct.
********************************************************************/

#ifndef PARTENSOR_CWISE_PROD_HPP
#define PARTENSOR_CWISE_PROD_HPP

#include "PARTENSOR_basic.hpp"

namespace partensor {

  inline namespace v1 {
    /**
     * @brief Element Wise product among Matrices.
     * 
     * Expand implementation of @c Eigen @c cwiseProduct (element wise product)
     * for two or more Matrices.
     * 
     * @tparam Matrices A template parameter pack ( @c stl @c variadic ) type, 
     *                  with possible multiple Matrices.
     * 
     * @param mat1 [in] A @c partensor::Matrix.
     * @param mat2 [in] A @c partensor::Matrix.
     * @param mats [in] Possible 0 or more A @c partensor Matrices.
     * 
     * @returns Returns the result @c Matrix with the element wise product among the
     *          given matrices.
     */
    template<typename... Matrices>
    Matrix CwiseProd( Matrix   const &mat1, 
                      Matrix   const &mat2, 
                      Matrices const &... mats )
    {
      if constexpr (sizeof... (mats) == 0)
      {
        return mat1.cwiseProduct(mat2);
      }
      else
      {
        auto _temp = CwiseProd(mat2, mats...);

        return CwiseProd(mat1, _temp);
      }
    }

  } // end namespace v1
  
} // end namespace partensor

#endif // PARTENSOR_CWISE_PROD_HPP
