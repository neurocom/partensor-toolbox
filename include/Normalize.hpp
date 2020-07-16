#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      10/09/2019
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
* @file      Normalize.hpp
* @details
* Implementations for the normalization of the factors. The factors
* can be either @c Eigen Matrix or @c FactorDimTree.
********************************************************************/
#ifndef NORMALIZE_HPP
#define NORMALIZE_HPP

#include <math.h>
#include "PARTENSOR_basic.hpp"
#include "DimTrees.hpp"
#include "TensorOperations.hpp"
#include "Eigen/Core" 

namespace partensor
{

  inline namespace v1 {

	/**
   * 
   * Checks if all factors have orthogonal constraint, in order to avoid normalization
	 * after each factor update. After that if a non-orthogonal constraint being applied
	 * on a factor, then the @c the id of this factor is returned, to be used for 
	 * normalization function.
   * 
   * @tparam Status                  Status class based on @c cpd algorithm chosen.
   * @param  st             [in]     Reference to the @c Status class. Used in order
	 *                                 to extract the the factors' @c constraints array. 
   * @param  all_orthogonal [in,out] If all factors have @c orthogonal constraint then
	 *                                 the algorithm will not normalize factors after each
	 *                                 update. Otherwise, uses @c weight_factor factor to 
	 *                                 load the weights of the other factors.
   * @param  weight_factor  [in,out] The first factor that has no orthogonal constraint.
   */
    template<typename Status>
    void choose_normilization_factor(Status const &st, 
                                     bool &all_orthogonal=true, 
                                     int &weight_factor=0)
    {
      for(std::size_t i=0; i<st.options.constraints.size(); ++i)
      {
        if(st.options.constraints[i] != Constraint::orthogonality)
        {
          all_orthogonal = false;
          weight_factor  = i;
          break;
        }
      }
    }

    /**
     * @brief Factors normalization.
     * 
     * Normalizes the columns of all factors based on the last factor
     * from the @c stl array @c factors.
     * 
     * @tparam _TnsSize               Size of the @c factors and @c gramian arrays.
     *
     * @param  weight_factor [in]     The facotr that the weights of the other factors
     *                                will be loaded.
     * @param  R             [in]     Rank of factorization. (Number of columns in
     *                                each @c Matrix).
     * @param  gramian       [in,out] Quantity of @c factor^T @c * @c factor.
     * @param  factors       [in,out] An @c stl array containing all @c Matrix 
     *                                factors.
     *
     *
     * @note This implementation ONLY, if factors are of
     * @c Matrix type.
     *
     * @note NO @c orthogonal constraints must be applied on the 
     *       @c weight_factor factor.
     */
    template<std::size_t _TnsSize>
    void Normalize( int                         const  weight_factor,
                    int                         const  R, 
                    std::array<Matrix,_TnsSize>       &gramian, 
                    std::array<Matrix,_TnsSize>       &factors)
    {
      using Vector = Eigen::VectorXd;

      constexpr std::size_t         lastFactor  = _TnsSize - 1;

      int                           pass_flag   = 0;
      double                        cumul_power = 1;
      bool                          nonZeroFlag = true;

      std::array<Matrix,_TnsSize>   normMatrixList;

      std::array<Vector,lastFactor> lambda_fac;
      std::array<double,lastFactor> norm_factor;

      for(int i=0; i<static_cast<int>(_TnsSize); ++i)
      {
        if(i != weight_factor)
        {
          norm_factor[i-pass_flag] = 1;
          lambda_fac[i-pass_flag]  = Vector(static_cast<int>(R));
          lambda_fac[i-pass_flag]  = gramian[i].diagonal();
          normMatrixList[i]        = Matrix::Zero(static_cast<int>(R),static_cast<int>(R));
        }
        else
        {
          normMatrixList[i] = Matrix::Ones(static_cast<int>(R),static_cast<int>(R)); 
          pass_flag         = 1;
        }
      }

      for(int i=0; i<static_cast<int>(R); ++i)
      {
        for(int j=0; j<static_cast<int>(lastFactor); ++j)
        {
          norm_factor[j] = sqrt((lambda_fac[j])(i));
          nonZeroFlag    = !(norm_factor[j] == 0);
        }
        
        if(!nonZeroFlag)
        {
            for(int j=0; j<static_cast<int>(lastFactor); ++j)
              (lambda_fac[j])(i) = 1;
        }
        else
        {
            pass_flag = 0;
            for(int j=0; j<static_cast<int>(lastFactor); ++j)
            {
                if(j==weight_factor)
                {
                  pass_flag = 1;
                  continue;
                }
                factors[j].col(i)           *= 1/norm_factor[j-pass_flag];
                cumul_power                 *= norm_factor[j-pass_flag];
                (lambda_fac[j-pass_flag])(i) = norm_factor[j-pass_flag];
            }
            factors[weight_factor].col(i)   *= cumul_power;
            cumul_power                      = 1;
        }
      }

      pass_flag = 0;
      for(int i=0; i<static_cast<int>(lastFactor); ++i)
      {
          if(i==weight_factor)
          {
            pass_flag = 1;
            continue;
          }
          normMatrixList[i].noalias()   = lambda_fac[i-pass_flag] * lambda_fac[i-pass_flag].transpose();
          normMatrixList[weight_factor] = normMatrixList[weight_factor].cwiseProduct(normMatrixList[i]);
          gramian[i]                    = gramian[i].cwiseQuotient(normMatrixList[i]);
      }
      
      gramian[weight_factor] = gramian[weight_factor].cwiseProduct(normMatrixList[weight_factor]);

    }

    /**
     * @brief Factors normalization.
     * 
     * Normalizes the columns of all factors based on the last factor
     * from the @c stl array @c factors.
     * 
     * @tparam _TnsSize               Size of the @c factors array.
     * @tparam DimensionType          Array container for @c tnsDims.
     *
     * @param  weight_factor [in]     The facotr that the weights of the other factors
     *                                will be loaded.
     * @param  R             [in]     Rank of factorization. (Number of columns in   
     * 								                each @c FactorDimTree).
     * @param  tnsDims       [in]     The row dimension for each factor.
     * @param  factors       [in,out] An @c stl array containing all factors of 
     *                                type @c FactorDimTree.
     *
     *
     * @note This implementation ONLY, if factors are of
     * @c FactorDimTree type.
     *
     * @note NO @c orthogonal constraints must be applied on the 
     *       @c weight_factor factor.
     */
    template<std::size_t _TnsSize, typename DimensionType>
    void Normalize( int                                const  weight_factor,
                    int                                const  R, 
                    DimensionType                      const &tnsDims, 
                    std::array<FactorDimTree,_TnsSize>       &factors)
    {
      using MatrixArray    = std::array<Matrix,_TnsSize>;
      using FactorIterator = typename std::array<FactorDimTree,_TnsSize>::iterator; 
      using Vector         = Eigen::VectorXd;

      constexpr std::size_t         lastFactor  = _TnsSize - 1;

      int                           pass_flag   = 0;
      double                        cumul_power = 1;
      bool                          nonZeroFlag = true;

      MatrixArray                   factorsList;
      MatrixArray                   gramMatrixList;
      MatrixArray                   normMatrixList;

      std::array<Vector,lastFactor> lambda_fac;
      std::array<double,lastFactor> norm_factor;

      FactorIterator factorPtr = factors.begin();

      for(int i=0; i<static_cast<int>(_TnsSize); ++i)
      {
        factorsList[i]    = Matrix(tnsDims[i],R);
        factorsList[i]    = tensorToMatrix(factorPtr->factor,tnsDims[i],static_cast<int>(R));

        gramMatrixList[i] = Matrix(R,R);
        gramMatrixList[i] = tensorToMatrix(factorPtr->gramian,static_cast<int>(R),static_cast<int>(R));
        
        if(i != weight_factor)
        {
            norm_factor[i-pass_flag] = 1;
            lambda_fac[i-pass_flag]  = Vector(static_cast<int>(R));
            lambda_fac[i-pass_flag]  = gramMatrixList[i].diagonal();
            normMatrixList[i]        = Matrix::Zero(static_cast<int>(R),static_cast<int>(R));
        }
        else
        {
            normMatrixList[i] = Matrix::Ones(static_cast<int>(R),static_cast<int>(R)); 
            pass_flag         = 1;
        }
        factorPtr++;
      }

      for(int i=0; i<static_cast<int>(R); ++i)
      {
        for(int j=0; j<static_cast<int>(lastFactor); ++j)
        {
            norm_factor[j] = sqrt((lambda_fac[j])(i));
            nonZeroFlag    = !(norm_factor[j] == 0);
        }
        
        if(!nonZeroFlag)
        {
            for(int j=0; j<static_cast<int>(lastFactor); ++j)
              (lambda_fac[j])(i) = 1;
        }
        else
        {
            pass_flag = 0;
            for(int j=0; j<static_cast<int>(lastFactor); ++j)
            {
                if(j==weight_factor)
                {
                  pass_flag = 1;
                  continue;
                }
                factorsList[j].col(i)         *= 1/norm_factor[j-pass_flag];
                cumul_power                   *= norm_factor[j-pass_flag];
                (lambda_fac[j-pass_flag])(i)   = norm_factor[j-pass_flag];
            }
            factorsList[weight_factor].col(i) *= cumul_power;
            cumul_power                        = 1;
        }
      }

      pass_flag = 0;
      for(int i=0; i<static_cast<int>(lastFactor); ++i)
      {
        if(i==weight_factor)
        {
          pass_flag = 1;
          continue;
        }
        normMatrixList[i].noalias()   = lambda_fac[i-pass_flag] * lambda_fac[i-pass_flag].transpose();
        normMatrixList[weight_factor] = normMatrixList[weight_factor].cwiseProduct(normMatrixList[i]);
        gramMatrixList[i]             = gramMatrixList[i].cwiseQuotient(normMatrixList[i]);
      }
      
      gramMatrixList[weight_factor] = gramMatrixList[weight_factor].cwiseProduct(normMatrixList[weight_factor]);
      factorPtr                     = factors.begin();

      for(int i=0; i<static_cast<int>(_TnsSize); ++i)
      {
          // Fill with the normalized factors.
          factorPtr->factor  = matrixToTensor(factorsList[i], tnsDims[i], static_cast<int>(R));
          factorPtr->gramian = matrixToTensor(gramMatrixList[i], static_cast<int>(R), static_cast<int>(R));
          factorPtr++;
      }
    }

  } // end namespace v1 

} // end namespace partensor

#endif // end of NORMALIZE_HPP
