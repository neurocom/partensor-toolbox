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
* @file      TerminationConditions.hpp
* @details
* Containts implementations, that check if specific terminations
* conditions are satisfied. 
********************************************************************/

#ifndef PARTENSOR_TERMINATION_CONDITIONS_HPP
#define PARTENSOR_TERMINATION_CONDITIONS_HPP

#include "DataGeneration.hpp"
#include "math.h"

namespace partensor {

    /**
     * Struct containing default values, for the termination conditions.
     */
    struct Conditions {
      int    const max_iter = 500;          // Maximum Number of iterations
      double const ao_tol   = 1e-2;				  // Tolerance for AO Algorithm
      double const delta_1  = 1e-2;					// | Tolerance for Nesterov Algorithm
      double const delta_2  = 1e-2;					// |
    };

    static inline Conditions con; /** < A class @c Conditions object. */

    /**
     * Checks if @c countIter is greater that the @c max_iter of 
     * @c Conditions class.
     * 
     * @param countIter [in] Current iteration.
     * 
     * @returns If @c counterIter does NOT surpass @c max_iter then returns 1,
     *          otherwise returns 0.
     */
    inline int maxIterations(int const countIter)
    {
      return (countIter >= con.max_iter) ? 1 : 0;
    }

    /**
     * Checks if the current relative cost function has smaller value than 
     * the default tolerance @c ao_tol.
     * 
     * @param newValue  [in] Current value.
     * @param trueValue [in] True value.
     * 
     * @returns If relative cost function does NOT surpass @c ao_tol 
     *          then returns 1, otherwise returns 0.
     */
    inline int relativeCostFunction(double const newValue, double const trueValue)
    {
      return (newValue/sqrt(trueValue) <= con.ao_tol) ? 1 : 0;
    }

    /**
     * Checks if the current relative error has smaller value than 
     * the default tolerance @c ao_tol.
     * 
     * @param newValue  [in] Current value.
     * @param trueValue [in] True value.
     * 
     * @returns If relative error does NOT surpass @c ao_tol 
     *          then returns 1, otherwise returns 0.
     */
    inline int relativeError(double const newValue, double const trueValue)
    {
      return (abs(trueValue-newValue)/trueValue <= con.ao_tol) ? 1 : 0;
    }

    /**
     * Checks if the current objective value error has smaller value than 
     * the default tolerance @c ao_tol.
     * 
     * @param newValue  [in] Current value.
     * @param pastValue [in] True value.
     * 
     * @returns If objective value error does NOT surpass @c ao_tol 
     *          then returns 1, otherwise returns 0.
     */
    inline int objectiveValueError(double const newValue, double const pastValue)
    {
      return (abs(newValue-pastValue) <= con.ao_tol) ? 1 : 0;
    }

    /**
     * @brief Computes the "distance" between the current Tensor and the real one.
     * 
     * Calculate the @c squaredNorm() between the real matricized Tensor 
     * @c origMatrTns and the Tensor generated from a factorization algorithm.
     * 
     * @tparam MatrixArray      An array container type.
     * @param origMatrTns  [in] The matricization of the original/true Tensor.
     *                          It must be in @c Eigen Matrix format.
     * @param factors      [in] Contains the factors generated from a factorization
     *                          algorithm, like @c cpd, @c cpdDimTree, etc.
     * @param mode         [in] Specify in which mode, is @c origMatrTns matricized.
     * 
     * @returns The "distance" between @c origMatrTns and the generated Tensor.
     *          If returned value is 0, then the two Tensors are identical.
     */
    template<typename MatrixArray>
    double error( typename MatrixArrayTraits<MatrixArray>::value_type const &origMatrTns, 
                  MatrixArray                                         const &factors, 
                  std::size_t                                         const  mode        )
    {
      using Matrix = typename MatrixArrayTraits<MatrixArray>::value_type;

      Matrix localMatrTns = generateTensor(mode, factors);
      return (origMatrTns - localMatrTns).squaredNorm();
    }

} // end namespace partensor

#endif // PARTENSOR_TERMINATION_CONDITIONS_HPP
