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
* @file      Constants.hpp
* @details
* Implementation for various enumerations.
********************************************************************/
#ifndef PARTENSOR_CONSTANTS_HPP
#define PARTENSOR_CONSTANTS_HPP

namespace partensor {

	/**
	 * Based on the format of the Tensor, different
	 * implementations of the algorithms are used.
	 */
	enum class ProblemType : uint8_t {
	  dense      = 0,
	  sparse     = 1,
	  incomplete = 2
	};

	/**
	 * Based on which Method the factorization will be computed.
	 */
	enum class Method : uint8_t {
		als = 0,  // alternating least squares
		rnd = 1,  // randomized
		bc  = 2   // block coordinate descent
	};

	/**
	 * Possible implementation for each Factor.
	 */
	enum class Constraint : uint8_t {
		unconstrained = 0, /**< unconstrained */
		nonnegativity = 1, /**< nonnegativity */
		orthogonality = 2, /**< orthogonality */
		sparsity      = 3, /**< sparsity */
		constant      = 4  /**< constant */
	};

	/**
	 * In case of not initialized Tensor or Factors,
	 * choose one of the following distribution in order
	 * to produce synthetic data.
	 */
	enum class Distribution : uint8_t {
	  uniform  = 0,
	  gaussian = 1
	};

} // end namespace partensor

#endif // PARTENSOR_CONSTANTS_HPP
