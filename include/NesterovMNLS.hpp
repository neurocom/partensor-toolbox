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
* @file 	 NesterovMNLS.hpp
* @details
* Implementation of the Nesterov's  (Accelerated Gradient) algorithm
* with nonnegative constraints. It also provides, supporting functions
* for Nesterov's algorithm.
********************************************************************/

#ifndef PARTENSOR_NESTEROV_MNLS_HPP
#define PARTENSOR_NESTEROV_MNLS_HPP

#include "PARTENSOR_basic.hpp"

namespace partensor
{

	inline namespace v1 {
		/**
		 * @brief @c SVD decomposition of a @c Matrix.
		 * 
		 * Computes the @c svd decomposition of an @c Matrix with 
		 * @c Jacobi's implementation from @c Eigen.
		 * 
		 * @param  mat [in]     The @c Matrix to be decomposed.
		 * @param  L   [in,out] The maximum singular value from the @c svd (returned).
		 * @param  mu  [in,out] The minimum singular value from the @c svd (returned).
		 */
		void ComputeSVD( Matrix const &mat, 
						double       &L, 
						double       &mu )
		{
			Eigen::JacobiSVD<Matrix> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
			L = svd.singularValues().maxCoeff();
			mu = svd.singularValues().minCoeff();
		}

		/**
		 * Computes necessary quantities for @c NesterovMNLS (Nesterov matrix-nonnegative-
		 * least-squares) algorithm.
		 * 
		 * @param mu     [in]	  The minimum singular value from the 
		 *                        @c svd ( computed in @c ComputeSVD ).
		 * @param L      [in,out] The maximum singular value from the 
		 *                        @c svd ( computed in @c ComputeSVD ).
		 * @param lambda [in,out] Normalization parameter.
		 * @param q      [in,out] Inverse of condition number of the problem used 
		 *                        in @c UpdateAlpha.
		 */
		inline void GLambda( double  mu, 
							double &L, 
							double &lambda, 
							double &q     )
		{
			q = mu/L;
			if (1/q>1e6)
				lambda = 10 * mu;
			else if (1/q>1e3)
				lambda = mu;
			else
				lambda = mu/10;

			L += lambda;
			mu += lambda;
			q = mu/L;
		}

		/**
		 * Computes the interpolation quantity for @c NesterovMNLS (Nesterov 
		 * minimum-nonnegative-least-squares) algorithm.
		 * 
		 * @param alpha [in] Starting value for interpolation.
		 * @param q     [in] Inverse of condition number of the problem used in 
		 *                   @c NesterovMNLS.
		 * 
		 * @returns The interpolation quantity.
		 */
		inline double UpdateAlpha( double const alpha,
								double const q    )
		{
			double a, b, c, D;
			a = 1;
			b = alpha*alpha - q;
			c = -alpha*alpha;
			D = b*b - 4*a*c;

			return (-b+sqrt(D))/2;
		}

		/**
		 * Nesterov's algorithm with no-negative constraints and proximal term.
		 * 
		 * Let @c X belongs in @c R with dimensions @c m x @c n, @c A belongs in 
		 * @c R with dimensions @c m x @c r, @c B belongs in @c R with dimensions 
		 * @c n x @c r and consider the minimization problem
		 * @c 0.5*Frobenius_norm(X-AB')^2. We can use the following function in
		 * order to solve this problem.
		 * 
		 * @param  mat1    [in]  	The covariance @c Matrix B'*B.
		 * @param  mat2    [in]  	A @c Matrix containing ther resultt of -(X*B).
		 * @param  delta_1 [in]  	If @c Y is the updated matrix at each iteration, 
		 *                          then @c delta_1 is the maximum tolerance for the 
		 *                          value @c abs(gradient(Y).*Y).
		 * @param  delta_2 [in]     If @c Y is the updated matrix at each iteration, 
		 *                          then @c delta_2 is the minimum tolerance for the
		 *                          value @c gradient(Y).
		 * @param  res     [in,out] The result @c Matrix from Nesterov's algorithm.
		 */
		void NesterovMNLS( Matrix const &mat1, 
						   Matrix const &mat2, 
						   double const  delta_1, 
						   double const  delta_2, 
						   Matrix       &res    )
		{
			const int max_inner = 50;

			int m 	 = res.rows();
			int r 	 = res.cols();
			int iter = 0;
			double L, mu, lambda, q, alpha, new_alpha, beta;

			Matrix A(m, r);
			Matrix Y(m, r);
			Matrix new_A(m, r);
			Matrix grad_Y(m, r);
			Matrix _mat1 = mat1;
			Matrix _mat2 = mat2;
			Matrix Zero_Matrix = Matrix::Zero(m, r);

			ComputeSVD(mat1, L, mu);
			GLambda(mu, L, lambda, q);

			_mat1 += lambda * Matrix::Identity(r, r);
			_mat2 += lambda * res;
			// q 	   = mu/L;
			alpha  = 1;
			A 	   = res;
			Y 	   = res;

			while(1)
			{
				grad_Y 			  = -_mat2; 		// |
				grad_Y.noalias() += Y * _mat1;  	// | grad_Y = W + Y * Z.transpose();

				if ((grad_Y.cwiseProduct(Y).cwiseAbs().maxCoeff() <= delta_1 && grad_Y.minCoeff() >= -delta_2)  || (iter >= max_inner))
						break;

				new_A     = (Y - grad_Y/L).cwiseMax(Zero_Matrix);

				// if ( ( (new_A-Y).norm()/Y.norm()<=delta_1 ) || (iter >= max_inner))
				// 	break;

				new_alpha = UpdateAlpha(alpha, q);
				beta      = alpha * (1 - alpha) / (alpha*alpha + new_alpha);

				Y     = (1 + beta) * new_A - beta * A;
				A     = new_A;
				alpha = new_alpha;
				iter++;
			}
			res  = A;
		}
	
	} // end namespace v1

} // end namespace partensor

#endif // end of PARTENSOR_NESTEROV_MNLS_HPP
