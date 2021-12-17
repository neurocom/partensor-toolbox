#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      18/04/2019
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
* @file 	 NesterovMNLS.hpp
* @details
* Implementation of the Nesterov's  (Accelerated Gradient) algorithm
* with nonnegative constraints. It also provides, supporting functions
* for Nesterov's algorithm.
********************************************************************/

#ifndef PARTENSOR_NESTEROV_MNLS_HPP
#define PARTENSOR_NESTEROV_MNLS_HPP

#include "PARTENSOR_basic.hpp"
#include "MTTKRP.hpp"

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

		void ComputeEIG(Matrix const &mat, 
		                double       &L, 
						double       &mu)
		{
			Eigen::EigenSolver<Matrix> eig(mat);
			L  = (eig.eigenvalues().real()).maxCoeff();
			mu = (eig.eigenvalues().real()).minCoeff();
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

		void TuneLambda(const double &L, 
		                double       &lambda, 
						const double &ratio)
		{
			// if(ratio < 1)
			// {
			// 	lambda = 0.01;
			// }
			// else
			// {
			// 	lambda = (L * ratio) / (1 - ratio);
			// }
			lambda = L/ratio;
		}

		// Returns maximum eigenvalue L of square matrix mat
		inline double PowerMethod( Matrix &mat, const double epsilon)
		{
			Matrix x_init = Matrix::Random(mat.cols(),1);
			Matrix x_new = x_init;
			Matrix Ax = x_init;
			double norm_Ax;
			double lambda_max = 0;;

			int iter = 0;
			int MAX_ITER = 1e+4;

			while (1)
			{
				Ax.noalias() = mat * x_init;
				norm_Ax = Ax.norm();
				x_new.noalias() = 1/(norm_Ax + 1e-12) * Ax;


				if ((x_new - x_init).norm() <= epsilon || iter >= MAX_ITER)
				{
					lambda_max = (x_new.transpose() * (mat * x_new))(0); 
					break;
				}

				x_init = x_new;
				iter++;	
			}

			return lambda_max;
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

		// GTC Serial
		// transposed_v
		template<std::size_t _TnsSize>
		void NesterovMNLS(Matrix                             &mat1, 
						  std::array<Matrix, _TnsSize>       &factors, 
						  std::array<int, _TnsSize>    const &tns_dimensions, 
						  SparseMatrix                 const &tns_spMat,
                          std::array<int, _TnsSize-1>  const &offsets,
						  int    const                        max_nest_iter, 
						  double const                        ratio, 
						  int    const                        cur_mode, 
						  Constraint const                    constraint_i, 
						  Matrix                             &MTTKRP_T)
		{
			int m = factors[cur_mode].cols();
			int r = factors[cur_mode].rows();
			double L, mu, q, alpha, new_alpha, beta, lambda;
			int iter = 0;

			Matrix grad_Y(r, m);
			Matrix Y(r, m);
			Matrix new_A(r, m);
			Matrix A(r, m);
			Matrix Zero_Matrix = Matrix::Zero(r, m);

			ComputeEIG(mat1, L, mu);

			lambda = ratio;
			L = L + lambda;
			q = lambda / L;
			alpha = 1;

			A = factors[cur_mode];
			Y = factors[cur_mode];

			int last_mode = (cur_mode == static_cast<int>(_TnsSize) - 1) ? static_cast<int>(_TnsSize) - 2 : static_cast<int>(_TnsSize) - 1;

			Matrix temp_R_1(r, 1);
			Matrix temp_col = Matrix::Zero(r, 1);
			while (1)
			{
				grad_Y.setZero();

				if (iter >= max_nest_iter)
				{
					break;
				}

				// Compute grad_Y
				for (long int i = 0; i < tns_spMat.outerSize(); ++i)
				{
					temp_col.setZero();
					for (SparseMatrix::InnerIterator it(tns_spMat, i); it; ++it)
					{
						temp_R_1 = Matrix::Ones(r, 1);
						// Select rows of each factor an compute the respective row of the Khatri-Rao product.
						for (int mode_i = last_mode, kr_counter = static_cast<int>(_TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
						{
							if (mode_i == cur_mode)
							{
								continue;
							}
							long int row;
							row      = ((it.row()) / offsets[kr_counter]) % (tns_dimensions[mode_i]);
							temp_R_1 = temp_R_1.cwiseProduct(factors[mode_i].col(row));
							kr_counter--;
						}
						// Computation of row of Z according the relation (10) of the paper.
						temp_col += (temp_R_1.transpose() * Y.col(i))(0) * temp_R_1;
					}
					grad_Y.col(i) = temp_col;
				}

				// Add proximal term.
				grad_Y += MTTKRP_T + lambda * Y;
				if (constraint_i == Constraint::unconstrained)
				{
					new_A = (Y - grad_Y / L);
				}
				else // Use projection
				{
					new_A = (Y - grad_Y / L).cwiseMax(Zero_Matrix);
				}

				new_alpha = UpdateAlpha(alpha, q);
				beta      = alpha * (1 - alpha) / (alpha * alpha + new_alpha);

				Y = (1 + beta) * new_A - beta * A;

				// Update Y

				A     = new_A;
				alpha = new_alpha;
				iter++;
			}
			factors[cur_mode] = A;
			
		}

		// OpenMP
		template<std::size_t _TnsSize>
		void NesterovMNLS(Matrix                             &mat1, 
						  std::array<Matrix, _TnsSize>       &factors, 
						  std::array<int, _TnsSize>    const &tns_dimensions, 
						  SparseMatrix                 const &tns_spMat,
                          const std::array<int, _TnsSize-1>    &offsets,
						  Matrix                             &Y,
						  int    const                        max_nest_iter, 
						  double const                        ratio, 
						  int    const                        cur_mode, 
						  Constraint const                    constraint_i, 
						  Matrix                             &MTTKRP_T)
		{
			int r = factors[cur_mode].rows();
			double L, mu, q, alpha, new_alpha, beta, lambda;
			int iter = 0;
			long int row;

			Matrix new_A_vec(r, 1);
			Matrix Zero_Vec = Matrix::Zero(r, 1);

			ComputeEIG(mat1, L, mu);

			lambda = ratio;
			L = L + lambda;
			q = lambda / L;
			alpha = 1;

			#pragma omp master
			Y = factors[cur_mode];
			#pragma omp barrier

			int last_mode = (cur_mode == static_cast<int>(_TnsSize) - 1) ? static_cast<int>(_TnsSize) - 2 : static_cast<int>(_TnsSize) - 1;

			Matrix temp_R_1(r, 1);

			#pragma omp barrier
			while (1)
			{
				if (iter >= max_nest_iter)
				{
					break;
				}

				Matrix temp_col = Matrix::Zero(r, 1);
				new_alpha       = UpdateAlpha(alpha, q);
				beta            = alpha * (1 - alpha) / (alpha * alpha + new_alpha);

				// Compute grad_Y
				#pragma omp for schedule(dynamic) //ordered
				for (long int i = 0; i < tns_spMat.outerSize(); ++i)
				{
					temp_col.setZero();
					for (SparseMatrix::InnerIterator it(tns_spMat, i); it; ++it)
					{
						temp_R_1 = Matrix::Ones(r, 1);
						// Select rows of each factor an compute the respective row of the Khatri-Rao product.
						for (int mode_i = last_mode, kr_counter = static_cast<int>(_TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
						{
							if (mode_i == cur_mode)
							{
								continue;
							}
							row      = ((it.row()) / offsets[kr_counter]) % (tns_dimensions[mode_i]);
							temp_R_1 = temp_R_1.cwiseProduct(factors[mode_i].col(row));
							kr_counter--;
						}
						// Computation of row of Z according the relation (10) of the paper.
						temp_col += (temp_R_1.transpose() * Y.col(i))(0) * temp_R_1;
					}

					temp_col.noalias() += MTTKRP_T.col(i) + (lambda * Y.col(i));

					// Add proximal term.
					if (constraint_i == Constraint::unconstrained)
					{
						new_A_vec = (Y.col(i) - temp_col / L);
					}
					else // Use projection
					{
						new_A_vec = (Y.col(i) - temp_col / L).cwiseMax(Zero_Vec);
					}
					
					Y.col(i) = (1 + beta) * new_A_vec - beta * factors[cur_mode].col(i);
					factors[cur_mode].col(i) = new_A_vec;
				}
				alpha = new_alpha;
				iter++;
			}

			#pragma omp barrier
		}

		namespace dynamic_blocksize
		{
			template <std::size_t _TnsSize>
			void StochasticNesterovMNLS(std::array<Matrix, _TnsSize>      &factors,
										std::array<int, _TnsSize>   const &tns_dimensions,
										SparseMatrix                const &tns_spMat,
										std::array<int, _TnsSize-1> const &offsets,
										double                      const  c_stochastic_perc,
										int                         const  max_nest_iter,
										double                      const  lambda,
										int                         const  cur_mode)
			{
				int r = factors[cur_mode].rows();
				double L2, inv_L2;
				double sqrt_q = 0, beta = 0;
				int iter = 0;
				long int row;

				const Matrix zero_vec = Matrix::Zero(r, 1);
				Matrix new_A_vec = Matrix::Zero(r, 1);

				Matrix Y = factors[cur_mode];

				// int first_mode = (cur_mode == 0) ? 1 : 0;
				int last_mode = (cur_mode == static_cast<int>(_TnsSize) - 1) ? static_cast<int>(_TnsSize) - 2 : static_cast<int>(_TnsSize) - 1;

				Matrix temp_R_1 = Matrix::Ones(r, 1);

				Matrix temp_RxR(r, r);
				Matrix temp_col = Matrix::Zero(r, 1);

				std::srand(std::time(nullptr));

				while (1)
				{

					if (iter >= max_nest_iter)
					{
						break;
					}

					// Compute grad_Y
					for (long int i = 0; i < tns_spMat.outerSize(); ++i)
					{
						temp_col.setZero();
						// -- Pseudocode --
						// SparseMatrix::InnerIterator it(tns_spMat, i);
						// int length = tns_spMat.innerNonZeros(); // ???? check if innerNonzeros, we want to count the nonzeros entries in the row
						// int random_pivot = rand(length - blocksize)
						// it += random_pivot;

						// Get the number of nnz per row of matricization.
						long int nnzs_per_col = tns_spMat.innerVector(i).nonZeros();

						long int var_blocksize_i = static_cast<long int>(c_stochastic_perc * nnzs_per_col);

						if (var_blocksize_i > 0)
						{
							// Choose a pivot from [0, nnzs_per_col - blocksize].
							long int pivot = (std::rand() % (nnzs_per_col - var_blocksize_i + 1));

							SparseMatrix::InnerIterator it(tns_spMat, i);
							// Iterate over [pivot, pivot + var_blocksize_i] nnz elements per row.
							// it += pivot;
							for (long int acuum = 0; acuum < pivot; acuum++)
								++it; 

							// std::cout << pivot << "\t" << nnzs_per_col << std::endl;
							temp_RxR.setZero();

							for (long int sample = 0; sample < var_blocksize_i; sample++, ++it)
							{
								temp_R_1 = Matrix::Ones(r, 1);
								// Select rows of each factor an compute the respective row of the Khatri-Rao product.
								for (int mode_i = last_mode, kr_counter = static_cast<int>(_TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
								{
									if (mode_i == cur_mode)
									{
										continue;
									}
									row = ((it.row()) / offsets[kr_counter]) % (tns_dimensions[mode_i]);
									// temp_R_1 = temp_R_1.cwiseProduct(factors[mode_i].row(row));
									temp_R_1 = temp_R_1.cwiseProduct(factors[mode_i].col(row));
									kr_counter--;
								}
								// Computation of row of Z according the relation (10) of the paper.
								// Subtract each term of MTTKRP's row.
								temp_col.noalias() += ((temp_R_1.transpose() * Y.col(i))(0) - it.value()) * temp_R_1;

								// Estimate Hessian for each row.
								temp_RxR.noalias() += (temp_R_1 * temp_R_1.transpose());
							}

							// Solve with l2-regularization term
							// temp_col.noalias() += (lambda * Y.col(i));
							// Solve with Proximal term
							temp_col.noalias() += lambda * (Y.col(i) + factors[cur_mode].col(i));
							
							L2 = PowerMethod(temp_RxR, 1e-3);
							L2 += lambda;
							inv_L2 = 1 / L2;

							new_A_vec.noalias() = (Y.col(i) - inv_L2 * temp_col);
							new_A_vec           = (new_A_vec).cwiseMax(zero_vec);

							sqrt_q = sqrt(lambda * inv_L2);

							beta = (1 - sqrt_q) / (1 + sqrt_q);

							// Update Y
							Y.col(i) = (1 + beta) * new_A_vec - beta * factors[cur_mode].col(i);
							// Update i-th column of current factor
							factors[cur_mode].col(i) = new_A_vec;
						}
					}
					// alpha = new_alpha;
					iter++;
				}
			}

			
			namespace local_L
			{
				// OpenMP
				template<std::size_t _TnsSize>
				void StochasticNesterovMNLS(std::array<Matrix, _TnsSize>       &factors, 
											std::array<int, _TnsSize>    const &tns_dimensions, 
											SparseMatrix                 const &tns_spMat,
											const std::array<int, _TnsSize-1>  &offsets,
											double                       const  c_stochastic_perc,
											Matrix                             &Y,
											int    const                        max_nest_iter, 
											double const                        lambda, 
											int    const                        cur_mode)
				{
					int r = factors[cur_mode].rows();
					double sqrt_q = 0, beta = 0, L2 = 0, inv_L2 = 0;
					int iter = 0;
					long int row;

					Matrix new_A_vec = Matrix::Zero(r, 1);
					Matrix zero_vec  = Matrix::Zero(r, 1);

					#pragma omp master
					Y = factors[cur_mode];

					int last_mode = (cur_mode == static_cast<int>(_TnsSize) - 1) ? static_cast<int>(_TnsSize) - 2 : static_cast<int>(_TnsSize) - 1;

					Matrix temp_R_1 = Matrix::Ones(r, 1);
					Matrix temp_RxR(r, r);
					Matrix temp_col(r, 1);
					
					std::srand(std::time(nullptr));

					#pragma omp barrier
					while (1)
					{
						if (iter >= max_nest_iter)
						{
							break;
						}

						// Compute grad_Y						
						// #pragma omp for schedule(dynamic, 8) ordered 
						// #pragma omp for schedule(guided) nowait
						#pragma omp for schedule(dynamic) nowait
						for (long int i = 0; i < tns_spMat.outerSize(); ++i)
						{
							// Get the number of nnz per row of matricization.
							long int nnzs_per_col = tns_spMat.innerVector(i).nonZeros();

							long int var_blocksize_i = static_cast<long int>(c_stochastic_perc * nnzs_per_col);

							if (var_blocksize_i > 0)
							{
								temp_col.setZero();

								// Choose a pivot from [0, nnzs_per_col - blocksize].
								long int pivot = (std::rand() % (nnzs_per_col - var_blocksize_i + 1));

								SparseMatrix::InnerIterator it(tns_spMat, i);
								// Iterate over [pivot, pivot + var_blocksize_i] nnz elements per row.
								// it += pivot;
								for (long int acuum = 0; acuum < pivot; acuum++)
									++it;

								temp_RxR.setZero();

								for (long int sample = 0; sample < var_blocksize_i; sample++, ++it)
								{
									temp_R_1 = Matrix::Ones(r, 1);

									// Select rows of each factor an compute the respective row of the Khatri-Rao product.
									for (int mode_i = last_mode, kr_counter = static_cast<int>(_TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
									{
										if (mode_i == cur_mode)
										{
											continue;
										}
										row      = ((it.row()) / offsets[kr_counter]) % (tns_dimensions[mode_i]);
										temp_R_1 = temp_R_1.cwiseProduct(factors[mode_i].col(row));
										kr_counter--;
									}
									// Computation of row of Z according the relation (10) of the paper.
									temp_col.noalias() += ((temp_R_1.transpose() * Y.col(i))(0) - it.value()) * temp_R_1;
									
									// Estimate Hessian for each row.
									temp_RxR.noalias() += (temp_R_1 * temp_R_1.transpose());
								}

								temp_col.noalias() += lambda * (Y.col(i) + factors[cur_mode].col(i));

								L2 = PowerMethod(temp_RxR, 1e-3);
								L2 += lambda;
								inv_L2 = 1 / L2;
								
								new_A_vec.noalias() = (Y.col(i) - inv_L2 * temp_col);
								new_A_vec           = (new_A_vec).cwiseMax(zero_vec);
								
								sqrt_q = sqrt( lambda * inv_L2 );

								beta = (1 - sqrt_q) / (1 + sqrt_q);

								// Update Y
								Y.col(i) = (1 + beta) * new_A_vec - beta * factors[cur_mode].col(i);
								
								// Update i-th column of current factor
								factors[cur_mode].col(i) = new_A_vec;

							}
						}
						iter++;
					}
					#pragma omp barrier
				}
				
			}// end namespace local_L

		}// end namespace dynamic_blocksize
	
	} // end namespace v1

	namespace std_V
	{
		// GTC Serial
		template<std::size_t _TnsSize>
		void NesterovMNLS(Matrix                             &mat1, 
						  std::array<Matrix, _TnsSize>       &factors, 
						  std::array<int, _TnsSize>    const &tns_dimensions, 
						  SparseMatrix                 const &tns_spMat,
                          const std::array<int, _TnsSize-1>    &offsets,
						  int        const                    max_nest_iter, 
						  double     const                    ratio, 
						  int        const                    cur_mode, 
						  Constraint const                    constraint_i, 
						  Matrix     const                   &MTTKRP)
		{
			int m = factors[cur_mode].rows();
			int r = factors[cur_mode].cols();
			double L, mu, q, alpha, new_alpha, beta, lambda;
			int iter = 0;

			Matrix grad_Y(m, r);
			Matrix Y(m, r);
			Matrix new_A(m, r);
			Matrix A(m, r);
			Matrix Zero_Matrix = Matrix::Zero(m, r);

			ComputeEIG(mat1, L, mu);

			lambda = ratio; // TuneLambda(L, lambda, ratio);
			L = L + lambda;
			q = lambda / L;
			alpha = 1;

			A = factors[cur_mode];
			Y = factors[cur_mode]; // layer_factor

			int last_mode = (cur_mode == static_cast<int>(_TnsSize) - 1) ? static_cast<int>(_TnsSize) - 2 : static_cast<int>(_TnsSize) - 1;

			Matrix temp_1_R(1, r);
			while (1)
			{
				grad_Y.setZero();

				if (iter >= max_nest_iter)
				{
					break;
				}

				// Compute grad_Y
				Matrix temp_row = Matrix::Zero(1, r);
				for (long int i = 0; i < tns_spMat.outerSize(); ++i)
				{
					temp_row.setZero();
					for (SparseMatrix::InnerIterator it(tns_spMat, i); it; ++it)
					{
						temp_1_R = Matrix::Ones(1, r);
						// Select rows of each factor an compute the respective row of the Khatri-Rao product.
						for (int mode_i = last_mode, kr_counter = static_cast<int>(_TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
						{
							if (mode_i == cur_mode)
							{
								continue;
							}
							int row;
							row      = ((it.row()) / offsets[kr_counter]) % (tns_dimensions[mode_i]);
							temp_1_R = temp_1_R.cwiseProduct(factors[mode_i].row(row));
							kr_counter--;
						}
						// Computation of row of Z according the relation (10) of the paper.
						temp_row += (Y.row(i) * temp_1_R.transpose()) * temp_1_R;
					}
					grad_Y.row(i) = temp_row;
				}

				// Add proximal term.
				grad_Y += MTTKRP + lambda * Y;
				if (constraint_i == Constraint::unconstrained)
				{
					new_A = (Y - grad_Y / L);
				}
				else // Use projection
				{
					new_A = (Y - grad_Y / L).cwiseMax(Zero_Matrix);
				}

				new_alpha = UpdateAlpha(alpha, q);
				beta      = alpha * (1 - alpha) / (alpha * alpha + new_alpha);

				Y = (1 + beta) * new_A - beta * A;

				// Update Y

				A     = new_A;
				alpha = new_alpha;
				iter++;
			}
			factors[cur_mode] = A;
		}
	} // end of namespace std_V

	
	namespace local_L
	{
		// GTC Serial
		// transposed_v
		template<std::size_t _TnsSize>
		void NesterovMNLS(std::array<Matrix, _TnsSize>       &factors, 
						  std::array<int, _TnsSize>    const &tns_dimensions, 
						  SparseMatrix                 const &tns_spMat,
                          const std::array<int, _TnsSize-1>    &offsets,
						  int    const                        max_nest_iter, 
						  double const                        lambda, 
						  int    const                        cur_mode,
						  Matrix                             &MTTKRP_T)
		{
			int m = factors[cur_mode].cols();
			int r = factors[cur_mode].rows();

			Matrix inv_L2(tns_spMat.outerSize(),1);

			double sqrt_q = 0, beta = 0;
			double L2;
			int iter = 0;
			Matrix Y(r, m);
			Matrix A(r, m);

			const Matrix zero_vec    = Matrix::Zero(r, 1);
			Matrix       new_A_vec   = Matrix::Zero(r, 1);

			A = factors[cur_mode];
			Y = factors[cur_mode];

			int last_mode = (cur_mode == static_cast<int>(_TnsSize) - 1) ? static_cast<int>(_TnsSize) - 2 : static_cast<int>(_TnsSize) - 1;

			Matrix temp_R_1 = Matrix::Ones(r, 1);
			Matrix temp_RxR(r, r);
			
			while (1)
			{
				if (iter >= max_nest_iter)
				{
					break;
				}

				// Compute grad_Y
				Matrix temp_col = Matrix::Zero(r, 1);
				for (long int i = 0; i < tns_spMat.outerSize(); ++i)
				{
					temp_col.setZero();
					
					if (iter < 1)
					{
						temp_RxR.setZero();
					}

					for (SparseMatrix::InnerIterator it(tns_spMat, i); it; ++it)
					{
						temp_R_1 = Matrix::Ones(r, 1);
						// Select rows of each factor an compute the respective row of the Khatri-Rao product.
						for (int mode_i = last_mode, kr_counter = static_cast<int>(_TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
						{
							if (mode_i == cur_mode)
							{
								continue;
							}
							long long int row  = ((it.row()) / offsets[kr_counter]) % (tns_dimensions[mode_i]);
							temp_R_1           = temp_R_1.cwiseProduct(factors[mode_i].col(row));
							kr_counter--;
						}
						// Computation of row of Z according the relation (10) of the paper.
						temp_col.noalias() += (temp_R_1.transpose() * Y.col(i))(0) * temp_R_1;
						// Estimate Hessian for each row.
						if (iter < 1)
						{
							temp_RxR.noalias() += (temp_R_1 * temp_R_1.transpose());
						}
					}
					temp_col.noalias() += MTTKRP_T.col(i) + (lambda * Y.col(i));

					if (iter < 1)
					{
						L2 = PowerMethod(temp_RxR, 1e-3);
						L2 += lambda;
						inv_L2(i) = 1 / L2;
					}

					new_A_vec = (Y.col(i) - inv_L2(i) * temp_col).cwiseMax(zero_vec);

					sqrt_q = sqrt( lambda * inv_L2(i) );

					beta = (1 - sqrt_q) / (1 + sqrt_q);

					// Update Y
					Y.col(i) = (1 + beta) * new_A_vec - beta * factors[cur_mode].col(i);

					// Update i-th column of current factor
					factors[cur_mode].col(i) = new_A_vec;
				}
				iter++;

			}

		}

		

		// OpenMP
		template<std::size_t _TnsSize>
		void NesterovMNLS(Matrix                             &inv_L2, 
						  std::array<Matrix, _TnsSize>       &factors, 
						  std::array<int, _TnsSize>    const &tns_dimensions, 
						  SparseMatrix                 const &tns_spMat,
                          const std::array<int, _TnsSize-1>    &offsets,
						  Matrix                             &Y,
						  int    const                        max_nest_iter, 
						  double const                        lambda, 
						  int    const                        cur_mode,
						  Matrix                             &MTTKRP_T)
		{
			int r = factors[cur_mode].rows();
			double sqrt_q = 0, beta = 0;
			double L2;
			int iter = 0;
			long int row;

			Matrix new_A_vec = Matrix::Zero(r, 1);
			Matrix zero_vec  = Matrix::Zero(r, 1);

			#pragma omp master
			Y = factors[cur_mode];
			#pragma omp barrier

			int last_mode = (cur_mode == static_cast<int>(_TnsSize) - 1) ? static_cast<int>(_TnsSize) - 2 : static_cast<int>(_TnsSize) - 1;

			Matrix temp_R_1 = Matrix::Ones(r, 1);
			Matrix temp_RxR(r, r);
			Matrix temp_col(r, 1);

			#pragma omp barrier
			while (1)
			{
				if (iter >= max_nest_iter)
				{
					break;
				}

				// Compute grad_Y
				#pragma omp for schedule(dynamic) nowait
				for (long int i = 0; i < tns_spMat.outerSize(); ++i)
				{
					temp_col.setZero();
					if (iter < 1)
					{
						temp_RxR.setZero();
					}
					for (SparseMatrix::InnerIterator it(tns_spMat, i); it; ++it)
					{
						temp_R_1 = Matrix::Ones(r, 1);
						// Select rows of each factor an compute the respective row of the Khatri-Rao product.
						for (int mode_i = last_mode, kr_counter = static_cast<int>(_TnsSize) - 2; mode_i >= 0 && kr_counter >= 0; mode_i--)
						{
							if (mode_i == cur_mode)
							{
								continue;
							}
							row      = ((it.row()) / offsets[kr_counter]) % (tns_dimensions[mode_i]);
							temp_R_1 = temp_R_1.cwiseProduct(factors[mode_i].col(row));
							kr_counter--;
						}
						// Computation of row of Z according the relation (10) of the paper.
						temp_col.noalias() += (temp_R_1.transpose() * Y.col(i))(0) * temp_R_1;
						
						// Estimate Hessian for each row.
						// Compute only once!
						if (iter < 1)
						{
							temp_RxR.noalias() += (temp_R_1 * temp_R_1.transpose());
						}
					}

					temp_col.noalias() += MTTKRP_T.col(i) + (lambda * Y.col(i));

					if (iter < 1)
					{
						L2 = PowerMethod(temp_RxR, 1e-3);
						L2 += lambda;
						inv_L2(i) = 1 / L2;
					}
					
					new_A_vec = (Y.col(i) - inv_L2(i) * temp_col).cwiseMax(zero_vec);

					sqrt_q = sqrt( lambda * inv_L2(i) );

					beta = (1 - sqrt_q) / (1 + sqrt_q);

					// Update Y
					Y.col(i) = (1 + beta) * new_A_vec - beta * factors[cur_mode].col(i);
					
					// Update i-th column of current factor
					factors[cur_mode].col(i) = new_A_vec;

				}
				iter++;
			}
			#pragma omp barrier
		}

	} // end of namespace local_L

} // end namespace partensor

#endif // end of PARTENSOR_NESTEROV_MNLS_HPP
