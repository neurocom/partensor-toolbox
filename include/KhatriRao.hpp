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
* @file      KhatriRao.hpp
* @details
* Implementations of the Khatri-Rao Product for two or more matrices
* of @c Matrix type, using the @c kroneckerProduct function from 
* @c Eigen. 
********************************************************************/

#ifndef PARTENSOR_KHATRI_RAO_HPP
#define PARTENSOR_KHATRI_RAO_HPP

#include "PARTENSOR_basic.hpp"
#include "unsupported/Eigen/KroneckerProduct"

#if __has_include("tbb/parallel_for.h")
	#include "boost/range/irange.hpp"
	#include "tbb/parallel_for.h"
#endif /* __has_include("tbb/parallel_for.h") */

// #if USE_TBB
// 	#include "boost/range/irange.hpp"
// 	#include "tbb/parallel_for.h"
// #endif /* USE_TBB */

namespace partensor
{	
	inline namespace v1 {
		
		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		
		namespace internal {

			/**
			 * Computes the @c Khatri-Rao Product among 2 or more @c Eigen Matrices,
			 * (uses Sequential Policy) with the use of @c Eigen::kroneckerProduct.
			 * 
			 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
			 * @param  mat1 	[in] A @c Matrix.
			 * @param  mat2 	[in] A @c Matrix.
			 * @param  mats 	[in] Possible 0 or more Matrices of @c Matrix type.
			 * 
			 * @returns The result of the @c Khatri-Rao product, stored in a @c Matrix variable.
			 */
			template <typename ...Matrices>
			Matrix KhatriRao_seq( Matrix   const &mat1, 
								  Matrix   const &mat2, 
								  Matrices const &...mats )
			{
				if constexpr (sizeof... (mats) == 0)
				{
				int R 	= mat1.cols();
				int I1 	= mat1.rows();
				int I2 	= mat2.rows();

				Matrix res(I1*I2,R);

				for (int i=0; i < R; i++)
				{
					res.block(0,i,I1*I2,1) = kroneckerProduct(mat1.block(0,i,I1,1),mat2.block(0,i,I2,1));
				}

				return res;
				}
				else
				{
				auto _temp = KhatriRao_seq(mat2,mats...);

				return KhatriRao_seq(mat1,_temp);
				}
			}
		} // end namespace internal

		#endif // DOXYGEN_SHOULD_SKIP_THIS

		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/**
		 * @brief @c Khatri-Rao implementation with an @c Execution @c Policy applied.
		 * 
		 * Implementation for the @c Khatri-Rao product, among two or more Matrices of
		 * @c Matrix type. Also, an @c Execution @c Policy can be applied, like
		 * @c sequential or @c parallel, with typing @c execution::seq or @c execution::par.
		 * Default @c ExecutionPolicy is @c sequential.
		 * 
		 * @tparam ExecutionPolicy		Type of @c stl @c Execution @c Policy 
		 *                              (sequential, parallel).
		 * @tparam Matrices         	A template parameter pack ( @c stl @c variadic ) type, 
		 *                              with possible multiple Matrices.
		 * 
		 * @param  mat1 	       [in] A @c Matrix.
		 * @param  mat2 	       [in] A @c Matrix.
		 * @param  mats 	       [in] Possible 0 or more Matrices of @c Matrix type.
		 * 
		 * @returns The result of the @c Khatri-Rao product, stored in a @c Matrix variable.
		 */
		template <typename ExecutionPolicy, typename ...Matrices>
		execution::internal::enable_if_execution_policy<ExecutionPolicy,Matrix>
		KhatriRao( ExecutionPolicy       &&, 
				   Matrix          const &mat1, 
				   Matrix          const &mat2, 
				   Matrices        const &...mats )
		{
			return internal::KhatriRao_seq(mat1,mat2,mats...);
		}

		#endif // DOXYGEN_SHOULD_SKIP_THIS
		
		/**
		 * Computes the @c Khatri-Rao Product among 2 or more Matrices,
		 * with the use of @c Eigen::kroneckerProduct.
		 * 
		 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
		 * @param  mat1 	[in] A @c Matrix.
		 * @param  mat2 	[in] A @c Matrix.
		 * @param  mats 	[in] Possible 0 or more Matrices of @c Matrix type.
		 * 
		 * @returns An @c Eigen Matrix is returned of type @c DT. 
		 */
		template <typename ...Matrices>
		Matrix KhatriRao( Matrix   const &mat1, 
						  Matrix   const &mat2, 
						  Matrices const &...mats )
		{
			return KhatriRao(execution::seq,mat1,mat2,mats...);
		}
	
	} // namespace v1
	
	#if __has_include("tbb/parallel_for.h")
	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	namespace experimental {

		inline namespace v1{
			
			namespace internal {					

				/**
				 * Computes the @c Khatri-Rao Product among 2 or more @c Eigen Matrices,
				 * (uses Parallel Policy) with the use of @c Eigen::kroneckerProduct.
				 *
				 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
				 * @param  mat1 	[in] A @c Matrix.
				 * @param  mat2 	[in] A @c Matrix.
				 * @param  mats 	[in] Possible 0 or more Matrices of @c Matrix type.
				 * 
				 * @returns The result of the @c Khatri-Rao product, stored in a @c Matrix variable.
				 */
				template <typename ...Matrices>
				Matrix KhatriRao_par( Matrix   const &mat1, 
									Matrix   const &mat2, 
									Matrices const &...mats )
				{
					if constexpr (sizeof... (mats) == 0)
					{
						int R 	= mat1.cols();
						int I1 	= mat1.rows();
						int I2 	= mat2.rows();

						Matrix res(I1*I2,R);
						// auto      r = boost::irange(0,R);

						tbb::parallel_for(tbb::blocked_range<std::size_t>(0, R),
										[&](const tbb::blocked_range<size_t>& r) {
											for (std::size_t i = r.begin(); i != r.end(); ++i)
												res.block(0,i,I1*I2,1) = kroneckerProduct(mat1.block(0,i,I1,1),mat2.block(0,i,I2,1));
										} );

						return res;
					}
					else
					{
						//int R 	= mat1.cols();
						//int I 	= (mats.rows() * ... * mat2.rows());

						auto _temp = KhatriRao_par(mat2,mats...);

						return KhatriRao_par(mat1,_temp);
					}
				}
			}  // namespace internal
			
			/**
			 * @brief @c Khatri-Rao implementation with an @c Execution @c Policy applied.
			 * 
			 * Implementation for the @c Khatri-Rao product, among two or more Matrices of
			 * @c Matrix type. Also, an @c Execution @c Policy can be applied, like
			 * @c sequential or @c parallel, with typing @c execution::seq or @c execution::par.
			 * Default @c ExecutionPolicy is @c sequential.
			 * 
			 * @tparam ExecutionPolicy		Type of @c stl @c Execution @c Policy 
			 *                              (sequential, parallel).
			 * @tparam Matrices         	A template parameter pack ( @c stl @c variadic ) type, 
			 *                              with possible multiple Matrices.
			 * 
			 * @param  mat1 	       [in] A @c Matrix.
			 * @param  mat2 	       [in] A @c Matrix.
			 * @param  mats 	       [in] Possible 0 or more Matrices of @c Matrix type.
			 * 
			 * @returns The result of the @c Khatri-Rao product, stored in a @c Matrix variable.
			 */
			template <typename ExecutionPolicy, typename ...Matrices>
			execution::internal::enable_if_execution_policy<ExecutionPolicy,Matrix>
			KhatriRao( ExecutionPolicy       &&, 
					Matrix          const &mat1, 
					Matrix          const &mat2, 
					Matrices        const &...mats )
			{
				using  ExPolicy = std::remove_cv_t<std::remove_reference_t<ExecutionPolicy>>;

				if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
				{
				return partensor::v1::internal::KhatriRao_seq(mat1,mat2,mats...);
				}
				else if constexpr (std::is_same_v<ExPolicy,execution::parallel_policy>)
				{
				return internal::KhatriRao_par(mat1,mat2,mats...);
				}
				else
				{
				return partensor::v1::internal::KhatriRao_seq(mat1,mat2,mats...);
				}
			}

		} // end namespace
			
		namespace v2 {

			namespace internal {

				/**
				 * Computes the @c Khatri-Rao Product among 2 or more @c Eigen Matrices,
				 * (uses Sequential Policy) with the use of @c Eigen::kroneckerProduct and 
				 * Eigen::MatrixBase::eval which force immediate evaluation.
				 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
				 * @param  res 	[in,out] An @c Eigen Matrix.
				 * @param  I1 	[in] 	 The number of rows, of the @c res.
				 * @param  mat2 [in] 	 An @c Eigen Matrix.
				 * @param  mats [in] 	 In case of more than two Eigen Matrices.
				 * 
				 * @returns An @c Eigen Matrix in @c res.
				 */
				template <typename ...Matrices>
				int KhatriRao_seq( Matrix         &res, 
				                   int 		       I1, 
								   Matrix   const &mat2, 
								   Matrices const &...mats )
				{
					const int R  = mat2.cols();
					int       I2 = mat2.rows();

					[[maybe_unused]] int I 	= I1 * I2 * (mats.rows()*...*1);

					for (int i=0; i < R; i++)
					{
						res.block(0,i,I1*I2,1) = kroneckerProduct(res.block(0,i,I1,1),mat2.block(0,i,I2,1)).eval();
					}

					I1 *= I2;

					if constexpr (sizeof... (mats) == 0)
					{
						return I1;
					}
					else
					{
						return KhatriRao_seq(res, I1, mats...);
					}
				}

				/**
				 * Interface for the experimental @c Khatri-Rao Product among 2 or more 
				 * @c Eigen Matrices, (uses Sequential Policy) with the use of 
				 * @c Eigen::kroneckerProduct and @c Eigen::MatrixBase::eval, which forces 
				 * immediate evaluation.
				 * 
				 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
				 * @param  mat1 	[in] An @c Eigen Matrix.
				 * @param  mat2 	[in] An @c Eigen Matrix.
				 * @param  mats 	[in] In case of more than two Eigen Matrices.
				 * 
				 * @returns An @c Eigen Matrix is returned of type @c DT. 
				 */
				template <typename ...Matrices>
				Matrix KhatriRao_seq( Matrix   const &mat1, 
									  Matrix   const &mat2, 
									  Matrices const &...mats )
				{
					int R = mat1.cols();
					int I = mat1.rows() * mat2.rows() * (mats.rows()*...*1);

					Matrix res(I,R);

					res.block(0,0,mat1.rows(),mat1.cols()) = mat1;

					int  I_KR = KhatriRao_seq(res,mat1.rows(),mat2,mats...);

					assert(I == I_KR);

					return res;
				}

				/**
				 * \todo
				 * Computes the @c Khatri-Rao Product among 2 or more 
				 * @c Eigen Matrices, (uses Parallel Policy) with the use of 
				 * @c Eigen::kroneckerProduct and @c Eigen::MatrixBase::eval, which forces 
				 * immediate evaluation.
				 * 
				 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
				 * @param  res 	[in,out] An @c Eigen Matrix.
				 * @param  I1 	[in] 	 The number of rows, of the @c res.
				 * @param  mat2 [in] 	 An @c Eigen Matrix.
				 * @param  mats [in] 	 In case of more than two Eigen Matrices.
				 * 
				 * @returns An @c Eigen Matrix in @c res. 
				 */
				template <typename DT = DefaultDataType, typename ...Matrices>
				int KhatriRao_par( Matrix         &res, 
				                   int             I1, 
								   Matrix   const &mat2, 
								   Matrices const &...mats )
				{
					const int R  = mat2.cols();
					int       I2 = mat2.rows();

					// Matrix  res(I1*I2,R);
					// auto      r = boost::irange(0,R);

					tbb::parallel_for(tbb::blocked_range<std::size_t>(0, R),
						[&](const tbb::blocked_range<size_t>& r) {
							Eigen::VectorXd temp = Eigen::VectorXd::Zero(I2,1);
							for (std::size_t j = r.begin(); j != r.end(); j++)
							{
								temp = mat2.col(j);
								for (auto i = 0; i < I1; i++)//std::size_t
								{
								res.block(i*I2, j, I2, 1).noalias() =  res(i,j) * temp;
								}
							}
						} );

					I1 *= I2;

					if constexpr (sizeof... (mats) == 0)
					{
						return I1;
					}
					else
					{
						return KhatriRao_par(res, I1, mats...);
					}
				}

				/**
				 * \todo
				 * Interface for the @c experimental::v1 @c Khatri-Rao Product among 2 or more 
				 * @c Eigen Matrices, (uses Parallel Policy) with the use of 
				 * @c Eigen::kroneckerProduct and @c Eigen::MatrixBase::eval, which forces 
				 * immediate evaluation.
				 * 
				 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
				 * @param  mat1 	[in] An @c Eigen Matrix.
				 * @param  mat2 	[in] An @c Eigen Matrix.
				 * @param  mats 	[in] In case of more than two Eigen Matrices.
				 * 
				 * @returns An @c Eigen Matrix is returned of type @c DT. 
				 */
				template <typename DT = DefaultDataType, typename ...Matrices>
				Matrix KhatriRao_par( Matrix   const &mat1, 
									  Matrix   const &mat2, 
									  Matrices const &...mats )
				{
					int R = mat1.cols();
					int I = mat1.rows() * mat2.rows() * (mats.rows()*...*1);

					Matrix res(I,R);

					res.block(I-mat1.rows(),0,mat1.rows(),mat1.cols()) = mat1;

					int I_KR = KhatriRao_par(res,mat1.rows(),mat2,mats...);

					assert(I == I_KR);

					return res;
				}
			}   // namespace internal

			/**
			 * Interface for the @c Khatri-Rao implementation, that calls the requested  
			 * function, based on the @c ExecutionPolicy to use. If no @c ExecutionPolicy  
			 * is specified then @c sequential is being used.
			 * @tparam ExecutionPolicy		Type of @c stl @c Execution @c Policy (sequential, parallel).
			 * @tparam DT     				Type of @c Eigen Matrix (float, double, etc.). 
			 *                              Default value is @c double.
			 * @tparam Matrices         	A @c variadic type in case of more than 2 matrices.
			 * @param  ExecutionPolicy [in] Execution Policy to execute.
			 * @param  mat1 	       [in] An @c Eigen Matrix.
			 * @param  mat2 	       [in] An @c Eigen Matrix.
			 * @param  mats 	       [in] In case of more than two Eigen Matrices.
			 */
			template <typename ExecutionPolicy, typename ...Matrices>
			execution::internal::enable_if_execution_policy<ExecutionPolicy,Matrix>
			KhatriRao( ExecutionPolicy 		&&, 
			           Matrix         const &mat1, 
					   Matrix         const &mat2, 
					   Matrices       const &...mats )
			{
			  using  ExPolicy = std::remove_cv_t<std::remove_reference_t<ExecutionPolicy>>;

			  if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
			  {
			    return internal::KhatriRao_seq(mat1,mat2,mats...);
			  }
			  else if constexpr (std::is_same_v<ExPolicy,execution::parallel_policy>)
			  {
			    return internal::KhatriRao_par(mat1,mat2,mats...);
			  }
			  else
			    return internal::KhatriRao_seq(mat1,mat2,mats...);
			}

			/**
			 * Computes the @c Khatri-Rao Product among 2 or more @c Eigen Matrices,
			 * (uses Sequential Policy).
			 * 
			 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
			 * @param  mat1 	[in] An @c Eigen Matrix.
			 * @param  mat2 	[in] An @c Eigen Matrix.
			 * @param  mats 	[in] In case of more than two Eigen Matrices.
			 * 
			 * @returns An @c Eigen Matrix is returned of type @c DT. 
			 */
			template <typename ...Matrices>
			Matrix KhatriRao( Matrix const &mat1, 
									  Matrix const &mat2, 
									  Matrices  	 const &...mats )
			{
			  return KhatriRao(execution::seq,mat1,mat2,mats...);
			}
		} // end namespace v2

		namespace v3 {

			namespace internal {

				/**
				 * Computes the @c Khatri-Rao Product among 2 or more 
				 * @c Eigen Matrices, (uses Sequential Policy) with the use of 
				 * @c Eigen::kroneckerProduct and @c Eigen::InnerStride, in order to  
				 * avoid temporary variables.
				 * 
				 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
				 * @param  res 	[in,out] An @c Eigen Matrix.
				 * @param  I1 	[in] 	 The number of rows, of the @c res.
				 * @param  mat2 [in] 	 An @c Eigen Matrix.
				 * @param  mats [in] 	 In case of more than two Eigen Matrices.
				 * 
				 * @returns An @c Eigen Matrix in @c res. 
				 */
				template <typename DT = DefaultDataType, typename ...Matrices>
				int KhatriRao_seq( Matrix         &res, 
								   int             I1, 
								   Matrix   const &mat2, 
								   Matrices const &...mats )
				{
					const int R  = mat2.cols();
					int       I2 = mat2.rows();
					int 	  I  = I1 * I2 * (mats.rows()*...*1);

					for (int i=0; i < R; i++)
					{
						Eigen::Map<Matrix, 0, Eigen::InnerStride<>>(res.data()+i*I, I1*I2, 1, Eigen::InnerStride<>( res.innerStride()*(I/(I1*I2)))) =
						kroneckerProduct( Eigen::Map<Matrix, 0, Eigen::InnerStride<>>(res.data()+i*I, I1, 1, Eigen::InnerStride<>(res.innerStride()*(I/I1))), mat2.block(0,i,I2,1));
					}

					I1 *= I2;

					if constexpr (sizeof... (mats) == 0)
					{
						return I1;
					}
					else
					{
						return KhatriRao_seq(res, I1, mats...);
					}
				}

				/**
				 * Interface for the @c experimental::v2 @c Khatri-Rao Product among 2 or more 
				 * @c Eigen Matrices, (uses Sequential Policy) with the use of 
				 * @c Eigen::kroneckerProduct and @c Eigen::InnerStride, in order to  
				 * avoid temporary variables.
				 * 
				 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
				 * @param  mat1 	[in] An @c Eigen Matrix.
				 * @param  mat2 	[in] An @c Eigen Matrix.
				 * @param  mats 	[in] In case of more than two Eigen Matrices.
				 * 
				 * @returns An @c Eigen Matrix is returned of type @c DT. 
				 */
				template <typename ...Matrices>
				Matrix KhatriRao_seq( Matrix   const &mat1, 
									  Matrix   const &mat2, 
									  Matrices const &...mats)
				{
					int R = mat1.cols();
					int I = mat1.rows() * mat2.rows() * (mats.rows()*...*1);
					
					Matrix res(I,R);

					Eigen::Map<Matrix, 0, Eigen::InnerStride<>> (res.data(), mat1.rows(), R, Eigen::InnerStride<>(res.innerStride()*(I/mat1.rows()))) = mat1;

					int  I_KR = KhatriRao_seq(res,mat1.rows(),mat2,mats...);

					assert(I == I_KR);

					return res;
				}

				/**
				 * \todo
				 * Computes the @c Khatri-Rao Product among 2 or more 
				 * @c Eigen Matrices, (uses Parallel Policy) with the use of 
				 * @c Eigen::kroneckerProduct and @c Eigen::InnerStride, in order to  
				 * avoid temporary variables.
				 * 
				 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
				 * @param  res 	[in,out] An @c Eigen Matrix.
				 * @param  I1 	[in] 	 The number of rows, of the @c res.
				 * @param  mat2 [in] 	 An @c Eigen Matrix.
				 * @param  mats [in] 	 In case of more than two Eigen Matrices.
				 * 
				 * @returns An @c Eigen Matrix in @c res. 
				 */
				template <typename ...Matrices>
				int KhatriRao_par( Matrix         &res, 
				                   int             I1, 
								   Matrix   const &mat2, 
								   Matrices const &...mats )
				{
					const int R  = mat2.cols();
					int       I2 = mat2.rows();

					// Matrix  res(I1*I2,R);
					// auto            r = boost::irange(0,R);

					tbb::parallel_for(tbb::blocked_range<std::size_t>(0, R),
							[&](const tbb::blocked_range<size_t>& r) {
								Eigen::VectorXd temp = Eigen::VectorXd::Zero(I2,1);
								for (std::size_t j = r.begin(); j != r.end(); j++)
								{
									temp = mat2.col(j);
									for (auto i = 0; i < I1; i++)
									{
										res.block(i*I2, j, I2, 1).noalias() =  res(i,j) * temp;
									}
								}
							} );

					I1 *= I2;

					if constexpr (sizeof... (mats) == 0)
					{
						return I1;
					}
					else
					{
						return KhatriRao_par(res, I1, mats...);
					}
				}

				/**
				 * \todo
				 * Interface for the @c experimental::v2 @c Khatri-Rao Product among 2 or more 
				 * @c Eigen Matrices, (uses Parallel Policy) with the use of 
				 * @c Eigen::kroneckerProduct and @c Eigen::InnerStride, in order to  
				 * avoid temporary variables.
				 * 
				 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
				 * @param  mat1 	[in] An @c Eigen Matrix.
				 * @param  mat2 	[in] An @c Eigen Matrix.
				 * @param  mats 	[in] In case of more than two Eigen Matrices.
				 * 
				 * @returns An @c Eigen Matrix is returned of type @c DT. 
				 */
				template <typename ...Matrices>
				Matrix KhatriRao_par( Matrix   const &mat1, 
									  Matrix   const &mat2, 
									  Matrices const &...mats )
				{
					int R = mat1.cols();
					int I = mat1.rows() * mat2.rows() * (mats.rows()*...*1);

					Matrix res(I,R);

					res.block(I-mat1.rows(),0,mat1.rows(),mat1.cols()) = mat1;

					int I_KR = KhatriRao_par(res,mat1.rows(),mat2,mats...);

					assert(I == I_KR);

					return res;
				}
	    }   // namespace internal

			/**
			 * Interface for the @c Khatri-Rao implementation, that calls the requested  
			 * function, based on the @c ExecutionPolicy to use. If no @c ExecutionPolicy  
			 * is specified then @c sequential is being used.
			 * 
			 * @tparam ExecutionPolicy		Type of @c stl @c Execution @c Policy (sequential, parallel).
			 * @tparam Matrices         	A @c variadic type in case of more than 2 matrices.
			 * @param  ExecutionPolicy [in] Execution Policy to execute.
			 * @param  mat1 	       [in] An @c Eigen Matrix.
			 * @param  mat2 	       [in] An @c Eigen Matrix.
			 * @param  mats 	       [in] In case of more than two Eigen Matrices.
			 */
			template <typename ExecutionPolicy, typename ...Matrices>
			execution::internal::enable_if_execution_policy<ExecutionPolicy,Matrix>
			KhatriRao( ExecutionPolicy 		&&, 
					   Matrix         const &mat1, 
					   Matrix         const &mat2, 
					   Matrices       const &...mats)
			{
				using  ExPolicy = std::remove_cv_t<std::remove_reference_t<ExecutionPolicy>>;

				if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
				{
					return internal::KhatriRao_seq(mat1,mat2,mats...);
				}
				else if constexpr (std::is_same_v<ExPolicy,execution::parallel_policy>)
				{
					return internal::KhatriRao_par(mat1,mat2,mats...);
				}
				else
				return internal::KhatriRao_seq(mat1,mat2,mats...);
			}

			/**
			 * Computes the @c Khatri-Rao Product among 2 or more @c Eigen Matrices,
			 * (uses Sequential Policy).
			 * 
			 * @tparam Matrices      A @c variadic type in case of more than 2 matrices.
			 * @param  mat1 	[in] An @c Eigen Matrix.
			 * @param  mat2 	[in] An @c Eigen Matrix.
			 * @param  mats 	[in] In case of more than two Eigen Matrices.
			 * 
			 * @returns An @c Eigen Matrix is returned of type @c DT. 
			 */
			template <typename ...Matrices>
			Matrix KhatriRao( Matrix   const &mat1, 
							  Matrix   const &mat2, 
							  Matrices const &...mats )
			{
				return KhatriRao(execution::seq,mat1,mat2,mats...);
			}
		} // end namespace v3

		// namespace v4 {
		//
		// 	namespace internal {

	      // template <typename DT = DefaultDataType, typename ...Matrices>
	      // int KhatriRao_seq(Matrix &res, int I1, Matrix const &mat1, Matrix const &mat2, Matrices const &...mats)
	      // {
	      //   // const int R = mat2.cols();
				// 	//
	      //   // int I2 	= mat2.rows();
				// 	//
	      //   // int I 	= I1 * I2 * (mats.rows()*...*1);
				// 	//
	      //   // for (int i=0; i < R; i++)
	      //   // {
	      //   //   res.block(0,i,I1*I2,1) = kroneckerProduct(res.block(0,i,I1,1),mat2.block(0,i,I2,1)).eval();
	      //   // }
				// 	//
	      //   // I1 *= I2;
				//
	      //   if constexpr (sizeof... (mats) == 0)
	      //   {
				// 		for (int i=0; i < R; i++)
				// 		{
				// 			res.block(0,i,I1*I2,1) = kroneckerProduct(mat1.block(0,i,I1,1),mat2.block(0,i,I2,1)).eval();
				// 		}
				//
	      //     return I1;
	      //   }
	      //   else
	      //   {
	      //    return KhatriRao_seq(res, I1, mats...);
				//
 	      //   for (int i=0; i < R; i++)
 	      //   {
 	      //     Eigen::Map<MatrixType<dataType>, 0, Eigen::InnerStride<> >  (res.data()+i*I, I1*I2, 1, Eigen::InnerStride<>( res.innerStride()*(I/(I1*I2)))) =
 	      //     kroneckerProduct( Eigen::Map<MatrixType<dataType>, 0, Eigen::InnerStride<> >  (res.data()+i*I, I1, 1, Eigen::InnerStride<>( res.innerStride()*(I/I1))), mat2.block(0,i,I2,1));
 	      //   }
				//
	      //   }
	      // }
				//
	      // template <typename DT = DefaultDataType, typename ...Matrices>
				// Matrix KhatriRao_seq(Matrix const &mat1, Matrix const &mat2, Matrices const &...mats)
				// {
	      //   int    R = mat1.cols();
	      //   int    I = mat1.rows() * mat2.rows() * (mats.rows()*...*1);
	      //   Matrix res(I,R);
				//
				//   res.block(0,0,mat1.rows(),mat1.cols()) = mat1;
				//
	      //   int  I_KR = KhatriRao_seq(res,mat1,mat2,mats...);
				//
	      //   assert(I == I_KR);
				//
	      //   return res;
				// }

	  //     template <typename DT = DefaultDataType, typename ...Matrices>
	  //     int KhatriRao_par(Matrix &res, int I1, Matrix const &mat2, Matrices const &...mats)
		// 		{
	  //       const int R = mat2.cols();
		//
	  //       int I2 	= mat2.rows();
		//
	  //       // Matrix  res(I1*I2,R);
	  //       // auto      r = boost::irange(0,R);
		//
	  //       tbb::parallel_for(tbb::blocked_range<std::size_t>(0, R),
	  //                         [&](const tbb::blocked_range<size_t>& r) {
	  //                             Eigen::VectorXd temp = Eigen::VectorXd::Zero(I2,1);
	  //                             for (std::size_t j = r.begin(); j != r.end(); j++)
	  //                             {
	  //                               temp = mat2.col(j);
	  //                               for (std::size_t i = 0; i < I1; i++)
	  //                               {
	  //                                 res.block(i*I2, j, I2, 1).noalias() =  res(i,j) * temp;
	  //                               }
	  //                             }
	  //                         } );
		//
	  //       I1 *= I2;
		//
	  //       if constexpr (sizeof... (mats) == 0)
		// 		  {
	  //         return I1;
		// 		  }
		// 		  else
		// 		  {
	  //        return KhatriRao_par(res, I1, mats...);
		// 		  }
		// 		}
		//
	  //     template <typename DT = DefaultDataType, typename ...Matrices>
	  //     Matrix KhatriRao_par(Matrix const &mat1, Matrix const &mat2, Matrices const &...mats)
	  //     {
	  //       int    R = mat1.cols();
	  //       int    I = mat1.rows() * mat2.rows() * (mats.rows()*...*1);
	  //       Matrix res(I,R);
		//
	  //       res.block(I-mat1.rows(),0,mat1.rows(),mat1.cols()) = mat1;
		//
	  //       int  I_KR = KhatriRao_par(res,mat1.rows(),mat2,mats...);
		//
	  //       assert(I == I_KR);
		//
	  //       return res;
	  //     }
	  //   }   // namespace internal
		//
		// 	template <typename ExecutionPolicy, typename DT = DefaultDataType, typename ...Matrices>
		// 	execution::internal::enable_if_execution_policy<ExecutionPolicy,Matrix>
		// 	KhatriRao(ExecutionPolicy &&, Matrix const &mat1, Matrix const &mat2, Matrices const &...mats)
		// 	{
		// 	  using  ExPolicy = std::remove_cv_t<std::remove_reference_t<ExecutionPolicy>>;
		//
		// 	  if constexpr (std::is_same_v<ExPolicy,execution::sequenced_policy>)
		// 	  {
		// 	    return internal::KhatriRao_seq(mat1,mat2,mats...);
		// 	  }
		// 	  else if constexpr (std::is_same_v<ExPolicy,execution::parallel_policy>)
		// 	  {
		// 	    return internal::KhatriRao_par(mat1,mat2,mats...);
		// 	  }
		// 	  else
		// 	    return internal::KhatriRao_seq(mat1,mat2,mats...);
		// 	}
		//
		// 	template <typename DT = DefaultDataType, typename ...Matrices>
		// 	Matrix KhatriRao(Matrix const &mat1, Matrix const &mat2, Matrices const &...mats)
		// 	{
		// 	  return KhatriRao(execution::seq,mat1,mat2,mats...);
		// 	}
		// } // end namespace v4

	}  // end namespace experimental
	
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS
	#endif // end of #if __has_include("tbb/parallel_for.h")
} // end namespace partensor

#endif // end of PARTENSOR_KHATRI_RAO_HPP
