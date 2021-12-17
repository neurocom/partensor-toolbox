#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      08/02/2021
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
* @file      CUDAMTTKRP.hpp
* @details
* Implements the Matricized Tensor times Khatri Rao Product.
********************************************************************/

#ifndef PARTENSOR_CUDA_TENSOR_KHATRIRAO_PRODUCT_HPP
#define PARTENSOR_CUDA_TENSOR_KHATRIRAO_PRODUCT_HPP

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "add_vectors.hpp"
// #include "omp.h"
#include "PARTENSOR_basic.hpp"

#ifndef MAX_NUM_STREAMS
	#define MAX_NUM_STREAMS 1
#endif

namespace partensor {

	inline namespace MallocManaged 
  	{
		inline namespace nonTransposed 
  		{
			/** 
			 * Computes Matricized Tensor Times Khatri-Rao Product with the use
			 * of both @c omp and @c CUDA (CUDA streams).
			 * 
			 * @tparam _TnsSize                 Tensor Order. 
			 * @tparam Dimensions               Array type containing the Tensor dimensions.
			 *  
			 * @param  tnsDims         [in]     @c Stl array containing the Tensor dimensions, whose
			 *                                  length must be same as the Tensor order.
			 * @param  factors         [in]     An @c stl array with the factors.
			 * @param  tns_mat         [in]     Matricization of the Tensor based on @c mode.
			 * @param  mode            [in]     The dimension where the tensor was matricized 
			 *                                  and the MTTKRP will be computed.
			 * @param  num_threads     [in]     The number of available threads, defined by the environmental 
			 *                                  variable @c OMP_NUM_THREADS.
			 * @param  CUDA_tns_mat    [in]     @c Stl array containing number of MAX_NUM_STREAMS chunks of 
			 * 								    the matricized Tensor based on @c mode.
			 * @param  CUDA_PartialKRP [in]		@c Stl array containing number of MAX_NUM_STREAMS chunks of 
			 * 								    the partial KR product based on @c mode.
			 * @param  CUDA_MTTKRP     [in]		@c Stl array containing the resulted MTTKRP based on @c mode.
			 * @param  handle          [in]     CUDA handler.
			 * @param  Stream          [in]     CUDA stream.
			 * @param  result          [in/out] The result matrix of the multiplication of the 
			 *                              matricized tensor and the Khatri-Rao product.
			 */
			template <std::size_t _TnsSize, typename Dimensions>
			void hybrid_batched_mttkrp(Dimensions const &tnsDims,
									std::array<Matrix, _TnsSize> const &factors,
									Matrix const &tns_mat,
									std::size_t const mode,
									int const num_threads,
									std::array<double *, MAX_NUM_STREAMS> CUDA_tns_mat,
									std::array<double *, MAX_NUM_STREAMS> CUDA_PartialKRP,
									std::array<double *, MAX_NUM_STREAMS> CUDA_MTTKRP,
									cublasHandle_t handle,
									std::array<cudaStream_t, MAX_NUM_STREAMS> Stream,
									Matrix &result)
			{
				// #ifndef EIGEN_DONT_PARALLELIZE
				// 	Eigen::setNbThreads(1);
				// #endif

				const int   rank       = factors[0].cols();
				std::size_t last_mode  = (mode == _TnsSize-1)  ? (_TnsSize - 2) : (_TnsSize - 1);
				std::size_t first_mode = (mode == 0) ? 1 : 0; 

				result = Matrix::Zero(tnsDims[mode], rank);
				Matrix MTTKRP_CPU = Matrix::Zero(tnsDims[mode], rank);

				double alpha = 1.0;
				double beta = 0.0;

				// dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
				long int dim = 1;
				for(std::size_t i=0; i<_TnsSize; ++i)
				{
					if(i!=mode)
						dim *= tnsDims[i];
				}
				// I_(first_mode+1) x I_(first_mode+2) x ... x I_(last_mode), where <I_(first_mode)> #rows of the starting factor.
				int num_of_blocks              = static_cast<int>(dim / static_cast<long int>(tnsDims[first_mode]));
				std::array<int, _TnsSize-2> rows_offset;

				for (int i = static_cast<int>(_TnsSize - 3), j = last_mode; i >= 0; i--, j--)
				{
					if (j == static_cast<int>(mode))
					{
						j--;
					}
					if (i == static_cast<int>(_TnsSize - 3))
					{
						rows_offset[i] = num_of_blocks / tnsDims[j];
					}
					else
					{
						rows_offset[i] = rows_offset[i + 1] / tnsDims[j];
					}
				}

				omp_set_nested(0);
				#pragma omp parallel default(shared)  \
						proc_bind(close)              \
						reduction(sum : MTTKRP_CPU)   \
						num_threads(num_threads)
				{
					int t_id = omp_get_thread_num();
					int str_id = 0, gpu_block_idx = 0;

					#pragma omp for schedule(dynamic, MAX_NUM_STREAMS)
					for (std::size_t block_idx = 0; block_idx < num_of_blocks; block_idx++)
					{
						// Compute Kr = KhatriRao(A_(last_mode)(l,:), A_(last_mode-1)(k,:), ..., A_(2)(j,:))
						// Initialize vector Kr as Kr = A_(last_mode)(l,:)
						Matrix Kr(1, rank);
						Kr = factors[last_mode].row((block_idx / rows_offset[_TnsSize - 3]) % tnsDims[last_mode]);
						Matrix PartialKR(tnsDims[first_mode], rank);
						// compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(j)(...,:)
						for (int i = static_cast<int>(_TnsSize - 4), j = last_mode - 1; i >= 0; i--, j--)
						{
							if (j == static_cast<int>(mode))
							{
								j--;
							}
							Kr = (factors[j].row((block_idx / rows_offset[i]) % tnsDims[j])).cwiseProduct(Kr);
						}

						// Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(first_mode), as : KhatriRao(Kr, A_(first_mode)(:,:))
						for (int row = 0; row < tnsDims[first_mode]; row++)
						{
							PartialKR.row(row) = ((factors[first_mode].row(row)).cwiseProduct(Kr));
						}

						if (t_id == 0)
						{
							str_id = gpu_block_idx % MAX_NUM_STREAMS;
							cudaStreamAttachMemAsync(Stream[str_id], CUDA_tns_mat[str_id]);
							cudaStreamAttachMemAsync(Stream[str_id], CUDA_PartialKRP[str_id]);
							cudaStreamSynchronize(Stream[str_id]);

							Eigen::Map<Matrix>(CUDA_tns_mat[str_id], tnsDims[mode], tnsDims[first_mode]) = tns_mat.block(0, block_idx * tnsDims[first_mode], tnsDims[mode], tnsDims[first_mode]);

							Eigen::Map<Matrix>(CUDA_PartialKRP[str_id], tnsDims[first_mode], rank) = PartialKR;

							beta = static_cast<double>(gpu_block_idx > MAX_NUM_STREAMS - 1);

							cublasSetStream(handle, Stream[str_id]);

							if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tnsDims[mode], rank, tnsDims[first_mode],
											&alpha, CUDA_tns_mat[str_id], tnsDims[mode], CUDA_PartialKRP[str_id], tnsDims[first_mode],
											&beta, CUDA_MTTKRP[str_id], tnsDims[mode]) != CUBLAS_STATUS_SUCCESS)

							{
								std::cerr << "cublasDgemm failed" << std::endl;
							}
							gpu_block_idx++;
						}
						else
						{
							MTTKRP_CPU.noalias() +=  tns_mat.block(0, block_idx * tnsDims[first_mode], tnsDims[mode], tnsDims[first_mode]) * PartialKR;
						}
					}
				}

				cudaDeviceSynchronize();
				int length = tnsDims[mode] * rank;

				for (int str_id = 1; str_id < MAX_NUM_STREAMS; str_id++)
				{
					cuda_vecAdd(CUDA_MTTKRP[0], CUDA_MTTKRP[str_id], CUDA_MTTKRP[0], length);
					cudaDeviceSynchronize();
				}

				result = Eigen::Map<Matrix>(CUDA_MTTKRP[0], tnsDims[mode], rank);
				result += MTTKRP_CPU; // Sum up result.

				// #ifndef EIGEN_DONT_PARALLELIZE
				// 	Eigen::setNbThreads(num_threads);
				// #endif
			}

			/** 
			 * Computes Matricized Tensor Times Khatri-Rao Product with the use
			 * of both @c omp and @c CUDA.
			 * 
			 * @tparam _TnsSize                 Tensor Order. 
			 * @tparam Dimensions               Array type containing the Tensor dimensions.
			 *  
			 * @param  tnsDims         [in]     @c Stl array containing the Tensor dimensions, whose
			 *                                  length must be same as the Tensor order.
			 * @param  factors         [in]     An @c stl array with the factors.
			 * @param  tns_mat         [in]     Matricization of the Tensor based on @c mode.
			 * @param  mode            [in]     The dimension where the tensor was matricized 
			 *                                  and the MTTKRP will be computed.
			 * @param  num_threads     [in]     The number of available threads, defined by the environmental 
			 *                                  variable @c OMP_NUM_THREADS.
			 * @param  CUDA_tns_mat    [in]     Matrix containing a chunk of the matricized Tensor based on @c mode.
			 * @param  CUDA_PartialKRP [in]		Matrix containing a chunk of the partial KR product based on @c mode.
			 * @param  CUDA_MTTKRP     [in]		Matrix containing the resulted MTTKRP based on @c mode.
			 * @param  handle          [in]     CUDA handler.
			 * @param  result          [in/out] The result matrix of the multiplication of the 
			 *                              matricized tensor and the Khatri-Rao product.
			 */
			template <std::size_t _TnsSize, typename Dimensions>
			void hybrid_mttkrp(Dimensions const &tnsDims,
							   std::array<Matrix, _TnsSize> const &factors,
							   Matrix const &tns_mat,
							   std::size_t const mode,
							   int const num_threads,
							   double *CUDA_tns_mat,
							   double *CUDA_PartialKRP,
							   double *CUDA_MTTKRP,
							   cublasHandle_t handle,
							   Matrix &result)
			{
				// #ifndef EIGEN_DONT_PARALLELIZE
				// 	Eigen::setNbThreads(1);
				// #endif

				const int   rank       = factors[0].cols();
				std::size_t last_mode  = (mode == _TnsSize-1)  ? (_TnsSize - 2) : (_TnsSize - 1);
				std::size_t first_mode = (mode == 0) ? 1 : 0; 

				result = Matrix::Zero(tnsDims[mode], rank);
				Matrix MTTKRP_CPU = Matrix::Zero(tnsDims[mode], rank);

				double alpha = 1.0;
				double beta = 0.0;

				// dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
				long int dim = 1;
				for(std::size_t i=0; i<_TnsSize; ++i)
				{
					if(i!=mode)
						dim *= tnsDims[i];
				}
				// I_(first_mode+1) x I_(first_mode+2) x ... x I_(last_mode), where <I_(first_mode)> #rows of the starting factor.
				int num_of_blocks              = static_cast<int>(dim / static_cast<long int>(tnsDims[first_mode]));
				std::array<int, _TnsSize-2> rows_offset;

				for (int i = static_cast<int>(_TnsSize - 3), j = last_mode; i >= 0; i--, j--)
				{
					if (j == static_cast<int>(mode))
					{
						j--;
					}
					if (i == static_cast<int>(_TnsSize - 3))
					{
						rows_offset[i] = num_of_blocks / tnsDims[j];
					}
					else
					{
						rows_offset[i] = rows_offset[i + 1] / tnsDims[j];
					}
				}

				omp_set_nested(0);
				#pragma omp parallel default(shared)  \
						proc_bind(close)              \
						reduction(sum : MTTKRP_CPU)   \
						num_threads(num_threads)
				{
					int t_id = omp_get_thread_num();
					#pragma omp for schedule(static, 1)
					for (std::size_t block_idx = 0; block_idx < num_of_blocks; block_idx++)
					{
						// Compute Kr = KhatriRao(A_(last_mode)(l,:), A_(last_mode-1)(k,:), ..., A_(2)(j,:))
						// Initialize vector Kr as Kr = A_(last_mode)(l,:)
						Matrix Kr(1, rank);
						Kr = factors[last_mode].row((block_idx / rows_offset[_TnsSize - 3]) % tnsDims[last_mode]);
						Matrix PartialKR(tnsDims[first_mode], rank);
						// compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(j)(...,:)
						for (int i = static_cast<int>(_TnsSize - 4), j = last_mode - 1; i >= 0; i--, j--)
						{
							if (j == static_cast<int>(mode))
							{
								j--;
							}
							Kr = (factors[j].row((block_idx / rows_offset[i]) % tnsDims[j])).cwiseProduct(Kr);
						}

						// Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(first_mode), as : KhatriRao(Kr, A_(first_mode)(:,:))
						for (int row = 0; row < tnsDims[first_mode]; row++)
						{
							PartialKR.row(row) = ((factors[first_mode].row(row)).cwiseProduct(Kr));
						}

						if (t_id == 0)
						{
							Eigen::Map<Matrix>(CUDA_tns_mat, tnsDims[mode], tnsDims[first_mode]) = tns_mat.block(0, block_idx * tnsDims[first_mode], tnsDims[mode], tnsDims[first_mode]);

							Eigen::Map<Matrix>(CUDA_PartialKRP, tnsDims[first_mode], rank) = PartialKR;

							beta = static_cast<double>(block_idx > 0);

							if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tnsDims[mode], rank, tnsDims[first_mode],
											&alpha, CUDA_tns_mat, tnsDims[mode], CUDA_PartialKRP, tnsDims[first_mode],
											&beta, CUDA_MTTKRP, tnsDims[mode]) != CUBLAS_STATUS_SUCCESS)

							{
								std::cerr << "cublasDgemm failed" << std::endl;
							}
							cudaDeviceSynchronize();
						}
						else
						{
							MTTKRP_CPU.noalias() += tns_mat.block(0, block_idx * tnsDims[first_mode], tnsDims[mode], tnsDims[first_mode]) * PartialKR;
						}
					}
				}

				result = Eigen::Map<Matrix>(CUDA_MTTKRP, tnsDims[mode], rank);
				result += MTTKRP_CPU; // Sum up result.

				// #ifndef EIGEN_DONT_PARALLELIZE
				// 	Eigen::setNbThreads(num_threads);
				// #endif
			}
		} // namespace nonTransposed
		
		namespace transposed 
  		{
			/** 
			 * Computes Matricized Tensor Times Khatri-Rao Product with the use
			 * of both @c omp and @c CUDA (CUDA streams).
			 * 
			 * @tparam _TnsSize                 Tensor Order. 
			 * @tparam Dimensions               Array type containing the Tensor dimensions.
			 *  
			 * @param  tnsDims         [in]     @c Stl array containing the Tensor dimensions, whose
			 *                                  length must be same as the Tensor order.
			 * @param  factors         [in]     An @c stl array with the factors.
			 * @param  tns_mat         [in]     Matricization of the Tensor based on @c mode.
			 * @param  mode            [in]     The dimension where the tensor was matricized 
			 *                                  and the MTTKRP will be computed.
			 * @param  num_threads     [in]     The number of available threads, defined by the environmental 
			 *                                  variable @c OMP_NUM_THREADS.
			 * @param  CUDA_tns_mat    [in]     @c Stl array containing number of MAX_NUM_STREAMS chunks of 
			 * 									the matricized Tensor based on @c mode.
			 * @param  CUDA_PartialKRP [in]		@c Stl array containing number of MAX_NUM_STREAMS chunks of 
			 * 									the partial KR product based on @c mode.
			 * @param  CUDA_MTTKRP     [in]		@c Stl array containing the resulted MTTKRP based on @c mode.
			 * @param  handle          [in]     CUDA handler.
			 * @param  Stream          [in]     CUDA stream.
			 * @param  result          [in/out] The result matrix of the multiplication of the 
			 *                              matricized tensor and the Khatri-Rao product.
			 */
			template <std::size_t _TnsSize, typename Dimensions>
			void hybrid_batched_mttkrp(Dimensions const &tnsDims,
									std::array<Matrix, _TnsSize> const &factors,
									Matrix const &tns_mat,
									std::size_t const mode,
									int const num_threads,
									std::array<double *, MAX_NUM_STREAMS> CUDA_tns_mat,
									std::array<double *, MAX_NUM_STREAMS> CUDA_PartialKRP,
									std::array<double *, MAX_NUM_STREAMS> CUDA_MTTKRP,
									cublasHandle_t handle,
									std::array<cudaStream_t, MAX_NUM_STREAMS> Stream,
									Matrix &result)
			{
				// #ifndef EIGEN_DONT_PARALLELIZE
				// 	Eigen::setNbThreads(1);
				// #endif

				const int   rank       = factors[0].rows();
				std::size_t last_mode  = (mode == _TnsSize-1)  ? (_TnsSize - 2) : (_TnsSize - 1);
				std::size_t first_mode = (mode == 0) ? 1 : 0;

				result = Matrix::Zero(tnsDims[mode], rank);
				Matrix MTTKRP_CPU = Matrix::Zero(tnsDims[mode], rank);

				double alpha = 1.0;
				double beta = 0.0;

				// dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
				long int dim = 1;
				for(std::size_t i=0; i<_TnsSize; ++i)
				{
					if(i!=mode)
						dim *= tnsDims[i];
				}
				// I_(first_mode+1) x I_(first_mode+2) x ... x I_(last_mode), where <I_(first_mode)> #rows of the starting factor.
				int num_of_blocks              = static_cast<int>(dim / static_cast<long int>(tnsDims[first_mode]));
				std::array<int, _TnsSize-2> rows_offset;

				for (int i = static_cast<int>(_TnsSize - 3), j = last_mode; i >= 0; i--, j--)
				{
					if (j == static_cast<int>(mode))
					{
						j--;
					}
					if (i == static_cast<int>(_TnsSize - 3))
					{
						rows_offset[i] = num_of_blocks / tnsDims[j];
					}
					else
					{
						rows_offset[i] = rows_offset[i + 1] / tnsDims[j];
					}
				}

				omp_set_nested(0);
				#pragma omp parallel default(shared)  \
						proc_bind(close)              \
						reduction(sum : MTTKRP_CPU)   \
						num_threads(num_threads)
				{
					int t_id = omp_get_thread_num();
					int str_id = 0, gpu_block_idx = 0;

					#pragma omp for schedule(dynamic, MAX_NUM_STREAMS)
					for (std::size_t block_idx = 0; block_idx < num_of_blocks; block_idx++)
					{
						// Compute Kr = KhatriRao(A_(last_mode)(l,:), A_(last_mode-1)(k,:), ..., A_(2)(j,:))
						// Initialize vector Kr as Kr = A_(last_mode)(l,:)
						Matrix Kr(rank, 1);
						Kr = factors[last_mode].col((block_idx / rows_offset[_TnsSize - 3]) % tnsDims[last_mode]);
						Matrix PartialKR(rank, tnsDims[first_mode]);
						// compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(j)(...,:)
						for (int i = static_cast<int>(_TnsSize - 4), j = last_mode - 1; i >= 0; i--, j--)
						{
							if (j == static_cast<int>(mode))
							{
								j--;
							}
							Kr = (factors[j].col((block_idx / rows_offset[i]) % tnsDims[j])).cwiseProduct(Kr);
						}

						// Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(first_mode), as : KhatriRao(Kr, A_(first_mode)(:,:))
						for (int col = 0; col < tnsDims[first_mode]; col++)
						{
							PartialKR.col(col) = ((factors[first_mode].col(col)).cwiseProduct(Kr));
						}

						if (t_id == 0)
						{
							str_id = gpu_block_idx % MAX_NUM_STREAMS;
							cudaStreamAttachMemAsync(Stream[str_id], CUDA_tns_mat[str_id]);
							cudaStreamAttachMemAsync(Stream[str_id], CUDA_PartialKRP[str_id]);
							cudaStreamSynchronize(Stream[str_id]);

							Eigen::Map<Matrix>(CUDA_tns_mat[str_id], tnsDims[mode], tnsDims[first_mode]) = tns_mat.block(0, block_idx * tnsDims[first_mode], tnsDims[mode], tnsDims[first_mode]);

							Eigen::Map<Matrix>(CUDA_PartialKRP[str_id], rank, tnsDims[first_mode]) = PartialKR;

							beta = static_cast<double>(gpu_block_idx > MAX_NUM_STREAMS - 1);

							cublasSetStream(handle, Stream[str_id]);

							if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, tnsDims[mode], rank, tnsDims[first_mode],
											&alpha, CUDA_tns_mat[str_id], tnsDims[mode], CUDA_PartialKRP[str_id], rank,
											&beta, CUDA_MTTKRP[str_id], tnsDims[mode]) != CUBLAS_STATUS_SUCCESS)

							{
								std::cerr << "cublasDgemm failed" << std::endl;
							}
							gpu_block_idx++;
						}
						else
						{
							MTTKRP_CPU.noalias() += tns_mat.block(0, block_idx * tnsDims[first_mode], tnsDims[mode], tnsDims[first_mode]) * PartialKR.transpose();
						}
					}
				}

				cudaDeviceSynchronize();
				int length = tnsDims[mode] * rank;

				for (int str_id = 1; str_id < MAX_NUM_STREAMS; str_id++)
				{
					cuda_vecAdd(CUDA_MTTKRP[0], CUDA_MTTKRP[str_id], CUDA_MTTKRP[0], length);
					cudaDeviceSynchronize();
				}

				result = Eigen::Map<Matrix>(CUDA_MTTKRP[0], tnsDims[mode], rank);
				result += MTTKRP_CPU; // Sum up result.

				// #ifndef EIGEN_DONT_PARALLELIZE
				// 	Eigen::setNbThreads(num_threads);
				// #endif
			}

			/** 
			 * Computes Matricized Tensor Times Khatri-Rao Product with the use
			 * of both @c omp and @c CUDA.
			 * 
			 * @tparam _TnsSize                 Tensor Order. 
			 * @tparam Dimensions               Array type containing the Tensor dimensions.
			 *  
			 * @param  tnsDims         [in]     @c Stl array containing the Tensor dimensions, whose
			 *                                  length must be same as the Tensor order.
			 * @param  factors         [in]     An @c stl array with the factors.
			 * @param  tns_mat         [in]     Matricization of the Tensor based on @c mode.
			 * @param  mode            [in]     The dimension where the tensor was matricized 
			 *                                  and the MTTKRP will be computed.
			 * @param  num_threads     [in]     The number of available threads, defined by the environmental 
			 *                                  variable @c OMP_NUM_THREADS.
			 * @param  CUDA_tns_mat    [in]     Matrix containing a chunk of the matricized Tensor based on @c mode.
			 * @param  CUDA_PartialKRP [in]		Matrix containing a chunk of the partial KR product based on @c mode.
			 * @param  CUDA_MTTKRP     [in]		Matrix containing the resulted MTTKRP based on @c mode.
			 * @param  handle          [in]     CUDA handler.
			 * @param  result          [in/out] The result matrix of the multiplication of the 
			 *                              matricized tensor and the Khatri-Rao product.
			 */
			template <std::size_t _TnsSize, typename Dimensions>
			void hybrid_mttkrp(Dimensions const &tnsDims,
							   std::array<Matrix, _TnsSize> const &factors,
							   Matrix const &tns_mat,
							   std::size_t const mode,
							   int const num_threads,
							   double *CUDA_tns_mat,
							   double *CUDA_PartialKRP,
							   double *CUDA_MTTKRP,
							   cublasHandle_t handle,
							   Matrix &result)
			{
				// #ifndef EIGEN_DONT_PARALLELIZE
				// 	Eigen::setNbThreads(1);
				// #endif

				const int   rank       = factors[0].cols();
				std::size_t last_mode  = (mode == _TnsSize-1)  ? (_TnsSize - 2) : (_TnsSize - 1);
				std::size_t first_mode = (mode == 0) ? 1 : 0; 

				result = Matrix::Zero(tnsDims[mode], rank);
				Matrix MTTKRP_CPU = Matrix::Zero(tnsDims[mode], rank);

				double alpha = 1.0;
				double beta = 0.0;

				// dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
				long int dim = 1;
				for(std::size_t i=0; i<_TnsSize; ++i)
				{
					if(i!=mode)
						dim *= tnsDims[i];
				}
				// I_(first_mode+1) x I_(first_mode+2) x ... x I_(last_mode), where <I_(first_mode)> #rows of the starting factor.
				int num_of_blocks              = static_cast<int>(dim / static_cast<long int>(tnsDims[first_mode]));
				std::array<int, _TnsSize-2> rows_offset;

				for (int i = static_cast<int>(_TnsSize - 3), j = last_mode; i >= 0; i--, j--)
				{
					if (j == static_cast<int>(mode))
					{
						j--;
					}
					if (i == static_cast<int>(_TnsSize - 3))
					{
						rows_offset[i] = num_of_blocks / tnsDims[j];
					}
					else
					{
						rows_offset[i] = rows_offset[i + 1] / tnsDims[j];
					}
				}

				omp_set_nested(0);
				#pragma omp parallel default(shared)  \
						proc_bind(close)              \
						reduction(sum : MTTKRP_CPU)   \
						num_threads(num_threads)
				{
					int t_id = omp_get_thread_num();
					#pragma omp for schedule(dynamic, 1)
					for (std::size_t block_idx = 0; block_idx < num_of_blocks; block_idx++)
					{
						// Compute Kr = KhatriRao(A_(last_mode)(l,:), A_(last_mode-1)(k,:), ..., A_(2)(j,:))
						// Initialize vector Kr as Kr = A_(last_mode)(l,:)
						Matrix Kr(rank, 1);
						Kr = factors[last_mode].col((block_idx / rows_offset[_TnsSize - 3]) % tnsDims[last_mode]);
						Matrix PartialKR(rank, tnsDims[first_mode]);
						// compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(j)(...,:)
						for (int i = static_cast<int>(_TnsSize - 4), j = last_mode - 1; i >= 0; i--, j--)
						{
							if (j == static_cast<int>(mode))
							{
								j--;
							}
							Kr = (factors[j].col((block_idx / rows_offset[i]) % tnsDims[j])).cwiseProduct(Kr);
						}

						// Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(first_mode), as : KhatriRao(Kr, A_(first_mode)(:,:))
						for (int col = 0; col < tnsDims[first_mode]; col++)
						{
							PartialKR.col(col) = ((factors[first_mode].col(col)).cwiseProduct(Kr));
						}

						if (t_id == 0)
						{
							Eigen::Map<Matrix>(CUDA_tns_mat, tnsDims[mode], tnsDims[first_mode]) = tns_mat.block(0, block_idx * tnsDims[first_mode], tnsDims[mode], tnsDims[first_mode]);

							Eigen::Map<Matrix>(CUDA_PartialKRP, tnsDims[first_mode], rank) = PartialKR;

							beta = static_cast<double>(block_idx > 0);

							if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, tnsDims[mode], rank, tnsDims[first_mode],
											&alpha, CUDA_tns_mat, tnsDims[mode], CUDA_PartialKRP, tnsDims[first_mode],
											&beta, CUDA_MTTKRP, tnsDims[mode]) != CUBLAS_STATUS_SUCCESS)

							{
								std::cerr << "cublasDgemm failed" << std::endl;
							}
							cudaDeviceSynchronize();
						}
						else
						{
							MTTKRP_CPU.noalias() += tns_mat.block(0, block_idx * tnsDims[first_mode], tnsDims[mode], tnsDims[first_mode]) * PartialKR.transpose();
						}
					}
				}

				result = Eigen::Map<Matrix>(CUDA_MTTKRP, tnsDims[mode], rank);
				result += MTTKRP_CPU; // Sum up result.

				// #ifndef EIGEN_DONT_PARALLELIZE
				// 	Eigen::setNbThreads(num_threads);
				// #endif
			}			
		} // namespace transposed		

	} // namespace MallocManaged

} // end namespace partensor

#endif // PARTENSOR_CUDA_TENSOR_KHATRIRAO_PRODUCT_HPP
