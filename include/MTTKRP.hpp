#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
 * @date      15/04/2019
 * @author    Christos Tsalidis
 * @author    Yiorgos Lourakis
 * @author    George Lykoudis
 * @copyright 2019 Neurocom All Rights Reserved.
 */
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      MTTKRP.hpp
* @details
* Implements the Matricized Tensor times Khatri Rao Product.
********************************************************************/

#ifndef PARTENSOR_TENSOR_KHATRIRAO_PRODUCT_HPP
#define PARTENSOR_TENSOR_KHATRIRAO_PRODUCT_HPP

#include "PARTENSOR_basic.hpp"
#include "PartialKhatriRao.hpp"

// #define EIGEN_DONT_PARALLELIZE
// #define NUM_SOCKETS                1
#define STATIC_SCHEDULE_CHUNK_SIZE 1
/* -- Declare custom reduction of variable_type : Matrix -- */
#pragma omp declare reduction(sum                           \
                              : Eigen::MatrixXd           \
                              : omp_out = omp_out + omp_in) \
    initializer(omp_priv = Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))

namespace partensor {

  inline namespace v1 {

    /**
     * Computes the "Partial" Khatri-Rao product and the Matricized Tensor 
     * Khatri-Rao Product (MTTKRP). More specifically, computes the Khatri-Rao
     * product for all @c factors apart from factors[@c mode] and use it to 
     * finally calculate the MTTKRP.
     * 
     * @tparam _TnsSize         Size of the @c factors array.
     * 
     * @param  factors [in]     An @c stl array with the factors.
     * @param  tns_mat [in]     Matricization of the Tensor based on @c mode.
     * @param  mode    [in]     The dimension where the tensor was matricized 
     *                          and the MTTKRP will be computed.
     * @param  krao    [in,out] The result Khatri-Rao product for the 
     *                          @c factors excluding the @c mode factor.
     * @param  result  [in/out] The result matrix of the multiplication of the 
     *                          matricized tensor and the Khatri-Rao product.
     */
    template<std::size_t _TnsSize>
    void mttkrp(std::array<Matrix,_TnsSize> const &factors,
                Matrix                      const &tns_mat,
                int                         const &mode,
                Matrix                            &krao,
                Matrix                            &result)
    {
      krao             = PartialKhatriRao(factors, mode);
      result.noalias() = tns_mat * krao;
    }

    /*
     * Get number of threads, defined by the environmental variable $(OMP_NUM_THREADS). 
     */
    inline int get_num_threads()
    {
      int threads;
      #pragma omp parallel
      {
        threads = omp_get_num_threads();
      }
      return threads;
    }

    /** 
     * Computes Matricized Tensor Times Khatri-Rao Product with the use
     * of @c OpenMP.
     * 
     * @tparam _TnsSize             Tensor Order. 
     * @tparam Dimensions           Array type containing the Tensor dimensions.
     *  
     * @param  tnsDims     [in]     @c Stl array containing the Tensor dimensions, whose
     *                              length must be same as the Tensor order.
     * @param  factors     [in]     An @c stl array with the factors.
     * @param  tns_mat     [in]     Matricization of the Tensor based on @c mode.
     * @param  mode        [in]     The dimension where the tensor was matricized 
     *                              and the MTTKRP will be computed.
     * @param  num_threads [in]     The number of available threads, defined by the environmental 
     *                              variable @c OMP_NUM_THREADS.
     * @param  result      [in/out] The result matrix of the multiplication of the 
     *                              matricized tensor and the Khatri-Rao product.
     */
    template <std::size_t _TnsSize, typename Dimensions>
    void mttkrp(Dimensions                  const &tnsDims, 
                std::array<Matrix,_TnsSize> const &factors,
                Matrix                      const &tns_mat,
                std::size_t                 const  mode, 
                int                         const  num_threads,
                Matrix                            &result)
    {
      #ifndef EIGEN_DONT_PARALLELIZE
          Eigen::setNbThreads(1);
      #endif

      constexpr std::size_t NUM_SOCKETS = 1;
      std::size_t           inner_num_threads = num_threads / NUM_SOCKETS;
      if (inner_num_threads < 1) 
        inner_num_threads = 1;

      const int   rank       = factors[0].cols();
      std::size_t last_mode  = (mode == _TnsSize-1)  ? (_TnsSize - 2) : (_TnsSize - 1);
      std::size_t first_mode = (mode == 0) ? 1 : 0; 
      
      result = Matrix::Zero(tnsDims[mode], rank);

      // dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
      long int dim = 1;
      for(std::size_t i=0; i<_TnsSize; ++i)
      {
        if(i!=mode)
          dim *= tnsDims[i];
      }
      // I_(first_mode+1) x I_(first_mode+2) x ... x I_(last_mode), where <I_(first_mode)> #rows of the starting factor.
      int num_of_blocks              = static_cast<int>(dim / static_cast<long int>(tnsDims[first_mode]));
      int numOfBlocks_div_NumSockets = num_of_blocks / NUM_SOCKETS;
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
      
      // --- If Factors are Transposed ---
      // omp_set_nested(1);
      // #pragma omp parallel for num_threads(NUM_SOCKETS) proc_bind(spread)
      // for (std::size_t sock_id=0; sock_id<NUM_SOCKETS; sock_id++)
      // {
      //   #pragma omp parallel for reduction(sum: result) schedule(static, STATIC_SCHEDULE_CHUNK_SIZE) num_threads(inner_num_threads) proc_bind(close)
      //   for (std::size_t block_idx = sock_id * numOfBlocks_div_NumSockets; block_idx < (sock_id + 1) * numOfBlocks_div_NumSockets + (sock_id + 1 == NUM_SOCKETS) * (num_of_blocks % NUM_SOCKETS); block_idx++)
      //   {
      //     // Compute Kr = KhatriRao(A_(last_mode)(l,:), A_(last_mode-1)(k,:), ..., A_(2)(j,:))
      //     // Initialize vector Kr as Kr = A_(last_mode)(l,:)
      //     Matrix Kr(rank,1);
      //     Kr = factors[last_mode].col((block_idx / rows_offset[_TnsSize - 3]) % tnsDims[last_mode]);
      //     Matrix PartialKR(rank, tnsDims[first_mode]);

      //     // compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(j)(...,:)
      //     for (std::size_t i = _TnsSize - 4, j = last_mode - 1; i >= 0; i--, j--)
      //     {
      //       if (j == mode)
      //       {
      //         j--;
      //       }
      //       Kr = (factors[j].col((block_idx / rows_offset[i]) % tnsDims[j])).cwiseProduct(Kr);
      //     }

      //     // Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(first_mode), as : KhatriRao(Kr, A_(first_mode)(:,:))
      //     for (int col = 0; col < tnsDims[first_mode]; col++)
      //     {
      //       PartialKR.col(col) = ((factors[first_mode].col(col)).cwiseProduct(Kr));
      //     }
          
      //     result.noalias() += tns_mat.block(0, block_idx * tnsDims[first_mode], tnsDims[mode], tnsDims[first_mode]) * PartialKR.transpose();                 
      //   }
      // }
      // #ifndef EIGEN_DONT_PARALLELIZE
      //     Eigen::setNbThreads(num_threads);
      // #endif

      omp_set_nested(1);
      #pragma omp parallel for num_threads(NUM_SOCKETS) proc_bind(spread)
      for (std::size_t sock_id=0; sock_id<NUM_SOCKETS; sock_id++)
      {
        #pragma omp parallel for reduction(sum: result) schedule(static, STATIC_SCHEDULE_CHUNK_SIZE) num_threads(inner_num_threads) proc_bind(close)
        for (std::size_t block_idx = sock_id * numOfBlocks_div_NumSockets; block_idx < (sock_id + 1) * numOfBlocks_div_NumSockets + (sock_id + 1 == NUM_SOCKETS) * (num_of_blocks % NUM_SOCKETS); block_idx++)
        // for (int block_idx = 0; block_idx < num_of_blocks; block_idx++)
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

          result.noalias() += tns_mat.block(0, block_idx * tnsDims[first_mode], tnsDims[mode], tnsDims[first_mode]) * PartialKR;                 
        }
      }
      #ifndef EIGEN_DONT_PARALLELIZE
          Eigen::setNbThreads(num_threads);
      #endif
    }

  } // end namespace v1

} // end namespace partensor

#endif // PARTENSOR_TENSOR_KHATRIRAO_PRODUCT_HPP
