// This project requires CUDA version 11.0 or later.
// #define USE_CUDA 1

#define USE_MPI 0
#define MAX_NUM_STREAMS 8

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "add_vectors.hpp"

#include "PARTENSOR.hpp"
#include "CUDAMTTKRP.hpp"
#include "MTTKRP.hpp"

// #include "../../include/cuda_mttkrp.hpp"
// #include "../../include/createTensorFromFactors.hpp"
// #include "../../include/timers.hpp"
// #include "../../include/mttkrp.hpp"

// #define TNS_ORDER 3

int main ( void ){
    //
    static constexpr std::size_t tensor_order = 3;
    static constexpr std::size_t rank = 40;
    static constexpr int dims = 500;
    std::array<int, tensor_order> tnsDims = {dims, dims, dims};

    using Tensor = partensor::Tensor<tensor_order>;
    using Matrix = partensor::Matrix;

    std::array<Matrix, tensor_order> true_factors;
    std::array<Matrix, tensor_order> true_factors_T;

    std::cout << "> Tensor of -Order \t =  " << tensor_order << "\n\t    -Rank \t = " << rank << "\n\t    -Dimensions  =";
    for (int mode = 0; mode < tensor_order; mode++)
    {
        std::cout << " " << tnsDims[mode];
    }
    
    std::cout << std::endl;

    for(std::size_t i=0; i<tensor_order; ++i)
    {
        true_factors[i] =  Matrix::Random(tnsDims[i], (int)rank);
        true_factors_T[i] = true_factors[i].transpose();
    }

    // tnsX.resize(tnsDims);
    Matrix tnsX_mat_0 = partensor::generateTensor(0, true_factors);
    // tnsX = partensor::matrixToTensor(tnsX_mat_0, tnsDims); // matrix based

    const int mode = 0;
    int mode_1 = 1;

    Matrix MTTKRP = Matrix::Zero(tnsDims[mode], rank);

    /*======================================================================================================================================================*/
    // Initialize CUDA environment...
    cublasHandle_t handle;

    std::array<cudaStream_t, MAX_NUM_STREAMS> Stream;
    std::array<double *, MAX_NUM_STREAMS> CUDA_Tensor_X;
    std::array<double *, MAX_NUM_STREAMS> CUDA_Partial_KRP;
    std::array<double *, MAX_NUM_STREAMS> CUDA_MTTKRP;

    // VectorXi tensor_dims
    // Allocate Blocks once for each mode n. Put in ALS_CPD

    for(int str_id = 0; str_id < MAX_NUM_STREAMS; str_id++)
    {
        cudaMallocManaged((void **) &CUDA_Tensor_X[str_id],     tnsDims[mode]    * tnsDims[mode_1] * sizeof(double)); // unified mem. for Matr. Tensor X,
        cudaMallocManaged((void **) &CUDA_Partial_KRP[str_id],  tnsDims[mode_1]  *     rank        * sizeof(double)); // unified mem. for Partial KRP,
        cudaMallocManaged((void **) &CUDA_MTTKRP[str_id],       tnsDims[mode]    *     rank        * sizeof(double)); // unified mem. for MTTKRP.
        cudaStreamCreate(&Stream[str_id]);
    }
    cudaDeviceSynchronize();


    cublasCreate(&handle);

    // -- Compute MTTKRP on GPU --
    partensor::timer.startChronoHighTimer();
    partensor::MallocManaged::hybrid_batched_mttkrp(tnsDims, true_factors, tnsX_mat_0, mode, partensor::v1::get_num_threads(),
                                                    CUDA_Tensor_X, CUDA_Partial_KRP, CUDA_MTTKRP, handle, Stream, MTTKRP);

    // partensor::MallocManaged::transposed::hybrid_batched_mttkrp(tnsDims, true_factors_T, tnsX_mat_0, mode, partensor::v1::get_num_threads(),
    //                                                 CUDA_Tensor_X, CUDA_Partial_KRP, CUDA_MTTKRP, handle, Stream, MTTKRP);

    double end_cuda = partensor::timer.endChronoHighTimer();
    std::cout << "Finished after : " << end_cuda << " sec (outer)" << "\tusing " << partensor::v1::get_num_threads() << " threads and " << MAX_NUM_STREAMS << " streams" << std::endl; 

    // -- Compute MTTKRP on CPU --
    Matrix MTTKRP2 = Matrix::Zero(tnsDims[mode], rank);

    partensor::timer.startChronoHighTimer();
    partensor::mttkrp(tnsDims, true_factors, tnsX_mat_0, 0, partensor::v1::get_num_threads(), MTTKRP2);
    end_cuda = partensor::timer.endChronoHighTimer();
    std::cout << "Finished after : " << end_cuda << " sec (outer)" << std::endl;

    // Print Results
    std::cout << "-----\t-----\t-----\t-----\t-----\t-----" << std::endl;
    std::cout << " | MTTKRP | = " << MTTKRP.norm() << " using GPU" << std::endl;
    std::cout << " | MTTKRP | = " << MTTKRP2.norm() << " using CPU" << std::endl;
    double error = (MTTKRP2 - MTTKRP).norm();
    if (error > 1e-4)
    {
        std::cout << "\033[1;37;41m[ERROR!]\033[0m" << "\033[1;31m: | Error | = " << error << "\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[1;37;42m[OK]\033[0m" << "\033[1;32m: | Error | = " << error << "\033[0m" << std::endl;
    }
    std::cout << "---------------------------------------------" << std::endl;

    // Put in ALS_CPD
    // free memory
    for(int str_id = 0; str_id < MAX_NUM_STREAMS; str_id++)
    {
        cudaFree(CUDA_Tensor_X[str_id]);
        cudaFree(CUDA_Partial_KRP[str_id]);
        cudaFree(CUDA_MTTKRP[str_id]);
        cudaStreamDestroy(Stream[str_id]);
    }

    cublasDestroy(handle); // destroy CUBLAS context...

    return 0 ;
}
