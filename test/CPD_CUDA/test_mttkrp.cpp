// This project requires CUDA version 11.0 or later.
// #define USE_CUDA 1

#define USE_MPI 0

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
    static constexpr std::size_t rank = 100;
    static constexpr int dims = 1000;
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
    double *CUDA_Tensor_X;
    double *CUDA_Partial_KRP;
    double *CUDA_MTTKRP;

    // VectorXi tensor_dims
    // Allocate Blocks once for each mode n. Put in ALS_CPD

    cudaMallocManaged((void **) &CUDA_Tensor_X,     tnsDims[mode]    * tnsDims[mode_1] * sizeof(double)); // unified mem. for Matr. Tensor X,
    cudaMallocManaged((void **) &CUDA_Partial_KRP,  tnsDims[mode_1]  *     rank        * sizeof(double)); // unified mem. for Partial KRP,
    cudaMallocManaged((void **) &CUDA_MTTKRP,       tnsDims[mode]    *     rank        * sizeof(double)); // unified mem. for MTTKRP.

    cudaDeviceSynchronize();


    cublasCreate(&handle);

    // -- Compute MTTKRP on GPU --
    partensor::timer.startChronoHighTimer();
    partensor::MallocManaged::hybrid_mttkrp(tnsDims, true_factors, tnsX_mat_0, mode, partensor::v1::get_num_threads(),
                                                    CUDA_Tensor_X, CUDA_Partial_KRP, CUDA_MTTKRP, handle, MTTKRP);

    // partensor::MallocManaged::transposed::hybrid_mttkrp(tnsDims, true_factors_T, tnsX_mat_0, mode, partensor::v1::get_num_threads(),
    //                                                 CUDA_Tensor_X, CUDA_Partial_KRP, CUDA_MTTKRP, handle, MTTKRP);

    double end_cuda = partensor::timer.endChronoHighTimer();
    std::cout << "Finished after : " << end_cuda << " sec (outer)" << "\tusing " << partensor::v1::get_num_threads() << " threads" << std::endl; 

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
        std::cout << "\033[1;37;41m[ERROR!]\033[0m" << "\033[1;32m: | Error | = " << error << "\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[1;37;42m[OK]\033[0m" << "\033[1;32m: | Error | = " << error << "\033[0m" << std::endl;
    }
    std::cout << "---------------------------------------------" << std::endl;

    // free memory
    cudaFree(CUDA_Tensor_X);
    cudaFree(CUDA_Partial_KRP);
    cudaFree(CUDA_MTTKRP);

    cublasDestroy(handle); // destroy CUBLAS context...

    return 0 ;
}
