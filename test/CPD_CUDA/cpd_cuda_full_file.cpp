#define EIGEN_DONT_PARALLELIZE
#define USE_CUDA 1
// #define USE_OMP 0
#define MAX_NUM_STREAMS 8

#include <iostream>
#include "omp.h"
#include "PARTENSOR.hpp"

int main(int argc, char** argv)
{
    partensor::Init(argc,argv);
    
    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 100;
    
    using Tensor     = partensor::Tensor<tensor_order>;
    using Status     = partensor::CudaStatus<Tensor>;
    using Options    = partensor::CudaOptions<Tensor>;
    using Constraint = partensor::Constraint;

    std::array<int,tensor_order>        tnsDims = {400, 400, 400};

    std::array<Constraint,tensor_order> constraints;

    std::array<std::string, tensor_order + 1> paths;

    paths[0] = "../data/tnsX_30_400.bin";
    paths[1] = "../data/init_factor_A_30_400.bin";
    paths[2] = "../data/init_factor_B_30_400.bin";
    paths[3] = "../data/init_factor_C_30_400.bin";

    Options options;

    options.max_iter = 100;
    options.nesterov_delta_1 = 1e-4;
    options.nesterov_delta_2 = 1e-4;
    std::fill(options.constraints.begin(), options.constraints.end(), Constraint::nonnegativity);

    options.writeToFile = true;

    options.final_factors_paths[0] = "../data/resulted_factor_0.bin";
    options.final_factors_paths[1] = "../data/resulted_factor_1.bin";
    options.final_factors_paths[2] = "../data/resulted_factor_2.bin";

    options.normalization = false;
    options.acceleration  = true;

    Status status = partensor::cpd(partensor::execution::cuda, tnsDims, rank, paths, options);


    // for(std::size_t i=0; i<tensor_order; ++i)
    //   std::cout << "\n factor " << i << "\n" << status.factors[i] << std::endl;

    return 0;   
}