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
    constexpr std::size_t rank         = 150;
    
    using Tensor     = partensor::Tensor<tensor_order>;
    using Status     = partensor::CudaStatus<Tensor>;
    using Options    = partensor::CudaOptions<Tensor>;
    using Constraint = partensor::Constraint;

    std::array<int,tensor_order>        tnsDims = {200, 200, 200};
    std::array<Constraint,tensor_order> constraints;

    std::string path = "../data/tnsX.bin";

    Options options;

    options.max_iter = 10;
    options.nesterov_delta_1 = 1e-6;
    options.nesterov_delta_2 = 1e-6;
    options.constraints[0] = Constraint::nonnegativity;
    options.constraints[1] = Constraint::nonnegativity;
    options.constraints[2] = Constraint::nonnegativity;
    options.writeToFile = false;
    
    Status status = partensor::cpd(partensor::execution::cuda, tnsDims, rank, path, options);

    return 0;   
}
