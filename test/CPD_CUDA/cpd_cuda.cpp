#define EIGEN_DONT_PARALLELIZE
#define USE_CUDA 1
#define MAX_NUM_STREAMS 8

#include <iostream>
#include "omp.h"
#include "PARTENSOR.hpp"

int main(int argc, char** argv)
{
    partensor::Init(argc,argv);
    
    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 50;
    
    using Tensor     = partensor::Tensor<tensor_order>;
    using Status     = partensor::CudaStatus<Tensor>;
    using Options    = partensor::CudaOptions<Tensor>;
    using Constraint = partensor::Constraint;
    Options options;

    options.max_iter = 5;
    std::array<int,tensor_order>        tnsDims = {200, 200, 200};
    std::array<Constraint,tensor_order> constraints;

    Tensor tnsX;

    std::fill(constraints.begin(), constraints.end(), Constraint::unconstrained);
    partensor::makeTensor(tnsDims, constraints, rank, tnsX);
    Status status = partensor::cpd(partensor::execution::cuda, tnsX, rank, options);

    // for(std::size_t i=0; i<tensor_order; ++i)
    //   std::cout << "\n factor " << i << "\n" << status.factors[i] << std::endl;

    return 0;   
}