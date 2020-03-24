#define USE_OPENMP 1

#include "PARTENSOR.hpp"

int main(int argc, char** argv)
{
    partensor::Init(argc,argv);

    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 2;
    
    using Tensor  = partensor::Tensor<tensor_order>;
    using Status  = partensor::OmpStatus<Tensor>;

    std::array<int,tensor_order>             tnsDims = {10, 11, 12};
    std::array<std::string, tensor_order+1>  paths;

    paths[0] = "../data/tns.bin";
    paths[1] = "../data/A.bin";
    paths[2] = "../data/B.bin";
    paths[3] = "../data/C.bin";
    Status status = partensor::cpd(partensor::execution::omp, tnsDims, rank, paths);
    
    for(std::size_t i=0; i<tensor_order; ++i)
      std::cout << "\n factor " << i << "\n" << status.factors[i] << std::endl;

    return 0;   
}