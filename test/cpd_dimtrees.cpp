#include "PARTENSOR.hpp"

int main(int argc, char** argv)
{
    partensor::Init(argc,argv);
    
    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 2;
    
    using Tensor     = partensor::Tensor<tensor_order>;
    using Status     = partensor::Status<Tensor>;
    using Constraint = partensor::Constraint;

    std::array<int,tensor_order>        tnsDims = {10, 11, 12};
    std::array<Constraint,tensor_order> constraints;

    Tensor tnsX;

    std::fill(constraints.begin(), constraints.end(), Constraint::unconstrained);
    partensor::makeTensor(tnsDims, constraints, rank, tnsX);

    Status status = partensor::cpdDimTree(tnsX, rank);
    
    for(std::size_t i=0; i<tensor_order; ++i)
      std::cout << "\n factor " << i << "\n" << status.factors[i] << std::endl;

    return 0;   
}