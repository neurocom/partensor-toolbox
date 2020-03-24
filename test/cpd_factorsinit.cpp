#include "PARTENSOR.hpp"

int main(int argc, char** argv)
{
    partensor::Init(argc,argv);

    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 2;
    
    using Matrix     = partensor::Matrix;
    using Tensor     = partensor::Tensor<tensor_order>;
    using Status     = partensor::Status<Tensor>;
    using Constraint = partensor::Constraint;

    std::array<int,tensor_order>        tnsDims = {10, 11, 12};
    std::array<Matrix,tensor_order>     init_factors;
    std::array<Constraint,tensor_order> constraints;

    Tensor  tnsX;
    
    std::fill(constraints.begin(), constraints.end(), Constraint::unconstrained);
    partensor::makeTensor(tnsDims, constraints, rank, tnsX);
    partensor::makeFactors(tnsDims, constraints, rank, init_factors);

    Status status = partensor::cpd(tnsX, rank, init_factors);
    
    for(std::size_t i=0; i<tensor_order; ++i)
      std::cout << "\n factor " << i << "\n" << status.factors[i] << std::endl;

    return 0;   
}