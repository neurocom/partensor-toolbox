#include "PARTENSOR.hpp"

int main(int argc, char** argv)
{
    partensor::Init(argc,argv);

    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 2;
    
    using Tensor     = partensor::Tensor<tensor_order>;
    using Status     = partensor::Status<Tensor>;
    using Options    = partensor::Options<Tensor>;
    using Constraint = partensor::Constraint;

    std::array<int,tensor_order> tnsDims = {10, 11, 12};

    Tensor  tnsX;
    Options options;

    options.max_iter       = 50;
    options.constraints[0] = Constraint::nonnegativity;
    options.constraints[1] = Constraint::unconstrained;
    options.constraints[2] = Constraint::orthogonality;
    
    partensor::makeTensor(tnsDims, options.constraints, rank, tnsX);

    Status status = partensor::cpdDimTree(tnsX, rank, options);
    
    for(std::size_t i=0; i<tensor_order; ++i)
      std::cout << "\n factor " << i << "\n" << status.factors[i] << std::endl;

    return 0;   
}