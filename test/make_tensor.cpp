#include "PARTENSOR.hpp"
#include <iostream>

int main()
{
    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 2;
    
    using Tensor     = partensor::Tensor<tensor_order>;
    using Constraint = partensor::Constraint;

    std::array<int,tensor_order>        tnsDims = {10, 11, 12};
    std::array<Constraint,tensor_order> constraints;

    Tensor tnsX;

    std::fill(constraints.begin(), constraints.end(), Constraint::unconstrained);
    partensor::makeTensor(tnsDims, constraints, rank, tnsX);

    std::cout << "Tensor \n" << tnsX << std::endl;
    return 0;   
}