#include "PARTENSOR.hpp"
#include <iostream>

using namespace partensor;

int main()
{
    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 5;

    std::array<int, tensor_order>        tensor_dimensions = {12, 10, 20};
    std::array<Constraint, tensor_order> constraints;
    
    std::fill(constraints.begin(), constraints.end(), Constraint::unconstrained);
    
    // In case of Matrix Module
    std::array<Matrix, tensor_order>  matrix_factors;
    makeFactors(tensor_dimensions, constraints, rank, matrix_factors);
    std::cout << "matrix_factors[0]\n" << matrix_factors[0] << std::endl;

    // In case of Tensor<2> Module
    std::array<Tensor<2>, tensor_order>  tensor_factors;
    makeFactors(tensor_dimensions, constraints, rank, tensor_factors);
    std::cout << "tensor_factors[1]\n" << tensor_factors[1] << std::endl;

    // In case of FactorDimTree Module
    std::array<FactorDimTree, tensor_order>  factors_DimTrees;
    makeFactors(tensor_dimensions, constraints, rank, factors_DimTrees);
    std::cout << "factors_DimTrees[2]\n" << factors_DimTrees[2].factor << std::endl;

    return 0;
}