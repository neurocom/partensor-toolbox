#include "PARTENSOR.hpp"
#include <iostream>

using namespace partensor;

int main()
{
    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t matrix_order = 2;

    using Tensor_2d = Tensor<matrix_order>;

    const int rank = 5;
    const int rowA = 10;
    const int rowB = 12;
    const int rowC = 15;

    Tensor_2d A(rowA, rank);
    Tensor_2d B(rowB, rank);
    Tensor_2d C(rowC, rank);

    generateRandomTensor(A);
    generateRandomTensor(B);
    generateRandomTensor(C);

    std::array<Tensor_2d, tensor_order> factorArray = {A, B, C};
    Tensor<tensor_order>  tnsX;

    tnsX = generateTensor(factorArray);
    std::cout << "Tensor\n" << tnsX << std::endl;

    return 0;
}