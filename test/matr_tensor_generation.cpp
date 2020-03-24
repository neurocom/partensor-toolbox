#include "PARTENSOR.hpp"
#include <iostream>

using namespace partensor;

int main()
{
    const int rank = 5;
    const int rowA = 10;
    const int rowB = 12;
    const int rowC = 15;

    Matrix A(rowA, rank);
    Matrix B(rowB, rank);
    Matrix C(rowC, rank);
    
    generateRandomMatrix(A);
    generateRandomMatrix(B);
    generateRandomMatrix(C);

    std::array<Matrix, 3> factors = {A, B, C};

    Matrix matricized_tensor = generateTensor(0, factors);
    std::cout << "Matricization of first mode\n" << matricized_tensor << std::endl;
    
    return 0;
}