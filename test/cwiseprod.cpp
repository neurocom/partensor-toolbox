#include "PARTENSOR.hpp"

using namespace partensor;

int main()
{
    const int row = 5;
    const int col = 5;

    Matrix A(row, col);
    Matrix B(row, col);
    Matrix C(row, col);
    Matrix D(row, col);

    generateRandomMatrix(A);
    generateRandomMatrix(B);
    generateRandomMatrix(C);
    generateRandomMatrix(D);

    Matrix result_2(row, col);
    Matrix result_3(row, col);
    Matrix result_4(row, col);

    result_2 = CwiseProd(A, B);
    result_3 = CwiseProd(A, B, C);
    result_4 = CwiseProd(A, B, C, D);
    
    return 0;
}