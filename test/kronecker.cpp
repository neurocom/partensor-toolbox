#include "PARTENSOR.hpp"

using namespace partensor;

int main()
{
    const int row = 5;
    const int col = 4;

    Matrix A(row, col);
    Matrix B(row, col);
    Matrix C(row, col);
    Matrix D(row, col);

    generateRandomMatrix(A);
    generateRandomMatrix(B);
    generateRandomMatrix(C);
    generateRandomMatrix(D);

    Matrix result_2(row*row, col*col);
    Matrix result_3(row*row*row, col*col*col);
    Matrix result_4(row*row*row*row, col*col*col*col);

    result_2 = Kronecker(A, B);
    result_3 = Kronecker(A, B, C);
    result_4 = Kronecker(A, B, C, D);

    return 0;
}