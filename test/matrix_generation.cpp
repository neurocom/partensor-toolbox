#include "PARTENSOR.hpp"
#include <iostream>

using namespace partensor;

int main()
{
    Matrix mat(4,5);
    generateRandomMatrix(mat);

    std::cout << "Matrix\n" << mat << std::endl;
    return 0;
}