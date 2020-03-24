#include "PARTENSOR.hpp"

using namespace partensor;

int main()
{
    constexpr std::size_t        tensor_order = 3;

    constexpr int                dim0         = 3;
    constexpr int                dim1         = 4;
    constexpr int                dim2         = 5;
    
    std::array<int,tensor_order> tensor_dims  = {dim0, dim1, dim2};
    Tensor<tensor_order>         tnsX;

    tnsX.resize(tensor_dims);
    generateRandomTensor(tnsX);

    Matrix mat1(dim0, dim1*dim2);
    Matrix mat2(dim1, dim0*dim2);
    Matrix mat3(dim2, dim0*dim1);

    mat1 = Matricization(tnsX, 0);
    mat2 = Matricization(tnsX, 1);
    mat3 = Matricization(tnsX, 2);

    return 0;
}