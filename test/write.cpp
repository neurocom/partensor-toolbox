#include "PARTENSOR.hpp"

using namespace partensor;

int main()
{    
    constexpr std::size_t tensor_order = 3;
    constexpr int         dim0         = 2;
    constexpr int         dim1         = 5;
    
    std::array<int,tensor_order> tnsDims = {10, 11, 12};
    int tnsDims_prod = std::accumulate(tnsDims.begin(), tnsDims.end(), 1, std::multiplies<int>());

    Matrix               mtx(dim0,dim1);
    Tensor<tensor_order> tnsX;

    tnsX.resize(tnsDims);

    generateRandomMatrix(mtx);
    generateRandomTensor(tnsX);

    std::cout << "matrix\n" << mtx << std::endl;
    std::cout << "\ntensor\n" << tnsX << std::endl;

    write(mtx,"../data/matrix.bin",dim0*dim1);
    write(tnsX,"../data/tensor.bin",tnsDims_prod);
    
    return 0;   
}