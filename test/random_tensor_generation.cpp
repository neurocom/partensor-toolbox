#include "PARTENSOR.hpp"
#include <iostream>

using namespace partensor;

int main()
{
    constexpr std::size_t        tensor_order = 3;

    std::array<int,tensor_order> tnsDims = {10, 11, 12};

    Tensor<tensor_order>         tnsX;
    tnsX.resize(tnsDims);
    
    // zero variable can also be ignored
    generateRandomTensor(tnsX, 0); 
    std::cout << "Uniform distribution\n" << tnsX << std::endl;

    generateRandomTensor(tnsX, 1);
    std::cout << "\nNormal distribution\n" << tnsX << std::endl;
    
    return 0;
}