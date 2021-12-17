#include <iostream>
#include "PARTENSOR.hpp"

int main()
{
    constexpr std::size_t        tensor_order = 3;
    constexpr std::size_t        rank         = 2;
    std::array<int,tensor_order> tensor_dims  = {10, 10, 10};
    
    using  Tensor    = partensor::Tensor<tensor_order>;
    using  Status    = partensor::Status<Tensor>;
    
    Tensor tnsX;
    tnsX.resize(tensor_dims);

    partensor::generateRandomTensor(tnsX);
    Status status = partensor::cpd(tnsX, rank);
    
    for(std::size_t i=0; i<tensor_order; ++i)
      std::cout << "\n factor " << i << "\n" << status.factors[i] << std::endl;

    return 0;   
}