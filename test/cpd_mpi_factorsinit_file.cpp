#define USE_MPI 1

#include "PARTENSOR.hpp"

int main(int argc, char** argv)
{
    partensor::Init(argc,argv);
    partensor::MPI_Communicator _comm = partensor::Partensor()->MpiCommunicator();
    
    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 2;
    
    using Tensor = partensor::Tensor<tensor_order>;
    using Status = partensor::MpiStatus<Tensor>;

    std::array<int,tensor_order>            tnsDims = {10, 11, 12};
    std::array<std::string, tensor_order+1> paths;

    paths[0] = "../data/tnsX_3.bin";
    paths[1] = "../data/A_3.bin";
    paths[2] = "../data/B_3.bin";
    paths[3] = "../data/C_3.bin";
    Status status = partensor::cpd(partensor::execution::mpi, tnsDims, rank, paths);
    
    if(_comm.rank() == 0)
    {
        for(std::size_t i=0; i<tensor_order; ++i)
            std::cout << "\n factor " << i << "\n" << status.factors[i] << std::endl;
    }
    
    return 0;   
}