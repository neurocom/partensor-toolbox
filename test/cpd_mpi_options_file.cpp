#define USE_MPI 1

#include "PARTENSOR.hpp"

int main(int argc, char** argv)
{
    partensor::Init(argc,argv);
    partensor::MPI_Communicator _comm = partensor::Partensor()->MpiCommunicator();
    
    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 2;
    
    using Tensor     = partensor::Tensor<tensor_order>;
    using Status     = partensor::MpiStatus<Tensor>;
    using Options    = partensor::MpiOptions<Tensor>;
    using Constraint = partensor::Constraint;
    
    Options options;

    std::array<int,tensor_order> tnsDims = {10, 11, 12};
    std::string                  path    = "../data/tns.bin";

    options.max_iter       = 50;
    options.constraints[0] = Constraint::nonnegativity;
    options.constraints[1] = Constraint::unconstrained;
    options.constraints[2] = Constraint::orthogonality;

    Status status = partensor::cpd(partensor::execution::mpi, tnsDims, rank, path, options);
    
    if(_comm.rank() == 0)
    {
        for(std::size_t i=0; i<tensor_order; ++i)
            std::cout << "\n factor " << i << "\n" << status.factors[i] << std::endl;
    }
    
    return 0;   
}