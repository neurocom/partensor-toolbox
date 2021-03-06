#define USE_MPI 1

#include "PARTENSOR.hpp"

int main(int argc, char** argv)
{
    partensor::Init(argc,argv);
    partensor::MPI_Communicator _comm  = partensor::Partensor()->MpiCommunicator();

    constexpr std::size_t tensor_order = 3;
    constexpr std::size_t rank         = 2;
    
    using Matrix     = partensor::Matrix;
    using Tensor     = partensor::Tensor<tensor_order>;
    using Status     = partensor::MpiStatus<Tensor>;
    using Options    = partensor::MpiOptions<Tensor>;
    using Constraint = partensor::Constraint;

    std::array<int,tensor_order>    tnsDims = {10, 11, 12};
    std::array<Matrix,tensor_order> init_factors;

    Tensor  tnsX;
    Options options;

    options.max_iter       = 50;
    options.constraints[0] = Constraint::nonnegativity;
    options.constraints[1] = Constraint::unconstrained;
    options.constraints[2] = Constraint::orthogonality;
    
    partensor::makeTensor(tnsDims, options.constraints, rank, tnsX);
    partensor::makeFactors(tnsDims, options.constraints, rank, init_factors);

    Status status = partensor::cpdDimTree(partensor::execution::mpi, tnsX, rank, options, init_factors);
    
    if(_comm.rank() == 0)
    {
        for(std::size_t i=0; i<tensor_order; ++i)
        std::cout << "\n factor " << i << "\n" << status.factors[i] << std::endl;
    }
    
    return 0;   
}