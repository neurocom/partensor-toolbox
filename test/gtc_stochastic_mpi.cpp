#define USE_MPI 1

#include <iostream>
#include <cstdlib>      // std::rand, std::srand
#include "PARTENSOR.hpp"

using namespace partensor;

int main(int argc, char** argv)
{
    static constexpr std::size_t      TnsSize      = 3;
    const std::size_t                 R            = 10;
    const int                         nnz          = 3000;
  	std::array<int, TnsSize>          tnsDims      = {50, 50, 50};
    
    using SparseTensor = partensor::SparseTensor<TnsSize>;//std::array<SparseMatrix, _TnsSize>;
    using DataType     = SparseTensorTraits<SparseTensor>::DataType;
    using Matrix       = SparseTensorTraits<SparseTensor>::MatrixType;
    using MatrixArray  = SparseTensorTraits<SparseTensor>::MatrixArray;
    using Constraints  = SparseTensorTraits<SparseTensor>::Constraints;

    using Status   = partensor::MpiSparseStatus<TnsSize>;
    using Options  = partensor::MpiSparseOptions<TnsSize>;

  	Constraints       constraints;
    Matrix            Ratings_Base_T(static_cast<int>(TnsSize)+1, nnz);
    
    std::array<int, TnsSize> procs = {2,2,2};

    for (int mode = 0; mode < static_cast<int>(TnsSize) + 1; mode++)
    {
        if(mode < static_cast<int>(TnsSize))
        {
            for (int counter=0; counter<nnz; counter++)
            {
                Ratings_Base_T(mode, counter) = rand() % (tnsDims[mode]);
            }
        }
        else
        {
            for (int counter=0; counter<nnz; counter++)
            {
                Ratings_Base_T(mode, counter) = static_cast<double>(rand()) / RAND_MAX;
            }
        }
    }

    // std::fill(constraints.begin(), constraints.end(), Constraint::unconstrained);
    std::fill(constraints.begin(), constraints.end(), Constraint::nonnegativity);
    // std::fill(constraints.begin(), constraints.end(), Constraint::symmetric_nonnegativity);

    ptl::MPI_Communicator _comm = ptl::Partensor()->MpiCommunicator();

    Options               opt;
    opt.proc_per_mode     = procs;
    opt.rank              = R;
    opt.tnsDims           = tnsDims;
    opt.nonZeros          = nnz;
    opt.max_nesterov_iter = 20;
    opt.c_stochastic_perc = 0.5;
    
    opt.constraints            = constraints;
    opt.acceleration           = true;
    opt.initialized_factors    = false;
    opt.read_factors_from_file = false;
    ptl::Init(argc,argv);

    Status s1 = partensor::gtc_stochastic(ptl::execution::mpi, Ratings_Base_T, opt);

    if(_comm.rank() == 0)
    {
        std::cout << "Relative cost function: " << s1.f_value/s1.frob_tns << std::endl;
    }

    return 0;   
}