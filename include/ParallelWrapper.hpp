#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      07/10/2019
* @author    Christos Tsalidis
* @author    Yiorgos Lourakis
* @author    George Lykoudis
* @copyright 2019 Neurocom All Rights Reserved.
*/
#endif // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      ParallelWrapper.hpp
* @details
* Implements wrapper functions from @c Boost @c mpi library and
* @c OpenMPI, necessary for this project.
********************************************************************/

#ifndef PARTENSOR_PARALLEL_WRAPPER_HPP
#define PARTENSOR_PARALLEL_WRAPPER_HPP

#include <vector>
#include "boost/mpi/communicator.hpp"
#include "boost/mpi/collectives.hpp"
#include "boost/mpi/environment.hpp"
#include "boost/mpi/cartesian_communicator.hpp"

namespace partensor {
  
  inline namespace v1 {
    /**
     * @brief @c MPI implementation using @c Boost's module mpi.
     */
    
    using Boost_Environment      = boost::mpi::environment;             /**< Typdef for @c environment class from boost. */
    using Boost_Communicator 	   = boost::mpi::communicator;            /**< Typdef for @c communicator class from boost. */
    using Boost_CartDimension    = boost::mpi::cartesian_dimension;     /**< Typdef for @c cartesian_dimension class from boost. */
    using Boost_CartTopology     = boost::mpi::cartesian_topology;      /**< Typdef for @c cartesian_topology class from boost. */
    using Boost_CartCommunicator = boost::mpi::cartesian_communicator;  /**< Typdef for @c cartesian_communicator class from boost. */

    /** @brief Initialize, finalize, and query the @c MPI environment.
     *
     *  The @c environment class is used to initialize, finalize, and
     *  query the MPI environment. 
     * 
     *  The instance of @c environment will initialize MPI (by calling @c
     *  MPI_Init) in its constructor and finalize MPI (by calling @c
     *  MPI_Finalize for normal termination or @c MPI_Abort for an
     *  uncaught exception) in its destructor.
     *
     *  The use of @c environment is not mandatory. Users may choose to
     *  invoke @c MPI_Init and @c MPI_Finalize manually. In this case, no
     *  @c environment object is needed. If one is created, however, it
     *  will do nothing on either construction or destruction.
     */
    struct environment : public Boost_Environment 
    {
      /** Initialize the MPI environment.
       * 
       * If the MPI environment has not already been initialized,
       * initializes MPI with a call to @c boost::mpi::environment.
       * 
       * @param argc                Number of arguments provided in @p argv, as 
       *                            passed into the program's @c main function.
       * @param argv                Array of argument strings passed to the program
       *                            via @c main function.
       * @param abort_on_exception  When true, this object will abort the
       *                            program if it is destructed due to an 
       *                            uncaught exception.
       */
      environment(int &argc, char** &argv, bool abort_on_exception = true) : Boost_Environment(argc, argv, abort_on_exception) {};
    
      /** Shuts down the MPI environment.
       *
       *  If this @c environment object was used to initialize the MPI
       *  environment, and the MPI environment has not already been shut
       *  down (finalized), this destructor will shut down the MPI
       *  environment. Under normal circumstances, this only involves
       *  invoking @c MPI_Finalize. However, if destruction is the result
       *  of an uncaught exception and the @c abort_on_exception parameter
       *  of the constructor had the value @c true, this destructor will
       *  invoke @c MPI_Abort with @c MPI_COMM_WORLD to abort the entire
       *  MPI program with a result code of -1.
       */
      ~environment() = default;

      using Boost_Environment::abort;
    };

    /**
     * @brief A communicator that permits communication and
     * synchronization among a set of processes.
     * 
     * The @c communicator class abstracts a set of communicating
     * processes in MPI. All of the processes that belong to a certain
     * communicator can determine the size of the communicator, their rank 
     * within the communicator, and communicate with any other processes
     * in the communicator.
     */
    struct communicator : public Boost_Communicator
    {
      /**
       * Build a new MPI communicator for @c MPI_COMM_WORLD.
       *
       * Constructs a MPI communicator that attaches to @c
       * MPI_COMM_WORLD, using boost::mpi::communicator. 
       */
      communicator() : Boost_Communicator() { }
      
      using Boost_Communicator::rank;    /**< Determines the rank of the executing process in a communicator. */
      using Boost_Communicator::size;    /**< Determines the number of processes in a communicator. */
      using Boost_Communicator::barrier; /**< Wait for all processes within a communicator to reach the barrier. */
    };

    /**
     * @brief Specify the size and periodicity of the grid in a single dimension.
     */
    struct cartesian_dimension : public Boost_CartDimension
    {      
      /**
       * @param sz The size of the grid n this dimension. 
       * @param p  Is the grid periodic in this dimension. 
       */
      cartesian_dimension(int sz = 0, bool p = true) : Boost_CartDimension(sz,p) {}
    };

    /**
     * @brief Describe the topology of a cartesian grid.
     * 
     * Behave mostly like a sequence of @c cartesian_dimension with the notable 
     * exception that its size is fixed.
     */
    struct cartesian_topology : public Boost_CartTopology
    { 
      /**
       * @brief Use dimensions specification provided in the sequence container as initial values.
       * @param dims must be a sequence container.
       */ 
      template<class InitArr_>
      cartesian_topology(InitArr_ dims) : Boost_CartTopology(dims) {};
    };

    /**
     * @brief An MPI communicator with a cartesian topology.
     *
     * A @c cartesian_communicator is a communicator whose topology is
     * expressed as a grid. Cartesian communicators have the same
     * functionality as communicators, but also allow one to query
     * the relationships among processes and the properties of the grid.
     */
    struct cartesian_communicator : public Boost_CartCommunicator
    {
      friend struct communicator;
      using communicator::rank;

      /**
       *  Create a new communicator whose topology is described by the
       *  given cartesian. The indices of the vertices in the cartesian will be
       *  assumed to be the ranks of the processes within the
       *  communicator. There may be fewer vertices in the cartesian than
       *  there are processes in the communicator; in this case, the
       *  resulting communicator will be a NULL communicator.
       *
       *  @param comm The communicator that the new, cartesian communicator
       *  will be based on. 
       * 
       *  @param dims the cartesian dimension of the new communicator. The size indicate 
       *  the number of dimension. Some dimensions be set to zero, in which case
       *  the corresponding dimension value is left to the system.
       *  
       *  @param reorder Whether MPI is permitted to re-order the process
       *  ranks within the returned communicator, to better optimize
       *  communication. If true, the ranks of each process in the returned process
       *  will be new starting from zero.
       */
      cartesian_communicator( communicator       const &comm,
                              cartesian_topology const &dims,
                              bool               reorder = true) : Boost_CartCommunicator(comm, cartesian_topology(dims), reorder) {}

      /**
       * Create a new cartesian communicator whose topology is a subset of
       * an existing cartesian communicator.
       * @param comm the original communicator.
       * @param keep and array containing the dimension to keep from the existing 
       * communicator.
       */
      cartesian_communicator( cartesian_communicator const &comm,
                              std::vector<int>       const &keep) : Boost_CartCommunicator(comm, keep) {}
    };

    /**
     * @brief Wrapper type to explicitly indicate that a input data 
     * can be overriden with an output value.
     */
    template<typename T>
    using inplace_t = typename boost::mpi::inplace_t<T>;
    
    /**
     *  @brief Wrap an input data to indicate that it can be overriden 
     *  with an ouput value.
     *  @param inout the contributing input value, it will be overriden 
     *  with the output value where one is expected. If it is a pointer, 
     *  the number of elements will be provided separately.
     * 
     *  @returns The wrapped value or pointer.
     */
    // template<typename T>
    // inplace_t<T> inplace(T& inout) { 
    //   return inplace_t<T>(inout);
    // }
    
    template<typename T>
    inplace_t<T*> inplace(T* inout) { 
      return inplace_t<T*>(inout);
    }

    /**
     *  @brief Compute the maximum of two values.
     */
    template<typename T>
    using maximum = typename boost::mpi::maximum<T>;

    /**
     *  @brief Compute the minimum of two values.
     */
    template<typename T>
    using minimum = typename boost::mpi::minimum<T>;

    /**
     *  @brief Combine the values stored by each process into a single
     *  value available to all processes.
     *
     *  @c all_reduce is a collective algorithm that combines the values
     *  stored by each process into a single value available to all
     *  processes. The values are combined in a user-defined way,
     *  specified via a function object. The type @c T of the values may
     *  be any type that is serializable or has an associated MPI data
     *  type.
     *
     *  When the type @c T has an associated MPI data type, this routine
     *  invokes @c MPI_Allreduce to perform the reduction. If possible,
     *  built-in MPI operations will be used; otherwise, @c all_reduce()
     *  will create a custom MPI_Op for the call to MPI_Allreduce.
     *
     *    @param comm [in] The communicator over which the reduction will
     *    occur.
     * 
     *    @param value [in] The local value to be combined with the local
     *    values of every other process. For reducing arrays, @c in_values
     *    is a pointer to the local values to be reduced and @c n is the
     *    number of values to reduce. See @c reduce for more information.
     *
     *    If wrapped in a @c inplace_t object, combine the usage of both
     *    input and $c out_value and the local value will be overwritten
     *    (a convenience function @c inplace is provided for the wrapping).
     *
     *    @param out_value [in,out] Will receive the result of the reduction
     *    operation. If this parameter is omitted, the outgoing value will
     *    instead be returned.
     *
     *    @param n [in] Indicated the size of the buffers of array type.
     *    @returns If no @p out_value parameter is supplied, returns the
     *    result of the reduction operation.
     * 
     *    @param op [in] The binary operation that combines two values of type
     *    @c T and returns a third value of type @c T. 
     */
    template<typename T, typename Op>
    inline void all_reduce( const cartesian_communicator &comm, 
                            inplace_t<T*>                 value, 
                            int                           n, 
                            Op                            op   )
    {
      boost::mpi::all_reduce(static_cast<const Boost_CartCommunicator &>(comm), value, n, op);
    }

    template<typename T, typename Op>
    inline void all_reduce( const cartesian_communicator &comm, 
                            T                            &value, 
                            T                            &out_value, 
                            Op                            op       )
    {
      boost::mpi::all_reduce(static_cast<const Boost_CartCommunicator &>(comm), value, out_value, op);
    }

    template<typename T, typename Op> 
    inline void all_reduce( const cartesian_communicator &comm, 
                            const T                      *value, 
                            int                           n, 
                            T                            *out_value, 
                            Op                            op)
    {
      boost::mpi::all_reduce(static_cast<const Boost_CartCommunicator &>(comm), value, n, out_value, op);
    }
    
    /*
     *  @brief Combine the values stored by each process into a single
     *  value at the root.
     *
     *  @c reduce is a collective algorithm that combines the values
     *  stored by each process into a single value at the @c root. The
     *  values can be combined arbitrarily, specified via a function
     *  object. The type @c T of the values may be any type that is
     *  serializable or has an associated MPI data type. One can think of
     *  this operation as a @c gather to the @p root, followed by an @c
     *  std::accumulate() over the gathered values and using the operation
     *  @c op. 
     *
     *  When the type @c T has an associated MPI data type, this routine
     *  invokes @c MPI_Reduce to perform the reduction. If possible,
     *  built-in MPI operations will be used; otherwise, @c reduce() will
     *  create a custom MPI_Op for the call to MPI_Reduce.
     *
     *    @param comm [in] The communicator over which the reduction will
     *    occur.
     *
     *    @param in_values [in] The local value to be combined with the local
     *    values of every other process. For reducing arrays, @c in_values
     *    contains a pointer to the local values. In this case, @c n is
     *    the number of values that will be reduced. Reduction occurs
     *    independently for each of the @p n values referenced by @p
     *    in_values, e.g., calling reduce on an array of @p n values is
     *    like calling @c reduce @p n separate times, one for each
     *    location in @p in_values and @p out_values.
     *
     *    @param out_values [in,out] Will receive the result of the reduction
     *    operation, but only for the @p root process. Non-root processes
     *    may omit if parameter; if they choose to supply the parameter,
     *    it will be unchanged. For reducing arrays, @c out_values
     *    contains a pointer to the storage for the output values.
     *
     *    @param op [in] The binary operation that combines two values of type
     *    @c T into a third value of type @c T. For types @c T that has
     *    associated MPI data types, @c op will either be translated into
     *    an @c MPI_Op (via @c MPI_Op_create) or, if possible, mapped
     *    directly to a built-in MPI operation. See @c is_mpi_op in the @c
     *    operations.hpp header for more details on this mapping. For any
     *    non-built-in operation, commutativity will be determined by the
     *    @c is_commmutative trait (also in @c operations.hpp): users are
     *    encouraged to mark commutative operations as such, because it
     *    gives the implementation additional latitude to optimize the
     *    reduction operation.
     *
     *    @param root [in] The process ID number that will receive the final,
     *    combined value. This value must be the same on all processes.
     */
    template<typename T, typename Op>
    void reduce(const cartesian_communicator &comm, 
                const T*                      in_values, 
                int                           n, 
                T*                            out_values, 
                Op                            op, 
                int                           root      )
    {
      boost::mpi::reduce(static_cast<const Boost_CartCommunicator &>(comm), in_values, n, out_values, op, root );
    }     

    /*
     *  @brief Similar to boost::mpi::scatter with the difference that the number
     *  of values stored at the root process does not need to be a multiple of
     *  the communicator's size.
     *
     *    @param comm [in] The communicator over which the scatter will occur.
     *
     *    @param in_values [in] A vector or pointer to storage that will contain
     *    the values to send to each process, indexed by the process rank.
     *    For non-root processes, this parameter may be omitted. If it is
     *    still provided, however, it will be unchanged.
     *
     *    @param sizes [in] A vector containing the number of elements each non-root
     *    process will receive.
     *
     *    @param out_values [in,out] The array of values received by each process.
     *
     *    @param root [in] The process ID number that will scatter the
     *    values. This value must be the same on all processes.
     */
    template<typename T>
    void scatterv(const cartesian_communicator &comm, 
                  const T*                      in_values, 
                  const std::vector<int>       &sizes, 
                  T*                            out_values, 
                  int                           root      )   
    {
      boost::mpi::scatterv(static_cast<const Boost_CartCommunicator &>(comm), in_values, sizes, out_values, root);
    }   

    /*
     *  @brief Gather the values stored at every process into a vector of
     *  values from each process.
     *
     *  @c all_gatherv is a collective algorithm that collects the values
     *  stored at each process into a vector of values at each
     *  process. This vector is indexed by the process number that the
     *  value came from. The type @c T of the values may be any type that
     *  is serializable or has an associated MPI data type.
     *
     *    @param comm The communicator over which the gather will occur.
     *
     *    @param in_values The array of values to be transmitted by each process.
     *
     *    @param in_size For each process this specifies the size of @p in_values.
     *
     *    @param out_values A pointer to storage that will be populated with
     *    the values from each process.
     * 
     *    @param sizes A vector containing the number of elements each 
     *    process will send.
     *
     *    @param displs A vector such that the i-th entry specifies the
     *    displacement (relative to @p out_values) from which to take the ingoing
     *    data at the @p root process. Overloaded versions for which @p displs is
     *    omitted assume that the data is to be placed contiguously at each process.
     *
     */
    template<typename T>
    void all_gatherv( const cartesian_communicator &comm, 
                      const T*                     in_values, 
                      int                          in_size,
                      T*                           out_values, 
                      const std::vector<int>      &sizes, 
                      const std::vector<int>      &displs    )
    {
      for(int layerRank = 0; layerRank<comm.size(); layerRank++)
        boost::mpi::gatherv(static_cast<const Boost_CartCommunicator &>(comm), in_values, in_size, out_values, sizes, displs, layerRank);
    }              

    /**
     * Creates a layer grid in a @c cartesian_communicator.
     * 
     * @tparam _TnsSize            Order of the Tensor.
     * @param  grid       [in]     The MPI_COMM_WORLD implemented in a 
     *                             @c cartesian @c communicator.
     * @param  layer_comm [in,out] An @c stl vector containing the newly 
     *                             created layer communicator, that still
     *                             belong in @c grid.
     * @param  layer_rank [in,out] An @c stl array containing the ranks of
     *                             each processor in each @c layer_comm.
     */
    template<std::size_t _TnsSize>
    void create_layer_grid( cartesian_communicator              &grid,
                            std::vector<cartesian_communicator> &layer_comm, 
                            std::array<int, _TnsSize>           &layer_rank )
    {
        std::vector<int>          layer_dims(_TnsSize-1);
        std::array<int, _TnsSize> free_coords;

        for (std::size_t i = 0; i < _TnsSize; ++i) 
        {
            std::fill(free_coords.begin(), free_coords.end(), 1);
            free_coords[i] = 0;
            int pos = 0;
            for (std::size_t j = 0; j < _TnsSize; ++j) {
                if (free_coords[j]) {
                    layer_dims[pos++] = free_coords[j] * j;
                }
            }
            // create the sub communicator in the cartesian communicator for layers
            layer_comm.push_back(cartesian_communicator(grid, layer_dims)); 
            // ID for each processor in layers sub communicator
            layer_rank[i] = layer_comm[i].rank();               
        }
    }

    /**
     * Creates a fiber grid in a @c cartesian_communicator.
     * 
     * @tparam _TnsSize            Order of the Tensor.
     * @param  grid       [in]     The MPI_COMM_WORLD implemented in a 
     *                             @c cartesian @c communicator.
     * @param  fiber_comm [in,out] An @c stl vector containing the newly 
     *                             created layer communicator, that still
     *                             belong in @c grid.
     * @param  fiber_rank [in,out] An @c stl array containing the ranks of
     *                             each processor in each @c fiber_comm.
     */
    template<std::size_t _TnsSize>
    void create_fiber_grid( cartesian_communicator const        &grid,
                            std::vector<cartesian_communicator> &fiber_comm, 
                            std::array<int, _TnsSize>           &fiber_rank )
    {
        std::vector<int> fiber_dims(1);

        for (std::size_t i = 0; i < _TnsSize; ++i) 
        {
            fiber_dims[0] = i;
            // create the sub communicator in the cartesian communicator for fibers
            fiber_comm.push_back(cartesian_communicator(grid, fiber_dims)); 
            // ID for each processor in fibers sub communicator      
            fiber_rank[i] = fiber_comm[i].rank();                          
        }
    }

    void DisCount(int *dis, int *count, int const size, int const dim, std::size_t const rank)
    {
      int x = dim / size;
      int y = dim % size;
      for (int i=0; i<size; i++)
      {
        count[i] = (i >= y) ? x*rank : (x+1)*rank;
        dis[i]   = 0;
        
        for (int j=0; j<i; j++)
          dis[i] += count[j];
      }
    }

    /**
     * Computes two arrays ( @c dis, @c count ) with the number of "lines"
     * from Tensor to skip and how many to read, based on tensor
     * dimensions @c dim, number of processors @c size and tensor @c rank.
     * 
     * @param dis   [in,out] The number of "lines" to skip per processor.
     * @param count [in,out] The number of "lines" to read per processor.
     * @param size  [in]     Number of processors.
     * @param dim   [in]     Tensor dimensions.
     * @param rank  [in]     Tensor rank.
     */
    void DisCount(std::vector<int> &dis, std::vector<int> &count, int const size, int const dim, std::size_t const rank)
    {
      int x = dim / size;
      int y = dim % size;
      
      for (int i=0; i<size; i++)
      {
        count.push_back((i >= y) ? x*rank : (x+1)*rank);
        dis.push_back(0);

        for (int j=0; j<i; j++)
          dis[i] += count[j];
      }
    }
       
  } // end namespace v1

  #ifndef DOXYGEN_SHOULD_SKIP_THIS
  namespace v2 {

    /*
     * Wrapper for MPI_Init, which initializes the MPI execution environment.
     * 
     * @param argc [in] Pointer to the number of arguments.
     * @param argv [in] Argument vector.
     */
    void Init(int argc, char **argv)
    {
      MPI_Init(&argc, &argv);
    }

    /*
     * Wrapper for MPI_Comm_rank, which determines the rank of the calling
     * process in the communicator.
     * 
     * @param comm [in]     Communicator.
     * @param rank [in,out] Rank of the calling process in group of comm.
     */
    void Comm_Rank( MPI_Comm const &comm,
                    int            &rank  )
    {
      MPI_Comm_rank(comm, &rank);
    }

    /*
     * Wrapper for MPI_Comm_size, which returns the size of the group
     * associated with a communicator.
     * 
     * @param comm [in]      Communicator.
     * @param size [int,out] Number of processes in the group of comm.
     */
    void Comm_Size( MPI_Comm const &comm,
                    int            &size  )
    {
      MPI_Comm_size(comm, &size);
    }

    /*
     * Wrapper for MPI_Abort, which terminates MPI execution environment.
     * 
     * @param comm      [in] Communicator.
     * @param errorCode [in] Error code to return to invoking environment.
     */
    void Abort( MPI_Comm const &comm,
                int      const  errorCode )
    {
      MPI_Abort(comm, errorCode);
    }

    /**
     * Wrapper for MPI_Allreduce, which combines values from all processes and
     * distributes the result back to all processes.
     * @tparam DataType           Type of the input/output data. Either @c Eigen Matrix or Tensor.
     * @param  comm      [in]     Communicator.
     * @param  size      [in]     Number of elements in send buffer.
     * @param  dt        [in,out] Starting address of receive buffer.
     */
    template <typename DataType_>
    void all_reduce( MPI_Comm  const &comm,
                     double    const  size,
                     DataType_       &dt    )
    {
      // In case of @c Eigen Matrix
      MPI_Allreduce(MPI_IN_PLACE, dt.data(), size, MPI_DOUBLE, MPI_SUM, comm); 
    }

    template <typename DataType_>
    void all_reduce ( MPI_Comm  const &comm,
                      DataType_       &value,
                      DataType_       &out_value,
                      int       const  size )
    {
      MPI_Allreduce(value.data(), out_value.data(), size, MPI_DOUBLE, MPI_SUM, comm);
    }

    /**
     * Wrapper for MPI_Allgatherv, which gathers data from all processes and
     * delivers it to all. Each process may contribute a different amount of data.
     * 
     * @tparam DataType          Type of the input/output data. Either Eigen Matrix or Tensor.
     * @param  comm     [in]     Communicator.
     * @param  sendBuf  [in]     Starting address of send buffer.
     * @param  sendSize [in]     Number of elements in send buffer.
     * @param  recvSize [in]     Array containing the number of elements that are received from each process.
     * @param  displs   [in]     Array where entry i specifies the displacement (relative to recvbuf) at which  
     *                           to place the incoming data from process i.
     * @param  root     [in]     Processor who will collect the data.
     * @param  recvBuf  [in,out] Address of receive buffer.
     */
    template <typename DataType_>
    void gatherv( MPI_Comm  const &comm,
                  DataType_ const &sendBuf,
                  int       const  sendSize,
                  int       const &recvSize,
                  int       const &displs,
                  int       const  root,
                  DataType_       &recvBuf)
    {
      MPI_Gatherv(sendBuf.data(), sendSize, MPI_DOUBLE, recvBuf.data(), &recvSize, &displs, MPI_DOUBLE, root, comm);
    }

    /**
     * Wrapper for MPI_Allgatherv, which gathers data from all processes and
     * delivers it to all. Each process may contribute a different amount of data.
     * @tparam DataType          Type of the input/output data. Either Eigen Matrix or Tensor.
     * @param  comm     [in]     Communicator.
     * @param  sendBuf  [in]     Starting address of send buffer.
     * @param  sendSize [in]     Number of elements in send buffer.
     * @param  recvSize [in]     Array containing the number of elements that are received from each process.
     * @param  displs   [in]     Array where entry i specifies the displacement (relative to recvbuf) at which 
     *                           to place the incoming data from process i.
     * @param  recvBuf  [in,out] Address of receive buffer.
     */
    template <typename DataType_>
    void all_gatherv( MPI_Comm  const &comm,
                     DataType_ const &sendBuf,
                     int       const  sendSize,
                     int       const &recvSize,
                     int       const &displs,
                     DataType_       &recvBuf  )
    {
      MPI_Allgatherv(sendBuf.data(), sendSize, MPI_DOUBLE, recvBuf.data(), &recvSize, &displs, MPI_DOUBLE, comm);
    }

    /**
     * Wrapper for MPI_Reduce_scatter, which Combines values and scatters the results.
     * @tparam DataType            Type of the input/output data. Either Eigen Matrix or Tensor.
     * @param  comm       [in]     Communicator.
     * @param  sendBuf    [in]     Starting address of send buffer.
     * @param  recvCounts [in]     Array specifying the number of elements in result distributed to each process. It must be identical on all calling processes.
     * @param  recvBuf    [in,out] Starting address of receive buffer.
     */
    template <typename DataType_>
    void reduce_scatter( MPI_Comm  const &comm,
                         DataType_ const &sendBuf,
                         int       const &recvCounts,
                         DataType_       &recvBuf    )
    {
			MPI_Reduce_scatter(sendBuf.data(), recvBuf.data(), &recvCounts, MPI_DOUBLE, MPI_SUM, comm);
    }

    /*
     * Wrapper for MPI_Cart_create, which makes a new communicator to which
     * Cartesian topology information has been attached.
     * @tparam _Size            Number of dimensions of Cartesian grid.
     * @param  dims    [in]     Array specifying the number of processes in each dimension.
     * @param  periods [in]     Logical array specifying whether the grid is periodic (1 = true) or not (0 = false) in each dimension.
     * @param  reorder [in]     Ranking may be reordered (1 = true) or not (0 = false).
     * @param  comm    [in,out] Communicator with new Cartesian topology.
     */
    template<std::size_t _Size>
    void Cart_Create( std::array<int,_Size> const &dims,
                      std::array<int,_Size> const &periods,
                      int                   const  reorder,
                      MPI_Comm                    &comm    )
    {
      MPI_Cart_create(MPI_COMM_WORLD, static_cast<int>(_Size), dims.data(), periods.data(), reorder, &comm);
    }

    /*
     * Wrapper for MPI_Cart_coords, which determines process coords in
     * Cartesian topology given rank in group.
     * @tparam _Size            Number of dimensions of Cartesian grid.
     * @param  comm    [in]     Communicator with Cartesian structure.
     * @param  rank    [in]     Rank of a process within group of comm.
     * @param  coords  [in,out] Array containing the Cartesian coordinates of specified process.
     */
    template<std::size_t _Size>
    void Cart_Coords( MPI_Comm               const &comm,
                      int                    const  rank,
                      std::array<int, _Size>       &coords)
    {
      MPI_Cart_coords(comm, rank, static_cast<int>(_Size), coords.data());
    }

    /*
     * Wrapper for MPI_Cart_sub, which partitions a communicator into subgroups,
     * and form lower-dimensional Cartesian subgrids.
     * @tparam _Size            Number of dimensions of Cartesian grid.
     * @param  comm    [in]     Communicator with Cartesian structure.
     * @param  rDims   [in]     The ith entry of rDims specifies whether the ith dimension is kept in the subgrid (1 = true) or is dropped (0 = false).
     * @param  subComm [in,out] Communicator containing the subgrid that includes the calling process.
     */
    template<std::size_t _Size>
    void Cart_Sub( MPI_Comm               const &comm,
                   std::array<int, _Size> const &rDims,
                   MPI_Comm                     &subComm )
    {
      MPI_Cart_sub(comm, rDims.data(), &subComm);
    }

    /*
     * Wrapper for MPI_Barrier, for synchronization between MPI processes
     * @param comm [in] Communicator.
     */
    void Barrier(MPI_Comm const &comm)
    {
      MPI_Barrier(comm);
    }

    /*
     * Wrapper for Comm_Free, which marks a communicator object for deallocation.
     * @param comm [in] Communicator.
     */
    void Comm_Free(MPI_Comm &comm)
    {
      MPI_Comm_free(&comm);
    }

    /*
     * Wrapper for MPI_Finalize, that checks whether MPI has been finalized.
     */
    void Finalize()
    {
      MPI_Finalize();
    }

  } // end namespace v2
  #endif // DOXYGEN_SHOULD_SKIP_THIS

} // end namespace partensor

#endif // PARTENSOR_PARALLEL_WRAPPER_HPP
