#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      16/07/2019
* @author    Technical University of Crete team:
*            Athanasios P. Liavas
*            Paris Karakasis
*            Christos Kolomvakis
*            John Papagiannakos
*            Siaminou Nina
* @author    Neurocom SA team:
*            Christos Tsalidis
*            Georgios Lourakis
*            George Lykoudis
*/
#endif // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      PARTENSOR_basic.hpp
* @details
* Containts the the most basic information for @c PARTENSOR project.
********************************************************************/

#ifndef PARTENSOR_BASIC_HPP
#define PARTENSOR_BASIC_HPP

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS 1

#if defined __GNUC__ && __GNUC__>=6
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wignored-attributes"
# pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#include "Config.hpp"

#include <mutex>

#if USE_MPI
#include "ParallelWrapper.hpp"
#endif /* USE_MPI */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "execution.hpp"
#include "Constants.hpp"
#include "Tensor.hpp"

namespace partensor
{
#if USE_MPI
  using MPI_Environment  = partensor::environment;
  using MPI_Communicator = partensor::communicator;
#endif /* USE_MPI */

  using Clock    = std::chrono::high_resolution_clock;  /**< Chrono type for measuring time. */
  using Duration = std::chrono::nanoseconds;            /**< Type for chrono and duration time. */ // mircoseconds??

  class Environment
  {
  public:
    Environment(int argc, char *argv[], char *envp[])
  #if USE_MPI
      : mMpiEnv(argc,argv,true)
  #endif /* USE_MPI */
    {
      (void) argc;
      (void) argv;
      (void) envp;

      // create an spdlog object with pattern:
      // %l       = The log level of the message (eg info, warn)
      // %Y-%m-%d = Year in 4 digits - Month 01-12 - Day of month 01-31 (eg 2019-09-19)
      // %r       = 12 hour clock (eg 02:55:02 pm)
      // %F       = Nanosecond part of the current second 000000000-999999999 (eg 256789123)
      // %n       = Logger's name (eg some logger name)
      // %v       = The actual text to log (eg "some user text")
      spdlog::set_pattern("<%l> : [%Y-%m-%d %r] [%F] [%n] %v");
      mLogger = spdlog::basic_logger_mt("Partensor", "../log/partensor.txt");
    }

    Environment(int argc, char *argv[]) : Environment(argc,argv,nullptr)
    { }

    Environment() : Environment(0, nullptr,nullptr)
    { }

    ~Environment()
    {
      mLogger->flush();
      // TODO spdlog::drop("Partensor");
    }

    static Environment *Partensor(int argc, char **argv, char **envp)
    {
      static std::mutex                   l_mutex;
      static std::unique_ptr<Environment> l_partensor(nullptr);

      if (!l_partensor)
      {
        std::lock_guard<std::mutex> lock(l_mutex);

        if (!l_partensor)
        {
          if (argc == 0)
            l_partensor.reset(new Environment());
          else if (envp == nullptr)
            l_partensor.reset(new Environment(argc,argv));
          else
            l_partensor.reset(new Environment(argc,argv,envp));
        }
      }

      return l_partensor.get();
    }

    std::shared_ptr<spdlog::logger> Logger()
    {
      return mLogger;
    }

#if USE_MPI
    MPI_Environment &MpiEnvironment()
    {
      return mMpiEnv;
    }

    MPI_Communicator &MpiCommunicator()
    {
      return mMpiCom;
    }

#endif /* USE_MPI */
  private:
    std::shared_ptr<spdlog::logger>  mLogger;

  #if USE_MPI
    MPI_Environment   mMpiEnv;
    MPI_Communicator  mMpiCom;
  #endif  /* USE_MPI */
  };

  inline Environment *Partensor(int argc=0, char **argv=nullptr, char **envp=nullptr)
  {
    return Environment::Partensor(argc,argv,envp);
  }

  inline void Init(int argc, char **argv, char **envp)
  {
    Partensor(argc,argv,envp);
  }

  inline void Init(int argc, char **argv)
  {
    Partensor(argc,argv);
  }

  inline void Init()
  {
    Partensor();
  }

  /**
   * @brief Default Values for CPD algorithm.
   * 
   * Contains default values for either constraints, error threshholders
   * or termination conditions parameters. These values can be changed
   * using @c Options struct and the appropriate @c cpd or @c cpdDimTree 
   * call.
   * 
   * @tparam Tensor_  Type(data type and order) of input Tensor.
   */
  template<typename Tensor_>
  struct DefaultValues {

    static std::size_t constexpr TnsSize = TensorTraits<Tensor_>::TnsSize;

    using DoubleArray = typename TensorTraits<Tensor_>::DoubleArray;
    using Constraints = typename TensorTraits<Tensor_>::Constraints;
    using IntArray    = typename TensorTraits<Tensor_>::IntArray;

    static Method      constexpr DefaultMethod = Method::als;                     /**< Default value for Method is als. */
    // static Constraint  constexpr DefaultConstraint = Constraint::unconstrained;   /**< Default value for Constraint is unconstrained. */
    static Constraint  constexpr DefaultConstraint = Constraint::nonnegativity;   /**< Default value for Constraint is nonnegativity. */
    static double      constexpr DefaultThresholdError = 1e-3;                    /**< Default value for cost function's threshold. */
    static double      constexpr DefaultNesterovTolerance = 1e-2;                 /**< Default value for Nesterov's tolerance. */
    static unsigned    constexpr DefaultMaxIter = 20;                             /**< Default value outer loop maximum iterations. */
    static Duration    constexpr DefaultMaxDuration = Duration(10000);            /**< Default value outer loop maximum duration. */
    static double      constexpr DefaultLambda = 0.1;                             /**< Default value for lambda. */
    static double      constexpr DefaultProcessorPerMode = 2;                     /**< Default value for number of processors per tensor mode. */
    static int         constexpr DefaultAccelerationCoefficient = 3;              /**< Default value for acceleration coefficient. */
    static int         constexpr DefaultAccelerationFail = 0;                     /**< Default value for acceleration fail. */
    static bool        constexpr DefaultAcceleration = true;                      /**< Default value for acceleration. */
    static bool        constexpr DefaultNormalization = true;                     /**< Default value for normalization. */
    
    static bool        constexpr DefaultWriteToFile = false;                      /**< Default value for write final factors to files. */
    
    static DoubleArray constexpr DefaultLambdas = []() constexpr -> auto {
      DoubleArray c{};
      for (auto &e : c) e = DefaultLambda;
      return c;
    } ();

    static Constraints constexpr DefaultConstraints = []() constexpr -> auto {
      Constraints c{};
      for (auto &e : c) e = DefaultConstraint;
      return c;
    } ();

    static IntArray constexpr DefaultProcessorsPerMode = []() constexpr -> auto {
      IntArray c{};
      for(auto &e : c) e = DefaultProcessorPerMode;
      return c;
    } ();

  };

  template<typename SparseTensor_>
  struct SparseDefaultValues {

    static std::size_t constexpr TnsSize = SparseTensorTraits<SparseTensor_>::TnsSize;

    using DoubleArray = typename SparseTensorTraits<SparseTensor_>::DoubleArray;
    using Constraints = typename SparseTensorTraits<SparseTensor_>::Constraints;
    using IntArray    = typename SparseTensorTraits<SparseTensor_>::IntArray;

    static Method      constexpr DefaultMethod = Method::als;                      /**< Default value for Method is als. */
    // static Constraint  constexpr DefaultConstraint = Constraint::unconstrained;   /**< Default value for Constraint is unconstrained. */
    static Constraint  constexpr DefaultConstraint = Constraint::nonnegativity;    /**< Default value for Constraint is nonnegativity. */
    static double      constexpr DefaultThresholdError = 1e-3;                     /**< Default value for cost function's threshold. */
    static double      constexpr DefaultNesterovTolerance = 1e-2;                  /**< Default value for Nesterov's tolerance. */
    static int         constexpr DefaultMaxNesterovIter   = 20;
    static unsigned    constexpr DefaultMaxIter = 20;                              /**< Default value outer loop maximum iterations. */
    static Duration    constexpr DefaultMaxDuration = Duration(10000);             /**< Default value outer loop maximum duration. */
    static double      constexpr DefaultC_stochastic_perc = 0.5;
    static double      constexpr DefaultLambda = 0.01;                             /**< Default value for lambda. */
    static double      constexpr DefaultProcessorPerMode = 2;                      /**< Default value for number of processors per tensor mode. */
    static int         constexpr DefaultAccelerationCoefficient = 3;               /**< Default value for acceleration coefficient. */
    static int         constexpr DefaultAccelerationFail = 0;                      /**< Default value for acceleration fail. */
    static bool        constexpr DefaultAcceleration = false;                      /**< Default value for acceleration. */
    static bool        constexpr DefaultAveraging = false;                         /**< Default value for averaging. */
    static bool        constexpr DefaultNormalization = false;                     /**< Default value for normalization. */
    static bool        constexpr DefaultInitializeFactors = false;                 /**< Default value for initializeFactors. */
    static bool        constexpr DefaultReadFactorsFromFile = false;               /**< Default value for initializeFactors. */
    
    static bool        constexpr DefaultWriteToFile = false;                       /**< Default value for write final factors to files. */

    static int        constexpr DefaultNonZeros     = 1000;                        /**< Default value for the number of non-zeros elements in a sparse matrix. */
    
    static DoubleArray constexpr DefaultLambdas = []() constexpr -> auto {
      DoubleArray c{};
      for (auto &e : c) e = DefaultLambda;
      return c;
    } ();

    static Constraints constexpr DefaultConstraints = []() constexpr -> auto {
      Constraints c{};
      for (auto &e : c) e = DefaultConstraint;
      return c;
    } ();

    static IntArray constexpr DefaultProcessorsPerMode = []() constexpr -> auto {
      IntArray c{};
      for(auto &e : c) e = DefaultProcessorPerMode;
      return c;
    } ();

  };

  /**
   * @brief Manage defaults parameters for CPD algorithm.
   * 
   * In case different parameters values need to be used, an @c Option object
   * must be created. After changing the default values, then this object can
   * be passed in the appropriate @c cpd operation.
   * 
   * @tparam Tensor_          Type(data type and order) of input Tensor.
   * @tparam ExecutionPolicy_ The policy that is used, either sequential or parallel
   *                          with mpi.
   */
  template < typename Tensor_,
             typename ExecutionPolicy_ = execution::sequenced_policy,  
             template <typename T> class DefaultValues_ = DefaultValues    >
  struct Options 
  {
      static constexpr std::size_t TnsSize = TensorTraits<Tensor_>::TnsSize;     /**< Tensor Order. */

      using DataType    = typename TensorTraits<Tensor_>::DataType;              /**< Tensor Data type. */
      using MatrixType  = typename TensorTraits<Tensor_>::MatrixType;            /**< @c Eigen Matrix with the same Data type with the Tensor. */
      using Constraints = typename TensorTraits<Tensor_>::Constraints;
      using DoubleArray = typename TensorTraits<Tensor_>::DoubleArray;
      using StringArray = std::array<std::string, TnsSize>; 

      Method        method;
      Constraints   constraints;
      double        threshold_error;
      double        nesterov_delta_1;
      double        nesterov_delta_2;
      DoubleArray   lambdas;
      unsigned      max_iter;
      Duration      max_duration;
      int           accel_coeff;
      int           accel_fail;    
      bool          acceleration;
      bool          normalization;
      bool          writeToFile;
      StringArray   final_factors_paths;

      Options() : method(DefaultValues_<Tensor_>::DefaultMethod),                        /**< Default value for Method is als.             */
                  constraints(DefaultValues_<Tensor_>::DefaultConstraints),              /**< Default value for Constraint is no negative. */
                  threshold_error(DefaultValues_<Tensor_>::DefaultThresholdError),       /**< Default value for cost function's threshold. */
                  nesterov_delta_1(DefaultValues_<Tensor_>::DefaultNesterovTolerance),   /**< Default value for Nesterov's tolerance.     */
                  nesterov_delta_2(DefaultValues_<Tensor_>::DefaultNesterovTolerance),   /**< Default value for Nesterov's tolerance.     */
                  lambdas(DefaultValues_<Tensor_>::DefaultLambdas),                      /**< Default value for lambda.                    */
                  max_iter(DefaultValues_<Tensor_>::DefaultMaxIter),                     /**< Default value outer loop maximum iterations. */
                  max_duration(DefaultValues_<Tensor_>::DefaultMaxDuration),             /**< Default value outer loop maximum duration.   */
                  accel_coeff(DefaultValues_<Tensor_>::DefaultAccelerationCoefficient),  /**< Default value for acceleration coefficient. */
                  accel_fail(DefaultValues_<Tensor_>::DefaultAccelerationFail),          /**< Default value for acceleration fail. */
                  acceleration(DefaultValues_<Tensor_>::DefaultAcceleration),            /**< Default value for acceleration. */
                  normalization(DefaultValues_<Tensor_>::DefaultNormalization),          /**< Default value for normalization. */
                  writeToFile(DefaultValues_<Tensor_>::DefaultWriteToFile)//,              /**< Default value for write final factors to files. */
                  // final_factors_paths(DefaultValues_<Tensor_>::DefaultFinalFactorsPaths) /**< Default value for path for factors at the end of the algorithm. */  
      { 
        for(std::size_t i=0; i<TnsSize; ++i)
        {
          final_factors_paths[i] = "final_" + std::to_string(i) + ".bin";
        }
      }
      Options(Options const &) = default;
      Options(Options      &&) = default;

      Options &operator=(Options const &) = default;
      Options &operator=(Options      &&) = default;
    };

  /*
      Sparse Options
  */
  template < std::size_t _TnsSize,
            typename ExecutionPolicy_ = execution::sequenced_policy,  
            template <typename T> class DefaultValues_ = SparseDefaultValues    >
  struct SparseOptions 
  {
      using SparseTenor = typename partensor::SparseTensor<_TnsSize>;
      static constexpr std::size_t TnsSize = SparseTensorTraits<SparseTenor>::TnsSize;          /**< Tensor Order. */

      using DataType         = typename SparseTensorTraits<SparseTenor>::DataType;              /**< Tensor Data type. */
      using MatrixType       = typename SparseTensorTraits<SparseTenor>::MatrixType;            /**< @c Eigen Matrix with the same Data type with the Tensor. */
      using Constraints      = typename SparseTensorTraits<SparseTenor>::Constraints;
      using DoubleArray      = typename SparseTensorTraits<SparseTenor>::DoubleArray;
      using SparseMatrixType = typename SparseTensorTraits<SparseTenor>::SparseMatrixType;
      using LongMatrixType   = typename SparseTensorTraits<SparseTenor>::LongMatrixType;
      using Dimensions       = typename SparseTensorTraits<SparseTenor>::Dimensions;
      using SparseTensor     = typename SparseTensorTraits<SparseTenor>::SparseTensor;
      using MatrixArray      = typename SparseTensorTraits<SparseTenor>::MatrixArray;
      using StringArray      = std::array<std::string, TnsSize>; 

      int                      rank;
      std::array<int, TnsSize> tnsDims;
      int                      nonZeros;
      bool                     initialized_factors;
      bool                     read_factors_from_file;
      MatrixArray              factorsInit;

      Method        method;
      Constraints   constraints;
      double        threshold_error;
      double        nesterov_delta_1;
      double        nesterov_delta_2;
      int           max_nesterov_iter;
      double        c_stochastic_perc;
      DoubleArray   lambdas;
      unsigned      max_iter;
      Duration      max_duration;
      int           accel_coeff;
      int           accel_fail;    
      bool          acceleration;
      bool          averaging;
      bool          normalization;
      bool          writeToFile;

      std::string   ratings_path;
      StringArray   initial_factors_paths;
      StringArray   final_factors_paths;

      SparseOptions() : initialized_factors(DefaultValues_<SparseTenor>::DefaultInitializeFactors),
                        read_factors_from_file(DefaultValues_<SparseTenor>::DefaultReadFactorsFromFile),
                        method(DefaultValues_<SparseTenor>::DefaultMethod),                        /**< Default value for Method is als.             */
                        constraints(DefaultValues_<SparseTenor>::DefaultConstraints),              /**< Default value for Constraint is no negative. */
                        threshold_error(DefaultValues_<SparseTenor>::DefaultThresholdError),       /**< Default value for cost function's threshold. */
                        nesterov_delta_1(DefaultValues_<SparseTenor>::DefaultNesterovTolerance),   /**< Default value for Nesterov's tolerance.     */
                        nesterov_delta_2(DefaultValues_<SparseTenor>::DefaultNesterovTolerance),   /**< Default value for Nesterov's tolerance.     */
                        max_nesterov_iter(DefaultValues_<SparseTenor>::DefaultMaxNesterovIter),
                        c_stochastic_perc(DefaultValues_<SparseTenor>::DefaultC_stochastic_perc),
                        lambdas(DefaultValues_<SparseTenor>::DefaultLambdas),                      /**< Default value for lambda.                    */
                        max_iter(DefaultValues_<SparseTenor>::DefaultMaxIter),                     /**< Default value outer loop maximum iterations. */
                        max_duration(DefaultValues_<SparseTenor>::DefaultMaxDuration),             /**< Default value outer loop maximum duration.   */
                        accel_coeff(DefaultValues_<SparseTenor>::DefaultAccelerationCoefficient),  /**< Default value for acceleration coefficient. */
                        accel_fail(DefaultValues_<SparseTenor>::DefaultAccelerationFail),          /**< Default value for acceleration fail. */
                        acceleration(DefaultValues_<SparseTenor>::DefaultAcceleration),            /**< Default value for acceleration. */
                        averaging(DefaultValues_<SparseTenor>::DefaultAveraging),                  /**< Default value for averaging. */
                        normalization(DefaultValues_<SparseTenor>::DefaultNormalization),          /**< Default value for normalization. */
                        writeToFile(DefaultValues_<SparseTenor>::DefaultWriteToFile)               /**< Default value for write final factors to files. */
                        // final_factors_paths(DefaultValues_<SparseTenor>::DefaultFinalFactorsPaths) /**< Default value for path for factors at the end of the algorithm. */  
      { 
        for(std::size_t i=0; i<TnsSize; ++i)
        {
          final_factors_paths[i] = "final_" + std::to_string(i) + ".bin";
        }
      }
      SparseOptions(SparseOptions const &) = default;
      SparseOptions(SparseOptions      &&) = default;

      SparseOptions &operator=(SparseOptions const &) = default;
      SparseOptions &operator=(SparseOptions      &&) = default;
    };

    template < typename Tensor_,
               typename ExecutionPolicy_ = execution::sequenced_policy,
               template <typename T> class DefaultValues_ = DefaultValues   >
    Options<Tensor_, ExecutionPolicy_, DefaultValues_> MakeOptions()
    {
      Options<Tensor_,ExecutionPolicy_,DefaultValues_> options;
      static constexpr std::size_t TnsSize = TensorTraits<Tensor_>::TnsSize;     /**< Tensor Order. */

      options.method              = DefaultValues_<Tensor_>::DefaultMethod;                  // Default value for Method is als.
      options.constraints         = DefaultValues_<Tensor_>::DefaultConstraints;             // Default value for Constraint is no negative.
      options.threshold_error     = DefaultValues_<Tensor_>::DefaultThresholdError;          // Default value for cost function's threshold.
      options.nesterov_delta_1    = DefaultValues_<Tensor_>::DefaultNesterovTolerance;       // Default value for Nesterov's tolerance.
      options.nesterov_delta_2    = DefaultValues_<Tensor_>::DefaultNesterovTolerance;       // Default value for Nesterov's tolerance.
      options.lambdas             = DefaultValues_<Tensor_>::DefaultLambdas;                 // Default value for lambda.
      options.max_iter            = DefaultValues_<Tensor_>::DefaultMaxIter;                 // Default value outer loop maximum iterations.
      options.max_duration        = DefaultValues_<Tensor_>::DefaultMaxDuration;             // Default value outer loop maximum duration.
      options.accel_coeff         = DefaultValues_<Tensor_>::DefaultAccelerationCoefficient; // Default value for acceleration coefficient.
      options.accel_fail          = DefaultValues_<Tensor_>::DefaultAccelerationFail;        // Default value for acceleration fail.
      options.acceleration        = DefaultValues_<Tensor_>::DefaultAcceleration;            // Default value for acceleration.
      // options.averaging           = DefaultValues_<Tensor_>::DefaultAveraging;               // Default value for averaging.
      options.normalization       = DefaultValues_<Tensor_>::DefaultNormalization;           // Default value for normalization.
      options.writeToFile         = DefaultValues_<Tensor_>::DefaultWriteToFile;             // Default value for write final factors to files.
      // options.final_factors_paths = DefaultValues_<Tensor_>::DefaultFinalFactorsPaths;       // Default value for path for factors at the end of the algorithm.
      for(std::size_t i=0; i<TnsSize; ++i)
      {
        options.final_factors_paths[i] = "final_" + std::to_string(i) + ".bin";
      }
      return options;
    }

    template < std::size_t _TnsSize,
               typename ExecutionPolicy_ = execution::sequenced_policy,
               template <typename T> class DefaultValues_ = SparseDefaultValues   >
    SparseOptions<_TnsSize, ExecutionPolicy_, DefaultValues_> MakeSparseOptions()
    {
      SparseOptions<_TnsSize,ExecutionPolicy_,DefaultValues_> options;
      using SparseTensor          = typename partensor::SparseTensor<_TnsSize>;
      static constexpr std::size_t TnsSize = SparseTensorTraits<SparseTensor>::TnsSize;     /**< Tensor Order. */

      options.method              = DefaultValues_<SparseTensor>::DefaultMethod;                  // Default value for Method is als.
      options.constraints         = DefaultValues_<SparseTensor>::DefaultConstraints;             // Default value for Constraint is no negative.
      options.threshold_error     = DefaultValues_<SparseTensor>::DefaultThresholdError;          // Default value for cost function's threshold.
      options.nesterov_delta_1    = DefaultValues_<SparseTensor>::DefaultNesterovTolerance;       // Default value for Nesterov's tolerance.
      options.nesterov_delta_2    = DefaultValues_<SparseTensor>::DefaultNesterovTolerance;       // Default value for Nesterov's tolerance.
      options.c_stochastic_perc   = DefaultValues_<SparseTensor>::DefaultC_stochastic_perc;
      options.lambdas             = DefaultValues_<SparseTensor>::DefaultLambdas;                 // Default value for lambda.
      options.max_iter            = DefaultValues_<SparseTensor>::DefaultMaxIter;                 // Default value outer loop maximum iterations.
      options.max_duration        = DefaultValues_<SparseTensor>::DefaultMaxDuration;             // Default value outer loop maximum duration.
      options.accel_coeff         = DefaultValues_<SparseTensor>::DefaultAccelerationCoefficient; // Default value for acceleration coefficient.
      options.accel_fail          = DefaultValues_<SparseTensor>::DefaultAccelerationFail;        // Default value for acceleration fail.
      options.acceleration        = DefaultValues_<SparseTensor>::DefaultAcceleration;            // Default value for acceleration.
      options.averaging           = DefaultValues_<SparseTensor>::DefaultAveraging;               // Default value for averaging.
      options.normalization       = DefaultValues_<SparseTensor>::DefaultNormalization;           // Default value for normalization.
      options.writeToFile         = DefaultValues_<SparseTensor>::DefaultWriteToFile;             // Default value for write final factors to files.
      // options.final_factors_paths = DefaultValues_<SparseTensor>::DefaultFinalFactorsPaths;       // Default value for path for factors at the end of the algorithm.
      for(std::size_t i=0; i<TnsSize; ++i)
      {
        options.final_factors_paths[i] = "final_" + std::to_string(i) + ".bin";
      }
      return options;
    }

    template < typename Tensor_,
               typename ExecutionPolicy_ = execution::sequenced_policy,
               template <typename T> class DefaultValues_ = DefaultValues     >
    Options<Tensor_,ExecutionPolicy_,DefaultValues_> MakeOptions(DefaultValues_<Tensor_> &&dv, ExecutionPolicy_ &&xp)
    {
      Options<Tensor_,ExecutionPolicy_,DefaultValues_> options;
      static constexpr std::size_t TnsSize = TensorTraits<Tensor_>::TnsSize;     /**< Tensor Order. */

      options.method              = DefaultValues_<Tensor_>::DefaultMethod;                  // Default value for Method is als.
      options.constraints         = DefaultValues_<Tensor_>::DefaultConstraints;             // Default value for Constraint is no negative.
      options.threshold_error     = DefaultValues_<Tensor_>::DefaultThresholdError;          // Default value for cost function's threshold.
      options.nesterov_delta_1    = DefaultValues_<Tensor_>::DefaultNesterovTolerance;       // Default value for Nesterov's tolerance.
      options.nesterov_delta_2    = DefaultValues_<Tensor_>::DefaultNesterovTolerance;       // Default value for Nesterov's tolerance.
      options.lambdas             = DefaultValues_<Tensor_>::DefaultLambdas;                 // Default value for lambda.
      options.max_iter            = DefaultValues_<Tensor_>::DefaultMaxIter;                 // Default value outer loop maximum iterations.
      options.max_duration        = DefaultValues_<Tensor_>::DefaultMaxDuration;             // Default value outer loop maximum duration.
      options.accel_coeff         = DefaultValues_<Tensor_>::DefaultAccelerationCoefficient; // Default value for acceleration coefficient.
      options.accel_fail          = DefaultValues_<Tensor_>::DefaultAccelerationFail;        // Default value for acceleration fail.
      options.acceleration        = DefaultValues_<Tensor_>::DefaultAcceleration;            // Default value for acceleration.
      // options.averaging           = DefaultValues_<Tensor_>::DefaultAveraging;               // Default value for averaging.
      options.normalization       = DefaultValues_<Tensor_>::DefaultNormalization;           // Default value for normalization.
      options.writeToFile         = DefaultValues_<Tensor_>::DefaultWriteToFile;             // Default value for write final factors to files.
      // options.final_factors_paths = DefaultValues_<Tensor_>::DefaultFinalFactorsPaths;       // Default value for path for factors at the end of the algorithm.
      for(std::size_t i=0; i<TnsSize; ++i)
      {
        options.final_factors_paths[i] = "final_" + std::to_string(i) + ".bin";
      }
      return options;
    }

    template < std::size_t _TnsSize,
               typename ExecutionPolicy_ = execution::sequenced_policy,
               template <typename T> class DefaultValues_ = SparseDefaultValues   >
    SparseOptions<_TnsSize, ExecutionPolicy_, DefaultValues_> MakeSparseOptions(DefaultValues_<partensor::SparseTensor<_TnsSize>> &&dv, ExecutionPolicy_ &&xp)
    {
      SparseOptions<_TnsSize,ExecutionPolicy_,DefaultValues_> options;
      using SparseTensor          = typename partensor::SparseTensor<_TnsSize>;
      static constexpr std::size_t TnsSize = SparseTensorTraits<SparseTensor>::TnsSize;     /**< Tensor Order. */

      options.method              = DefaultValues_<SparseTensor>::DefaultMethod;                  // Default value for Method is als.
      options.constraints         = DefaultValues_<SparseTensor>::DefaultConstraints;             // Default value for Constraint is no negative.
      options.threshold_error     = DefaultValues_<SparseTensor>::DefaultThresholdError;          // Default value for cost function's threshold.
      options.nesterov_delta_1    = DefaultValues_<SparseTensor>::DefaultNesterovTolerance;       // Default value for Nesterov's tolerance.
      options.nesterov_delta_2    = DefaultValues_<SparseTensor>::DefaultNesterovTolerance;       // Default value for Nesterov's tolerance.
      options.c_stochastic_perc   = DefaultValues_<SparseTensor>::DefaultC_stochastic_perc;
      options.lambdas             = DefaultValues_<SparseTensor>::DefaultLambdas;                 // Default value for lambda.
      options.max_iter            = DefaultValues_<SparseTensor>::DefaultMaxIter;                 // Default value outer loop maximum iterations.
      options.max_duration        = DefaultValues_<SparseTensor>::DefaultMaxDuration;             // Default value outer loop maximum duration.
      options.accel_coeff         = DefaultValues_<SparseTensor>::DefaultAccelerationCoefficient; // Default value for acceleration coefficient.
      options.accel_fail          = DefaultValues_<SparseTensor>::DefaultAccelerationFail;        // Default value for acceleration fail.
      options.acceleration        = DefaultValues_<SparseTensor>::DefaultAcceleration;            // Default value for acceleration.
      options.averaging           = DefaultValues_<SparseTensor>::DefaultAveraging;                    // Default value for averaging.      
      options.normalization       = DefaultValues_<SparseTensor>::DefaultNormalization;           // Default value for normalization.
      options.writeToFile         = DefaultValues_<SparseTensor>::DefaultWriteToFile;             // Default value for write final factors to files.
      // options.final_factors_paths = DefaultValues_<SparseTensor>::DefaultFinalFactorsPaths;       // Default value for path for factors at the end of the algorithm.
      for(std::size_t i=0; i<TnsSize; ++i)
      {
        options.final_factors_paths[i] = "final_" + std::to_string(i) + ".bin";
      }
      return options;
    }

    /**
     * @brief Returned Type of CPD algorithm.
     * 
     * @c Status is the returned type of @c cpd operations. In this struct 
     * exist the returned values, such as the @c cost @c function at the end of the
     * algorithm, or at what @c iteration the operation has ended. Also, includes  
     * the factors produced from @c cpd operation in an @c stl array of @c Matrix 
     * type and size same as the input Tensor order.
     * 
     * @tparam Tensor_          Type(data type and order) of input Tensor.
     * @tparam ExecutionPolicy  The policy that is used, either sequential or with mpi.
     */
    template < typename Tensor_,
               typename ExecutionPolicy_ = execution::sequenced_policy,
               template <typename T> class DefaultValues_ = DefaultValues  >
    struct Status
    {
      using MatrixArray  = typename TensorTraits<Tensor_>::MatrixArray; /**< @c Eigen Matrix with the same Data type with the Tensor. */

      Options<Tensor_,ExecutionPolicy_,DefaultValues_> options;
      double                                           frob_tns          = 0.0;         /**< Stores the Frobenius norm of an @c Eigen Tensor. */
      double                                           f_value           = 0.0;         /**< Stores the cost function. */
      double                                           rel_costFunction  = 0.0;         /**< Stores the relative cost function. */
      unsigned                                         ao_iter           = 0;           /**< Stores the iteration where the cost function reached the wanted threshold. */
      MatrixArray                                      factors;                         /**< An stl array with the resulting Factors from CPD factorization of the Eigen Tensor. */

      Status()               = default;
      Status(Status const &) = default;
      Status(Status      &&) = default;

      Status(Options<Tensor_,ExecutionPolicy_,DefaultValues_> const &opt) : options(opt)
      {  }

      Status &operator=(Status const &) = default;
      Status &operator=(Status      &&) = default;

      /*
      Constructor called, in case the user decides to change one or more 
      of the options and use them in the factorization.
      */
      Status &operator=(Options<Tensor_,ExecutionPolicy_,DefaultValues_> const &opts)
      {
      options = opts;

      return *this;
      }

      explicit operator Options<Tensor_,ExecutionPolicy_,DefaultValues_>& ()
      {
      return options;
      }
    };

    /**
     * Sparse Status
     */
    template < std::size_t _TnsSize,
               typename ExecutionPolicy_ = execution::sequenced_policy,
               template <typename T> class DefaultValues_ = SparseDefaultValues  >
    struct SparseStatus
    {
      using SparseTensor = typename partensor::SparseTensor<_TnsSize>;
      using MatrixArray  = typename SparseTensorTraits<SparseTensor>::MatrixArray; /**< @c Eigen Matrix with the same Data type with the Tensor. */

      SparseOptions<_TnsSize,ExecutionPolicy_,DefaultValues_> options;
      double                                                  frob_tns          = 0.0;         /**< Stores the Frobenius norm of an @c Eigen Tensor. */
      double                                                  f_value           = 0.0;         /**< Stores the cost function. */
      double                                                  rel_costFunction  = 0.0;         /**< Stores the relative cost function. */
      unsigned                                                ao_iter           = 0;           /**< Stores the iteration where the cost function reached the wanted threshold. */
      MatrixArray                                             factors;                         /**< An stl array with the resulting Factors from CPD factorization of the Eigen Tensor. */

      SparseStatus()               = default;
      SparseStatus(SparseStatus const &) = default;
      SparseStatus(SparseStatus      &&) = default;

      SparseStatus(SparseOptions<_TnsSize,ExecutionPolicy_,DefaultValues_> const &opt) : options(opt)
      {  }

      SparseStatus &operator=(SparseStatus const &) = default;
      SparseStatus &operator=(SparseStatus      &&) = default;

      /*
      Constructor called, in case the user decides to change one or more 
      of the options and use them in the factorization.
      */
      SparseStatus &operator=(SparseOptions<_TnsSize,ExecutionPolicy_,DefaultValues_> const &opts)
      {
        options = opts;

        return *this;
      }

      explicit operator SparseOptions<_TnsSize,ExecutionPolicy_,DefaultValues_>& ()
      {
        return options;
      }
    };

    template < typename Tensor_,
               typename ExecutionPolicy_ = execution::sequenced_policy,
               template <typename T> class DefaultValues_ = DefaultValues   >
    Status<Tensor_> MakeStatus()
    {
      Status<Tensor_,ExecutionPolicy_,DefaultValues_>  status;

      status.options = MakeOptions<Tensor_,ExecutionPolicy_,DefaultValues_>();

      return status;
    }

    template <typename Tensor_, typename ExecutionPolicy_, template <typename T> class DefaultValues_>
    Status<Tensor_,std::remove_cv_t<std::remove_reference_t<ExecutionPolicy_>>,DefaultValues_> MakeStatus(DefaultValues_<Tensor_> &&dv, ExecutionPolicy_ &&xp)
    {
      using ExecutionPolicy_t = std::remove_cv_t<std::remove_reference_t<ExecutionPolicy_>>;

      Status<Tensor_,ExecutionPolicy_t,DefaultValues_>  status;

      status.options = MakeOptions<Tensor_>(std::forward<ExecutionPolicy_>(xp));

      return status;
    }

    template < std::size_t _TnsSize,
               typename ExecutionPolicy_ = execution::sequenced_policy,
               template <typename T> class DefaultValues_ = SparseDefaultValues   >
    SparseStatus<_TnsSize> MakeSparseStatus()
    {
      SparseStatus<_TnsSize,ExecutionPolicy_,DefaultValues_>  status;

      status.options = MakeSparseOptions<_TnsSize,ExecutionPolicy_,DefaultValues_>();

      return status;
    }

    template < std::size_t _TnsSize,
               typename ExecutionPolicy_,
               template <typename T> class DefaultValues_   >
    SparseStatus<_TnsSize,std::remove_cv_t<std::remove_reference_t<ExecutionPolicy_>>,DefaultValues_> MakeSparseStatus(DefaultValues_<partensor::SparseTensor<_TnsSize>> &&dv, ExecutionPolicy_ &&xp)
    {      
      SparseStatus<_TnsSize,ExecutionPolicy_,DefaultValues_>  status;

      status.options = MakeSparseOptions<_TnsSize,ExecutionPolicy_,DefaultValues_>();

      return status;
    }

    template <typename Tensor_>
    struct Options<Tensor_,execution::openmpi_policy,DefaultValues> : public Options<Tensor_>
    {
      using Options<Tensor_,execution::sequenced_policy,DefaultValues>::constraints;

      using IntArray = typename TensorTraits<Tensor_>::IntArray;

      IntArray proc_per_mode;

      Options() : proc_per_mode(DefaultValues<Tensor_>::DefaultProcessorsPerMode)
      { }
      Options(Options const &) = default;
      Options(Options      &&) = default;

      Options &operator=(Options const &) = default;
      Options &operator=(Options      &&) = default;
    };

    template <typename Tensor_>
    Options<Tensor_,execution::openmpi_policy,DefaultValues> MakeOptions(execution::openmpi_policy &&)
    {
      Options<Tensor_,execution::openmpi_policy,DefaultValues>  options;

      static_cast<Options<Tensor_>&>(options) = MakeOptions<Tensor_>();

      options.proc_per_mode = DefaultValues<Tensor_>::DefaultProcessorsPerMode;

      return options;
    }

    template <std::size_t _TnsSize>
    struct SparseOptions<_TnsSize,execution::openmpi_policy,SparseDefaultValues> : public SparseOptions<_TnsSize>
    {
      using SparseOptions<_TnsSize,execution::sequenced_policy,SparseDefaultValues>::constraints;

      using SparseTensor = typename partensor::SparseTensor<_TnsSize>;
      using IntArray     = typename SparseTensorTraits<SparseTensor>::IntArray;

      IntArray proc_per_mode;

      SparseOptions() : proc_per_mode(SparseDefaultValues<SparseTensor>::DefaultProcessorsPerMode)
      { }
      SparseOptions(SparseOptions const &) = default;
      SparseOptions(SparseOptions      &&) = default;

      SparseOptions &operator=(SparseOptions const &) = default;
      SparseOptions &operator=(SparseOptions      &&) = default;
    };

    template <std::size_t _TnsSize>
    SparseOptions<_TnsSize,execution::openmpi_policy,SparseDefaultValues> MakeSparseOptions(execution::openmpi_policy &&)
    {
      using SparseTensor = typename partensor::SparseTensor<_TnsSize>;
      
      SparseOptions<_TnsSize,execution::openmpi_policy,SparseDefaultValues>  options;

      static_cast<SparseOptions<_TnsSize>&>(options) = MakeSparseOptions<_TnsSize>();

      options.proc_per_mode = SparseDefaultValues<SparseTensor>::DefaultProcessorsPerMode;

      return options;
    }

    template <typename Tensor_>
    using MpiOptions = Options<Tensor_,execution::openmpi_policy,DefaultValues>;

    template <typename Tensor_>
    using MpiStatus = Status<Tensor_,execution::openmpi_policy,DefaultValues>;

    template < std::size_t _TnsSize>
    using MpiSparseOptions = SparseOptions<_TnsSize,execution::openmpi_policy,SparseDefaultValues>;

    template < std::size_t _TnsSize>
    using MpiSparseStatus = SparseStatus<_TnsSize,execution::openmpi_policy,SparseDefaultValues>;

    template <typename Tensor_>
    using OmpOptions = Options<Tensor_,execution::openmp_policy,DefaultValues>;

    template <typename Tensor_>
    using OmpStatus = Status<Tensor_,execution::openmp_policy,DefaultValues>;

    template <typename Tensor_>
    using CudaOptions = Options<Tensor_, execution::cuda_policy, DefaultValues>;

    template <typename Tensor_>
    using CudaStatus = Status<Tensor_, execution::cuda_policy, DefaultValues>;

    template <std::size_t _TnsSize>
    using OmpSparseOptions = SparseOptions<_TnsSize,execution::openmp_policy,SparseDefaultValues>;

    template <std::size_t _TnsSize>
    using OmpSparseStatus = SparseStatus<_TnsSize,execution::openmp_policy,SparseDefaultValues>;

} // namespace partensor

namespace ptl = partensor;

#endif // PARTENSOR_BASIC_HPP
