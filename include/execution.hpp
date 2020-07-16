#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      30/03/2019
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
* @file      execution.hpp
* @details
* execution namespace defines the execution policies that
* partensor library implements.
********************************************************************/

#ifndef PARTENSOR_EXECUTION_HPP
#define PARTENSOR_EXECUTION_HPP

#include <type_traits>

namespace partensor::execution {
inline namespace v1 {

class sequenced_policy
{
public:
  // For internal use only
  static constexpr std::false_type __allow_unsequenced()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_vector()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_parallel()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_opencl()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_openmpi()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_openmp()
  {
    return std::false_type{};
  }
};

class parallel_policy
{
public:
  // For internal use only
  static constexpr std::false_type __allow_unsequenced()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_vector()
  {
    return std::false_type{};
  }

  static constexpr std::true_type __allow_parallel()
  {
    return std::true_type{};
  }

  static constexpr std::false_type __allow_opencl()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_openmpi()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_openmp()
  {
    return std::false_type{};
  }
};

class parallel_unsequenced_policy
{
public:
  // For internal use only
  static constexpr std::true_type __allow_unsequenced()
  {
      return std::true_type{};
  }

  static constexpr std::true_type __allow_vector()
  {
    return std::true_type{};
  }

  static constexpr std::true_type __allow_parallel()
  {
    return std::true_type{};
  }

  static constexpr std::false_type __allow_opencl()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_openmpi()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_openmp()
  {
    return std::false_type{};
  }
};

class unsequenced_policy
{
public:
  // For internal use only
  static constexpr std::true_type __allow_unsequenced()
  {
    return std::true_type{};
  }
  static constexpr std::true_type __allow_vector()
  {
    return std::true_type{};
  }
  static constexpr std::false_type __allow_parallel()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_opencl()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_openmpi()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_openmp()
  {
    return std::false_type{};
  }
};

class opencl_policy
{
public:
  // For internal use only
  static constexpr std::false_type __allow_unsequenced()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_vector()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_parallel()
  {
    return std::false_type{};
  }

  static constexpr std::true_type __allow_opencl()
  {
    return std::true_type{};
  }

  static constexpr std::false_type __allow_openmpi()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_openmp()
  {
    return std::false_type{};
  }
};

class openmpi_policy
{
public:
  // For internal use only
  static constexpr std::false_type __allow_unsequenced()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_vector()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_parallel()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_opencl()
  {
    return std::false_type{};
  }

  static constexpr std::true_type __allow_openmpi()
  {
    return std::true_type{};
  }

  static constexpr std::false_type __allow_openmp()
  {
    return std::false_type{};
  }
};

class openmp_policy
{
public:
  // For internal use only
  static constexpr std::false_type __allow_unsequenced()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_vector()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_parallel()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_opencl()
  {
    return std::false_type{};
  }

  static constexpr std::false_type __allow_openmpi()
  {
    return std::false_type{};
  }

  static constexpr std::true_type __allow_openmp()
  {
    return std::true_type{};
  }
};

constexpr sequenced_policy            seq{};
constexpr parallel_policy             par{};
constexpr parallel_unsequenced_policy par_unseq{};
constexpr unsequenced_policy          unseq{};
constexpr opencl_policy               ocl{};
constexpr openmpi_policy              mpi{};
constexpr openmp_policy               omp{};

template <class T>
struct is_execution_policy : std::false_type
{ };

template <>
struct is_execution_policy<sequenced_policy> : std::true_type
{  };

template <>
struct is_execution_policy<parallel_policy> : std::true_type
{ };

template <>
struct is_execution_policy<parallel_unsequenced_policy> : std::true_type
{ };

template <>
struct is_execution_policy<unsequenced_policy> : std::true_type
{ };

template <>
struct is_execution_policy<opencl_policy> : std::true_type
{ };

template <>
struct is_execution_policy<openmpi_policy> : std::true_type
{ };

template <>
struct is_execution_policy<openmp_policy> : std::true_type
{ };

template <class T>
constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

template <typename P>
using execution_policy_t = std::remove_cv_t<std::remove_reference_t<P>>;

} // v1

namespace internal
{
template <class ExecPolicy, class T>
using enable_if_execution_policy = typename std::enable_if<partensor::execution::is_execution_policy<typename std::decay<ExecPolicy>::type>::value,T>::type;
} // namespace internal

} // partensor::execution

#endif //PARTENSOR_EXECUTION_HPP
