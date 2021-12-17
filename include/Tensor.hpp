#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
* @date      21/03/2019
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
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file      Tensor.hpp
* @details
* Containts a variety of Types, used in @c partensor library.
* Implements different wrappers and traits for both @c Eigen Matrix  
* and @c Eigen Tensor modules.
********************************************************************/

#ifndef PARTENSOR_TYPES_HPP
#define PARTENSOR_TYPES_HPP

// #include "PARTENSOR_basic.hpp"
// #include "Constants.hpp"

namespace partensor {

  /**
   * Default Data Type for @c Eigen Data through project.
   */
  using DefaultDataType = double;

  /**
   * Alias for @c Eigen Matrix with rows and columns computed 
   * dynamically. Also the matrix is stored with column major.
   */
  using Matrix = Eigen::Matrix<DefaultDataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  /**
   * Alias for @c Eigen Matrix with rows and columns computed 
   * dynamically and the data are long int type. Also the matrix 
   * is stored with column major.
   */
  using LongMatrix = Eigen::Matrix<long int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  /**
   * Alias for @c Eigen Sparse Matrix with rows and columns computed 
   * dynamically. Also the matrix is stored with column major.
   */
	using SparseMatrix = Eigen::SparseMatrix<DefaultDataType, Eigen::ColMajor, long int>;

  /**
   * Alias for @c Eigen Tensor with data type equal to @c DefaultDataType
   * and tensor order equal to template parameter @c _TnsSize.
   * 
   * @tparam _TnsSize  Order of the Tensor
   */
  template<int _TnsSize>
  using Tensor = Eigen::Tensor<DefaultDataType, _TnsSize>;

  /**
   * Initialization of a templated struct with information about 
   * an @c Eigen Tensor.
   * 
   * @tparam Tensor @c Eigen Tensor Type.
   */
  template <typename Tensor>
  struct TensorTraits;

  /**
   * Specialization of templated struct @c TensorTraits.
   * 
   * @tparam _TnsSize  Tensor Order.
   */
  template <int _TnsSize>
  struct TensorTraits<Tensor<_TnsSize>>
  {
    using DataType    = DefaultDataType;
    using MatrixType  = partensor::Matrix;
    using Dimensions  = typename Tensor<_TnsSize>::Dimensions;
    using Constraints = std::array<Constraint,_TnsSize>;                      /**< Stl array of size TnsSize and containing Constraint type. */
    using MatrixArray = std::array<MatrixType,_TnsSize>;                      /**< Stl array of size TnsSize and containing MatrixType type. */
    using DoubleArray = std::array<double,_TnsSize>;                          /**< Stl array of size TnsSize and containing double type.     */
    using IntArray    = std::array<int,_TnsSize>;                             /**< Stl array of size TnsSize and containing int type. */

    static constexpr std::size_t TnsSize = _TnsSize;
  };
  
  template <typename Tensor>
  using MatrixArray = typename TensorTraits<Tensor>::MatrixArray;

  template <typename Tensor>
  using Constraints = typename TensorTraits<Tensor>::Constraints;

  template <typename Tensor>
  using DoubleArray = typename TensorTraits<Tensor>::DoubleArray;

  /**
   * Initialization of a templated struct with information about 
   * an @c Sparse @c Eigen.
   * 
   * @tparam @c Sparse @c Eigen Tensor Type.
   */
  template <typename SparseTensor>
  struct SparseTensorTraits;

  template <std::size_t _TnsSize>
  using SparseTensor = std::array<SparseMatrix, _TnsSize>;
  
  /**
   * Specialization of templated struct @c SparseMatrixTraits.
   * 
   * @tparam _TnsSize  Tensor Order.
   */
  template <std::size_t _TnsSize>
  struct SparseTensorTraits<SparseTensor<_TnsSize>> // std::array<SparseMatrix, _TnsSize>
  {
    using DataType          = DefaultDataType;
    using MatrixType        = partensor::Matrix;
    using SparseMatrixType  = partensor::SparseMatrix;
    using LongMatrixType    = partensor::LongMatrix;
    using Dimensions        = std::array<int,_TnsSize>;
    using Constraints       = std::array<Constraint,_TnsSize>;                      /**< Stl array of size TnsSize and containing Constraint type. */
    using SparseTensor      = std::array<SparseMatrix, _TnsSize>;
    using MatrixArray       = std::array<MatrixType,_TnsSize>;                      /**< Stl array of size TnsSize and containing MatrixType type. */
    using DoubleArray       = std::array<double,_TnsSize>;                          /**< Stl array of size TnsSize and containing double type.     */
    using IntArray          = std::array<int,_TnsSize>;                             /**< Stl array of size TnsSize and containing int type. */

    static constexpr std::size_t TnsSize = _TnsSize;
  };

  /**
   * Initialization of a templated struct. It is being used to hold
   * information about @c Eigen Matrix.
   * 
   * @tparam Matrix  @c Eigen Matrix Type.
   */
  template <typename Matrix>
  struct MatrixTraits;

  /**
   * Specialization of templated struct @c MatrixTraits.
   */
  template <>
  // struct MatrixTraits<Eigen::Matrix<DefaultDataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
  struct MatrixTraits<Matrix>
  {
    using DataType   = DefaultDataType;
    using MatrixType = partensor::Matrix;
  };

  /**
   * Initialization of a templated struct. It is being used to hold 
   * information about a container of @c Eigen Matrix type.
   * 
   * @tparam MA  An array container type.
   */
  template <typename MA>
  struct MatrixArrayTraits
  {
    using value_type = typename MA::value_type;
    using array_type = MA;

    static constexpr unsigned Size = std::tuple_size<MA>::value;
  };

  /**
   * Specialization of templated struct @c MatrixArrayTraits, for  
   * @c stl array.
   * 
   * @tparam T      Type of @c Eigen Data.
   * @tparam _Size  Size of the @c stl @c array.
   */
  template <typename T, std::size_t _Size>
  struct MatrixArrayTraits<std::array<T,_Size>>
  {
    using value_type = T;
    using array_type = std::array<T,_Size>;

    static constexpr std::size_t Size = _Size;
  };

  // template<typename MatrixArray>
  // using MatrixArrayType = typename MatrixArrayTraits<MatrixArray>::value_type;

  /**
   * Checks if input type @c T is equal to @c Matrix type.
   * 
   * @tparam T   Type of data to process.
   * 
   * @returns @c true if the data type is @c Matrix, otherwise returns @c false.
   */
	template<typename T>
	inline constexpr bool is_matrix = std::is_same<Matrix, typename std::decay<T>::type>::value;

  /**
   * Templated struct, which contains information about
   * a factor. It is used in factorization algorithms.
   * 
   * @tparam FactorType Either @c FactorDimTree or 2-dimension
   *                    @c Tensor.
   * 
   */
  template<typename FactorType>
  struct Factor
  {
     FactorType    factor;
     FactorType    gramian;
     Constraint    constraint;
  };

} // end namespace partensor

#endif // end of PARTENSOR_TYPES_HPP
