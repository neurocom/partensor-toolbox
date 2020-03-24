#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
 * @date      20/03/2019
 * @author    Christos Tsalidis
 * @author    Yiorgos Lourakis
 * @author    George Lykoudis
 * @copyright 2019 Neurocom All Rights Reserved.
 */
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file Matricization.hpp
* @details
* Implements the Tensor Matricization operation. 
*
* @warning The Tensor Order must be in @c [3,8]. 
* 
* Possible examples with Matrices of @c Matrix type from @c Tensor.hpp, 
* are the following.
*
*  - If Tensor order is 3, with Tensor @c tnsX(IxJxK),
*  @returns A @c Matrix, but the size depends on the matricization
*           @c mode. If it is on the first mode then the Matrix size is 
*           @c (IxJK), if it happened on the second, its size is 
*           @c (JxIK), and if the matricization happened on the 3rd mode 
*           then it has size @c (KxIJ).
*
*  - If Tensor order is 4, with Tensor @c tnsX(IxJxKxL),
*  @returns A @c Matrix, but the size depends on the matricization
*           @c mode. If it is on the first mode then the Matrix size is 
*           @c (IxJKL), if it happened on the second, its size is
*           @c (JxIKL), if the matricization happened on the 3rd mode
*           then it has size  @c (KxIJL), and in case of the 4th mode
*           it has size @c (LxIJK).
********************************************************************/

#ifndef PARTENSOR_MATRICIZATION_HPP
#define PARTENSOR_MATRICIZATION_HPP

#include "PARTENSOR_basic.hpp"
#include "TensorOperations.hpp"

namespace partensor {

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * @brief Necessary Permutation for Tensor matricization.
	 * 
	 * Compute the permutation array needed for shuffling Tensor Data based on the 
	 * mode the @c matricization will be applied.
	 *  
	 * @tparam _TnsSize  Tensor Order.
	 *
	 * @param  mode [in] Tensor dimesion which the @c matricization will be applied.
	 *
	 * @returns An @c Eigen array with size equal to Tensor order and with its dimesions
	 *          permuted based on the mode chosen.
	 */
	template<std::size_t _TnsSize>
	Eigen::array<int,_TnsSize> permute(std::size_t mode)
	{
		Eigen::array<int, _TnsSize> permutation;
		std::iota(permutation.begin(), permutation.end(), 0); // {0, 1, 2, ..., N}
	    permutation[0] = mode;      						  // {n, 0, 1, ..., n-1, n+1, ... , N}
	    for(std::size_t i=1; i <= mode; i++) {
	    	permutation[i] = i - 1;
	    }
	    return permutation;
	}
	#endif // DOXYGEN_SHOULD_SKIP_THIS

	inline namespace v1 {
	
		/**
		 *
		 * @brief Implementation of matricization operation over a Tensor.
		 * 
		 * Takes as input a Tensor with order equal to @c _TnsSize, and performs
		 * a matricization, more specifically a shuffling of the data, based on a 
		 * @c mode-dimension.
		 *  
		 * @tparam _TnsSize     Tensor Order of @c tnsX.
		 * @param  tnsX    [in] The Tensor to be matricized.
		 * @param  mode    [in] The dimension where the matricization will be performed.
		 *                      If @c mode=0, then the matricization will be performed in
		 *                      rows dimensions.
		 * 
		 * @returns A @c Matrix with the @c tnsX data permuted based on @c mode.
		 * 
		 * @warning @c mode variable takes values from range @c [0,_TnsSize-1].
		 *
         * @warning The result column dimension of the matricized tensor must have value
         * up until @c LONG_MAX.
		 */
		template<int _TnsSize>
		Matrix Matricization( Tensor<_TnsSize> const &tnsX, 
							  std::size_t      const  mode )
		{
			using Dimensions 		  = typename Tensor<_TnsSize>::Dimensions; // Type of @c Eigen Tensor Dimensions.
			const Dimensions& tnsDims = tnsX.dimensions();                     // Eigen Array with the lengths of each of Tensor Dimension.

			Tensor<2> matricedTns; // Temporary @c Eigen Tensor in order to keep the matricized Tensor. 
			auto      permutation = partensor::permute<_TnsSize>(mode);
			// Compute the column dimension for the matricized tensor.
			long int newColDim = 1;
			for(int i=0; i<_TnsSize; ++i)
			{
				if(i!=static_cast<int>(mode))
					newColDim *= tnsDims[i];
			}

			// reshape: View of the input tensor that has been reshaped to the specified new dimensions.
			// shuffle: A copy of the input tensor whose dimensions have been reordered according to the specified permutation.
			Eigen::array<long int, 2> reshape({static_cast<long int>(tnsDims[mode]), newColDim});
			matricedTns = tnsX.shuffle(permutation).reshape(reshape);
			// Map the @c Eigen Tensor to @c Eigen Matrix type.
			return tensorToMatrix(matricedTns, tnsDims[mode], newColDim);
		}
		
	}  // end namespace v1 

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	namespace experimental {

		inline namespace v1 {

			/**
			 * Computes the Matricization of an Tensor based on a 
			 * specified @c mode. 
			 * This implementation is a recursive function. Makes use of
			 * Tensor module Operations such as @c shuffle and @c reshape only 
			 * on cases of cubes (3D Tensors).
			 * For example, a 4D tensor, with dimension sizes 3x4x3x2, in 4th
			 * dimension has 2 cubes. The @c Matricization is applied on each
			 * cube, then saved in the a partition of the returned @c Matrix
			 * and then the final result is returned.
			 * 
			 * @tparam _TnsSize     Tensor Order.
			 * @param  tnsX    [in] The Tensor to be matricized.
			 * @param  mode    [in] Mode of @c Matricization. Takes values from [0, @c _TnsSize - 1].
			 * 
			 * @returns A @c Matrix with the @c tnsX data permuted based on @c mode.
			 */
			template<int _TnsSize>
			Matrix Matricization( Tensor<_TnsSize> const &tnsX, 
			                      std::size_t      const  mode )
			{
				using Dimensions          = typename Tensor<_TnsSize>::Dimensions; // Type of @c Eigen Tensor Dimensions.
				const Dimensions& tnsDims = tnsX.dimensions();                     // Eigen Array with the lengths of each of Tensor Dimension.

				Tensor<2> matricedTns; // Temporary @c Eigen Tensor in order to keep the matricized Tensor. 
				auto      permutation = partensor::permute<_TnsSize>(mode);

				// Compute the column dimension for the matricized tensor.
				long int newColDim = 1;
				for(int i=0; i<_TnsSize; ++i)
				{
					if(i!=static_cast<int>(mode))
						newColDim *= tnsDims[i];
				}

				if constexpr (_TnsSize<4)
				{
					// reshape: View of the input tensor that has been reshaped to the specified new dimensions.
					// shuffle: A copy of the input tensor whose dimensions have been reordered according to the specified permutation.
					Eigen::array<long int, 2> newshape({tnsDims[mode], newColDim});
					matricedTns = tnsX.shuffle(permutation).reshape(newshape);

	        		// Map the @c Eigen Tensor to @c Eigen Matrix type.
					return tensorToMatrix(matricedTns, tnsDims[mode], newColDim);
				}
				else
				{
					long int matricized_dim = 1;
					for(std::size_t i=1; i < permutation.size()-1; ++i)
					{
						matricized_dim *= tnsDims[permutation[i]];
					}
					Tensor<_TnsSize-1> cubeTns;
					Matrix             matricedTns(tnsDims[mode], newColDim);
					// Recursive Call for each cube, based on the mode if it is on the last dimension or not.
					if(mode<_TnsSize-1)
					{
						for(int cubeId=0; cubeId<tnsDims[_TnsSize-1]; cubeId++)
						{
							// Get a _TnsSize-1 order hypercube from whole tensor.
							cubeTns = tnsX.chip(cubeId,_TnsSize-1);
							// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
							matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization(cubeTns, mode);
						}
					}
					else
					{
						for(int cubeId=0; cubeId<tnsDims[_TnsSize-2]; cubeId++)
						{
							// Get a _TnsSize-1 order hypercube from whole tensor.
							cubeTns = tnsX.chip(cubeId,_TnsSize-2);
							// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
							matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization(cubeTns, _TnsSize-2);
						}
					}
					return matricedTns;
				}

	    	}

			template<int _TnsSize>
			Matrix Matricization_3( Tensor<_TnsSize> const &tnsX, 
									std::size_t      const  mode )
			{
				using Dimensions          = typename Tensor<_TnsSize>::Dimensions;
				const Dimensions& tnsDims = tnsX.dimensions();
				Tensor<2> matricedTns; // Temporary @c Eigen Tensor in order to keep the matricized Tensor. 
				auto      permutation = partensor::permute<_TnsSize>(mode);

				long int newColDim = 1;
				for(int i=0; i<_TnsSize; ++i)
				{
					if(i!=static_cast<int>(mode))
						newColDim *= tnsDims[i];
				}
				Eigen::array<long int, 2> newshape({tnsDims[mode], newColDim});
				matricedTns = tnsX.shuffle(permutation).reshape(newshape);

				// Map the @c Eigen Tensor to @c Eigen Matrix type.
				return tensorToMatrix(matricedTns, tnsDims[mode], newColDim);
			}

			template<int _TnsSize>
			Matrix Matricization_4( Tensor<_TnsSize> const &tnsX, 
									std::size_t      const  mode )
			{
				using Dimensions 		  = typename Tensor<_TnsSize>::Dimensions;
				const Dimensions& tnsDims = tnsX.dimensions();

				Tensor<_TnsSize-1> cubeTns; // Temporary @c Eigen Tensor in order to keep the matricized Tensor. 
				auto               permutation = partensor::permute<_TnsSize>(mode);

				long int newColDim = 1;
				for(int i=0; i<_TnsSize; ++i)
				{
					if(i!=static_cast<int>(mode))
						newColDim *= tnsDims[i];
				}
				long int matricized_dim = 1;
				for(std::size_t i=1; i < permutation.size()-1; ++i)
				{
					matricized_dim *= tnsDims[permutation[i]];
				}
				Matrix matricedTns(tnsDims[mode], newColDim);
				// Recursive Call for each cube, based on the mode if it is on the last dimension or not.
				if(mode<_TnsSize-1)
				{
					for(int cubeId=0; cubeId<tnsDims[_TnsSize-1]; cubeId++)
					{
						// Get a _TnsSize-1 order hypercube from whole tensor.
						cubeTns = tnsX.chip(cubeId,_TnsSize-1);
						// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
						matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_3(cubeTns, mode);
					}
				}
				else
				{
					for(int cubeId=0; cubeId<tnsDims[_TnsSize-2]; cubeId++)
					{
						// Get a _TnsSize-1 order hypercube from whole tensor.
						cubeTns = tnsX.chip(cubeId,_TnsSize-2);
						// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
						matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_3(cubeTns, _TnsSize-2);
					}
				}
				return matricedTns;
	    	}

			template<int _TnsSize>
			Matrix Matricization_5( Tensor<_TnsSize> const &tnsX, 
									std::size_t      const  mode )
			{
				using Dimensions          = typename Tensor<_TnsSize>::Dimensions;
				const Dimensions& tnsDims = tnsX.dimensions();

				Tensor<_TnsSize-1> cubeTns;
				auto               permutation = partensor::permute<_TnsSize>(mode);

				long int newColDim = std::accumulate(tnsDims.begin(), tnsDims.end(), 1, std::multiplies<long int>());
				newColDim 		  /= tnsDims[mode];

				long int matricized_dim = 1;
				for(std::size_t i=1; i < permutation.size()-1; ++i)
				{
					matricized_dim *= tnsDims[permutation[i]];
				}
				Matrix matricedTns(tnsDims[mode], newColDim);
				// Recursive Call for each cube, based on the mode if it is on the last dimension or not.
				if(mode<_TnsSize-1)
				{
					for(int cubeId=0; cubeId<tnsDims[_TnsSize-1]; cubeId++)
					{
						// Get a _TnsSize-1 order hypercube from whole tensor.
						cubeTns = tnsX.chip(cubeId,_TnsSize-1);
						// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
						matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_4(cubeTns, mode);
					}
				}
				else
				{
					for(int cubeId=0; cubeId<tnsDims[_TnsSize-2]; cubeId++)
					{
						// Get a _TnsSize-1 order hypercube from whole tensor.
						cubeTns = tnsX.chip(cubeId,_TnsSize-2);
						// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
						matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_4(cubeTns, _TnsSize-2);
					}
				}
				return matricedTns;
	    	}

			template<int _TnsSize>
			Matrix Matricization_6( Tensor<_TnsSize> const &tnsX, 
									std::size_t      const  mode )
			{
				using Dimensions          = typename Tensor<_TnsSize>::Dimensions;
				const Dimensions& tnsDims = tnsX.dimensions();

				Tensor<_TnsSize-1> cubeTns;
				auto               permutation = partensor::permute<_TnsSize>(mode);

				long int newColDim = std::accumulate(tnsDims.begin(), tnsDims.end(), 1, std::multiplies<long int>());
				newColDim         /= tnsDims[mode];

				long int matricized_dim = 1;
				for(std::size_t i=1; i < permutation.size()-1; ++i)
				{
					matricized_dim *= tnsDims[permutation[i]];
				}
				Matrix matricedTns(tnsDims[mode], newColDim);
				// Recursive Call for each cube, based on the mode if it is on the last dimension or not.
				if(mode<_TnsSize-1)
				{
					for(int cubeId=0; cubeId<tnsDims[_TnsSize-1]; cubeId++)
					{
						// Get a _TnsSize-1 order hypercube from whole tensor.
						cubeTns = tnsX.chip(cubeId,_TnsSize-1);
						// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
						matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_5(cubeTns, mode);
					}
				}
				else
				{
					for(int cubeId=0; cubeId<tnsDims[_TnsSize-2]; cubeId++)
					{
						// Get a _TnsSize-1 order hypercube from whole tensor.
						cubeTns = tnsX.chip(cubeId,_TnsSize-2);
						// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
						matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_5(cubeTns, _TnsSize-2);
					}
				}
				return matricedTns;
	    	}

			template<int _TnsSize>
			Matrix Matricization_7( Tensor<_TnsSize> const &tnsX, 
									std::size_t      const  mode )
			{
				using Dimensions          = typename Tensor<_TnsSize>::Dimensions;
				const Dimensions& tnsDims = tnsX.dimensions();

				Tensor<_TnsSize-1> cubeTns;
				auto               permutation = partensor::permute<_TnsSize>(mode);

				long int newColDim = std::accumulate(tnsDims.begin(), tnsDims.end(), 1, std::multiplies<long int>());
				newColDim 		  /= tnsDims[mode];

				long int matricized_dim = 1;
				for(std::size_t i=1; i < permutation.size()-1; ++i)
				{
					matricized_dim *= tnsDims[permutation[i]];
				}
				Matrix matricedTns(tnsDims[mode], newColDim);
				// Recursive Call for each cube, based on the mode if it is on the last dimension or not.
				if(mode<_TnsSize-1)
				{
					for(int cubeId=0; cubeId<tnsDims[_TnsSize-1]; cubeId++)
					{
						// Get a _TnsSize-1 order hypercube from whole tensor.
						cubeTns = tnsX.chip(cubeId,_TnsSize-1);
						// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
						matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_6(cubeTns, mode);
					}
				}
				else
				{
					for(int cubeId=0; cubeId<tnsDims[_TnsSize-2]; cubeId++)
					{
						// Get a _TnsSize-1 order hypercube from whole tensor.
						cubeTns = tnsX.chip(cubeId,_TnsSize-2);
						// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
						matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_6(cubeTns, _TnsSize-2);
					}
				}
				return matricedTns;
	    	}

			template<int _TnsSize>
			Matrix Matricization_8( Tensor<_TnsSize> const &tnsX, 
									std::size_t      const  mode )
			{
				using Dimensions          = typename Tensor<_TnsSize>::Dimensions;
				const Dimensions& tnsDims = tnsX.dimensions();

				Tensor<_TnsSize-1> cubeTns;
				auto               permutation = partensor::permute<_TnsSize>(mode);

				long int newColDim = std::accumulate(tnsDims.begin(), tnsDims.end(), 1, std::multiplies<long int>());
				newColDim 		  /= tnsDims[mode];

				long int matricized_dim = 1;
				for(std::size_t i=1; i < permutation.size()-1; ++i)
				{
					matricized_dim *= tnsDims[permutation[i]];
				}
				Matrix matricedTns(tnsDims[mode], newColDim);
				// Recursive Call for each cube, based on the mode if it is on the last dimension or not.
				if(mode<_TnsSize-1)
				{
					for(int cubeId=0; cubeId<tnsDims[_TnsSize-1]; cubeId++)
					{
						// Get a _TnsSize-1 order hypercube from whole tensor.
						cubeTns = tnsX.chip(cubeId,_TnsSize-1);
						// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
						matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_7(cubeTns, mode);
					}
				}
				else
				{
					for(int cubeId=0; cubeId<tnsDims[_TnsSize-2]; cubeId++)
					{
						// Get a _TnsSize-1 order hypercube from whole tensor.
						cubeTns = tnsX.chip(cubeId,_TnsSize-2);
						// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
						matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_7(cubeTns, _TnsSize-2);
					}
				}
				return matricedTns;
	    	}

		} // end namespace v1

		namespace v2 {

			/**
			 * Computes the Matricization of a Tensor based on a specified @c mode.
			 * 
			 * @tparam Tensor_  Type(data type and order) of input Tensor.
			 * @param  tnsX     [in] The Tensor to be matricized.
			 * @param  mode     [in] Mode of @c Matricization. Takes values from [0, @c TnsSize - 1].
			 * 
			 * @returns A @c Matrix with the @c tnsX data permuted based on @c mode.
			 */
			template<typename Tensor_>
			typename TensorTraits<Tensor_>::MatrixType Matricization( Tensor_    const &tnsX, 
																	  std::size_t const  mode )
			{
				// using DataType    = typename TensorTraits<Tensor_>::DataType;    // Type of @c Eigen Tensor Data (e.g double, float, etc.).
				constexpr std::size_t TnsSize = TensorTraits<Tensor_>::TnsSize;
			  	using Dimensions              = typename TensorTraits<Tensor_>::Dimensions;  // Type of @c Eigen Tensor Dimensions.

				const Dimensions& tnsDims = tnsX.dimensions(); // Eigen Array with the lengths of each of Tensor Dimension.
			  	Tensor<2>         matricedTns;                 // Temporary @c Eigen Tensor in order to keep the matricized Tensor. 
				auto              permutation = partensor::permute<TnsSize>(mode);
			  	
				// Compute the column dimension for the matricized tensor.
				long int newColDim = 1;
				for(std::size_t i=0; i<TnsSize; ++i)
				{
					if(i!=mode)
						newColDim *= tnsDims[i];
				}

				// reshape: View of the input tensor that has been reshaped to the specified new dimensions.
				// shuffle: A copy of the input tensor whose dimensions have been reordered according to the specified permutation.
				Eigen::array<long int, 2> reshape({tnsDims[mode], newColDim});
				matricedTns = tnsX.shuffle(permutation).reshape(reshape);
				// Map the @c Eigen Tensor to @c Eigen Matrix type.
				return tensorToMatrix(matricedTns, tnsDims[mode], newColDim);
			}

			/**
			 * Computes the Matricization of an Tensor based on a 
			 * specified @c mode.
			 * 
			 * This implementation is a recursive function. Makes use of
			 * Tensor module Operations such as @c shuffle and @c reshape only 
			 * on cases of cubes (3D Tensors).
			 * For example, a 4D tensor, with dimension sizes 3x4x3x2, in 4th
			 * dimension has 2 cubes. The @c Matricization is applied on each
			 * cube, then saved in the a partition of the returned @c Matrix
			 * and then the final result is returned.
			 * 
			 * @tparam Tensor_   Type(data type and order) of input Tensor.
			 * @param  tnsX      [in] The Tensor to be matricized.
			 * @param  mode      [in] Mode of @c Matricization. Takes values from [0, @c TnsSize - 1].
			 * 
			 * @returns A @c Matrix with the @c tnsX data permuted based on @c mode.
			 */
			template<typename Tensor_>
			typename TensorTraits<Tensor_>::MatrixType Matricization_rec( Tensor_    const &tnsX, 
						  												  std::size_t const  mode )
			{
				// using DataType    = typename TensorTraits<Tensor_>::DataType;    	 // Type of @c Eigen Tensor Data (e.g double, float, etc.).
		    	using MatrixType  = typename TensorTraits<Tensor_>::MatrixType;  		 // Type of @c Eigen Matrix based on Tensor Type.
				using Dimensions  = typename TensorTraits<Tensor_>::Dimensions;  		 // Type of @c Eigen Tensor Dimensions.

				static constexpr std::size_t TnsSize = TensorTraits<Tensor_>::TnsSize;   // Tensor Order.

				const Dimensions&  tnsDims     = tnsX.dimensions(); // Eigen Array with the lengths of each of Tensor Dimension.
				Tensor<2>          matricedTns;                     // Temporary @c Eigen Tensor in order to keep the matricized Tensor. 
				auto               permutation = partensor::permute<TnsSize>(mode);

				// Compute the column dimension for the matricized tensor.
				long int newColDim = std::accumulate(tnsDims.begin(), tnsDims.end(), 1, std::multiplies<long int>());
				newColDim 		  /= tnsDims[mode];

				if constexpr (TnsSize<4)
				{
					// reshape: View of the input tensor that has been reshaped to the specified new dimensions.
					// shuffle: A copy of the input tensor whose dimensions have been reordered according to the specified permutation.
					Eigen::array<long int, 2> newshape({tnsDims[mode], newColDim});
					matricedTns = tnsX.shuffle(permutation).reshape(newshape);

	        		// Map the @c Eigen Tensor to @c Eigen Matrix type.
					return tensorToMatrix(matricedTns, tnsDims[mode], newColDim);
				}
				else
				{
					long int matricized_dim = 1;
					for(std::size_t i=1; i < permutation.size()-1; ++i)
					{
						matricized_dim *= tnsDims[permutation[i]];
					}
					Tensor<TnsSize-1> cubeTns;
					MatrixType        matricedTns(tnsDims[mode], newColDim);
					// Recursive Call for each cube, based on the mode if it is on the last dimension or not.
					if(mode<TnsSize-1)
					{
						for(int cubeId=0; cubeId<tnsDims[TnsSize-1]; cubeId++)
						{
							// Get a TnsSize-1 order hypercube from whole tensor.
							cubeTns = tnsX.chip(cubeId,TnsSize-1);
							// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
							matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_rec(cubeTns, mode);
						}
					}
					else
					{
						for(int cubeId=0; cubeId<tnsDims[TnsSize-2]; cubeId++)
						{
							// Get a TnsSize-1 order hypercube from whole tensor.
							cubeTns = tnsX.chip(cubeId,TnsSize-2);
							// Save the matriced 3D Tensor to matricedTns, after returning from the recursive calls.
							matricedTns.block(0, cubeId * matricized_dim, tnsDims[mode], matricized_dim) = Matricization_rec(cubeTns, TnsSize-2);
						}
					}
					return matricedTns;
				}

	    	}

		} // end namespace v2

  } // end namespace experimental
  #endif // DOXYGEN_SHOULD_SKIP_THIS

} // end namespace partensor

#endif // PARTENSOR_MATRICIZATION_HPP
