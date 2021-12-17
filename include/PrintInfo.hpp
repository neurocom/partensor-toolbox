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
* @file 	 PrintInfo.hpp
* @details
* Provides a list with functions, that prints in the console.
********************************************************************/

#ifndef PARTENSOR_PRINT_INFO_HPP
#define PARTENSOR_PRINT_INFO_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include "TensorOperations.hpp"

namespace partensor
{

    /**
	 * Prints data either @c Matrix or @c Tensor
	 * in the console.
	 * 
	 * @tparam Data    Container type. Either @c 
	 *                 Matrix or @c Tensor.
	 * @param dat [in] @c Data container.
	 * 
	 */
	template<typename Data>
	void print(Data const &dat)
	{
		std::cout << "========== DATA ==========" << std::endl;
		std::cout << dat << std::endl;
	}

	/**
	 * Prints the size of either @c Matrix or 
	 * @c Tensor in the console.
	 * 
	 * @tparam Data    Container type. Either @c 
	 *                 Matrix or @c Tensor.
	 * @param dat [in] @c Data container.
	 */
	template<typename Data>
	void printSize(Data const &dat)
	{
		if constexpr (is_matrix<Data>)
		{
    		std::cout << "Matrix has size: " << dat.rows() << "x" << dat.cols() << std::endl;
		}
		else
		{
			using Dimensions = typename TensorTraits<Data>::Dimensions;
			const Dimensions& tns_dimensions = dat.dimensions();

			static constexpr std::size_t TnsSize = TensorTraits<Data>::TnsSize;
			std::cout << "Tensor has size: [";
			for(std::size_t i=0; i<TnsSize-1; ++i)
				std::cout << tns_dimensions[i] << ", ";

			std::cout << tns_dimensions[TnsSize-1] << std::endl;
		}
  	}

 	/**
	 * Checks if two Matrices or Tensors, are
	 * equal. In the first case make use of @c Eigen 
	 * function @c norm, while in the second one, uses
	 * @c norm function from @c TensorOperations.hpp
	 * 
	 * @tparam Data     Container type. Either @c 
	 *                  Matrix or @c Tensor.
	 * @param arg1 [in] First @c Data container to be compared.
	 * @param arg2 [in] Second @c Data container to be compared.
	 * 
	 * @returns If the data are equal returns 0, otherwise 
	 *          their norm.
	 */
	template<typename Data>
	double checkEquality(Data const &arg1, Data const &arg2)
	{
		if constexpr (is_matrix<Data>)
			return (arg1 - arg2).norm();
		else
			return norm(arg1 - arg2); 
  	}

 	/**
	 * Print the size of file in B, KB, MB, GB.
	 * 
	 * @param fileName [in] The path to the file.
	 */
	inline void matrixFileSize(char const *fileName)
	{
		double kb = 1000.0;
		double mb = pow(kb,2);
		double gb = pow(kb,3);
		std::ifstream::pos_type size;
		std::ifstream InputFile(fileName, std::ifstream::ate | std::ifstream::binary);

		std::cout << std::fixed << std::showpoint;
		std::cout << std::setprecision(2);

		size = InputFile.tellg();
		if(size>0)
		{
		std::cout << "File " << fileName << " has size ";
		if(size<kb)
			std::cout << size << " B" << std::endl;
		else if(size>=kb && size<mb)
			std::cout << static_cast<double>(size)/kb << " KB" << std::endl;
		else if(size>=mb && size<gb)
			std::cout << size/mb << " MB" << std::endl;
		else
			std::cout << size/gb << " GB" << std::endl;
		}
		else
		{
      		std::cout << "\nSomething went wrong, calculating size of " << fileName << " file." << std::endl;
		}
  	}

	/****************************************************
		Print the time has passed:

			1. module = 0(write function)
			2. module = 1(read function)

		elapsedTime = startTime - finishTime
	****************************************************/ 
	inline void printExecutionTime(int const module, double const elapsedTime)
	{
		if(module == 0){ // Writing
			std::cout << "\nElapsed time Writing the matrix: " << elapsedTime << " sec" << std::endl;
		}
		else if(module == 1){ // Reading
			std::cout << "\nElapsed time Reading the matrix: " << elapsedTime << " sec" << std::endl;
		}
		else
			std::cout << "Wrong Choice for computation of time." << std::endl;
	}

	/****************************************************
		 Print Data starting in pos address from a
		Boost Memory Mapped File of size regionSize.
	****************************************************/
	inline void printMatrixFromMMFile(double const *pos, std::size_t const regionSize)
	{
		unsigned int i;
		for(i=0; i<regionSize/sizeof(double); i++)
		std::cout << pos[i] << std::endl;

		std::cout << "Printed " << i << std::endl;
	}

	/****************************************************
      Creates an Tensor and calculates its
			size.
  	****************************************************/
	template<int rank>
	void createTensorAndCheckSize(int const maxDim)
	{
		Eigen::VectorXi dim(rank);
		if(rank>1 && rank<9)
		{
			std::cout << "Create a Tensor with " << rank << " modes and maximum dimension per mode: " << maxDim << std::endl;
			std::cout << "Tensor dimensions per rank: ";
			for(int i=0; i<rank; i++)
			{
				dim[i] = maxDim;
				std::cout << dim[i] << " ";
			}
		}
		Tensor<rank> tns;
		std::cout << "\nSize of simply initialized Tensor: " << sizeof(tns) << std::endl;

		tns.resize(dim);
		tns.setRandom();
	  std::cout << "Size of Tensor after resize of VectorXi - dim and filled with random numbers: " << sizeof(tns) << std::endl;
	}

} // end namespace partensor

#endif // end of PARTENSOR_PRINT_INFO_HPP
