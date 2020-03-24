#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
 * @date      21/03/2019
 * @author    Christos Tsalidis
 * @author    Yiorgos Lourakis
 * @author    George Lykoudis
 * @copyright 2019 Neurocom All Rights Reserved.
 */
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/********************************************************************/
/**
* @file 	 ReadWrite.hpp
* @details
* A variety of functions for reading/writing from/to files. 
* There are also @c read implementations, in case an @c MPI 
* environment has been established, to be used and distribute 
* the data.
********************************************************************/

#ifndef PARTENSOR_READ_WRITE_HPP
#define PARTENSOR_READ_WRITE_HPP

#include "PARTENSOR_basic.hpp"
#include "boost/interprocess/file_mapping.hpp"
#include "boost/interprocess/mapped_region.hpp"

#include <iostream>
#include <fstream>
// Open Function Sys Call
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
// Read-Write Functions Sys Call
#include <unistd.h>

namespace partensor {

	#ifndef DOXYGEN_SHOULD_SKIP_THIS		
	/**
	 * Boolean variable that checks if type T is a c-style string.
	 */ 
	template<typename T>
	inline constexpr bool is_c_str = std::is_same<char const *, typename std::decay<T>::type>::value ||
									 std::is_same<char *, typename std::decay<T>::type>::value;
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS

	// =====================================================
	// ================== WRITE FUNCTIONS ==================
	// =====================================================

	/**
	 * @brief Write function, where data written in appending mode.
	 * 
	 * Use @c stl @c ofstream to write to a file a single quantity.
	 * @tparam FileName  	 Type of input file, either @c std::string 
	 *                       or @c char*.
	 * @param  fvalue   [in] Value that will be written in file.
	 * @param  fileName [in] Path to the file.
	 */ 
	template<typename FileName>
	void writeToFile_append( double   const  fvalue, 
	                         FileName const &fileName )
	{
    	std::ofstream os;
		os.exceptions(std::ofstream::badbit | std::ofstream::failbit);
		try
		{
			os.open(fileName, std::ios::app);
			os << fvalue <<"\n";
			os.close();
		}
    	catch (std::ofstream::failure const &ex)
		{
			std::cerr << "Exception opening/writing/closing file: " << fileName << std::endl;
		}
	}

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Use @c stl @c ofstream to write to a file a struct of data.
	 * 
	 * In case the data to be written are @c Matrix type, then
	 * first creates a temporary variable to store the transpose of
	 * this @c dat and then write this variable to the file.
	 * 
	 * @tparam Data          Type of input data, either @c Matrix 
	 *                       or @c Tensor.
	 * @tparam FileName  	 Type of input file, either @c std::string 
	 *                       or @c char*.
	 * @param  dat      [in] Value that will be written in file.
	 * @param  fileName [in] Path to the file.
	 * @param  size     [in] Number of elements to write.
	 */ 
	template<typename Data, typename FileName>
	void writeToFile_cppStream( Data 		const &dat, 
								FileName    const &fileName, 
								std::size_t const  size)
	{
		std::ofstream os;
		os.exceptions(std::ofstream::badbit | std::ofstream::failbit);
		try
		{
			os.open(fileName, std::ios::binary | std::ios::trunc);

			std::size_t count = 0;
			if constexpr (is_matrix<Data>)
			{
				using DataType  = typename MatrixTraits<Data>::DataType;
				Matrix dat_T;
				dat_T           = dat.transpose();
				count           = size*sizeof(DataType);
				os.write(reinterpret_cast<char*>(dat_T.data()), count);				
			}
			else
			{
				using DataType  = typename TensorTraits<Data>::DataType;
				count           = size*sizeof(DataType);
				os.write((char*)dat.data(), count);
			}
			os.close();
		}
		catch (std::ofstream::failure const &ex)
		{
			std::cerr << "Exception opening/writing/closing file: " << fileName << std::endl;
		}
	}
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Use c-Type @c FILE and @c fopen to write to a file a struct of data.
	 * 
	 * In case the data to be written are @c Matrix type, then
	 * first creates a temporary variable to store the transpose of
	 * this @c dat and then write this variable to the file.
	 * 
	 * @tparam Data          Type of input data, either @c Matrix 
	 *                       or @c Tensor.
	 * @tparam FileName  	 Type of input file, either @c std::string 
	 *                       or @c char*.
	 * @param  dat      [in] Value that will be written in file.
	 * @param  fileName [in] Path to the file.
	 * @param  size     [in] Number of elements to write.
	 */ 
	template<typename Data, typename FileName>
	void writeToFile_cStream( Data        const &dat, 
	                          FileName 	  const &fileName, 
							  std::size_t const  size     )
	{
		try
		{
			FILE *of;
			if constexpr (! is_c_str<FileName>)
			{
				of = fopen(fileName.c_str(), "wb");
			}
			else
			{
				of = fopen(fileName, "wb");
			}

			std::size_t fw = 0;
			if constexpr (is_matrix<Data>)
			{
				using DataType = typename MatrixTraits<Data>::DataType;
				Matrix dat_T;
				dat_T = dat.transpose();
				fw    = fwrite(dat_T.data(), sizeof(DataType), size, of);
			}
			else
			{
				using DataType = typename TensorTraits<Data>::DataType;
				fw = fwrite(dat.data(), sizeof(DataType), size, of);
			}
			if(fw != size)
					throw 0;

			fclose(of);
		}
		catch(...)
		{
			std::cerr << "Exception opening/writing/closing file: " << fileName << std::endl;
    	}
  	}
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Use @c system @c calls to write to a file a struct of data.
	 * 
	 * In case the data to be written are @c Matrix type, then
	 * first creates a temporary variable to store the transpose of
	 * this @c dat and then write this variable to the file.
	 * 
	 * @tparam Data          Type of input data, either @c Matrix 
	 *                       or @c Tensor.
	 * @tparam FileName  	 Type of input file, either @c std::string 
	 *                       or @c char*.
	 * @param  dat      [in] Value that will be written in file.
	 * @param  fileName [in] Path to the file.
	 * @param  size     [in] Number of elements to write.
	 */ 
	template<typename Data, typename FileName>
	void writeToFile_sysCalls( Data 	   const &dat, 
							   FileName    const &fileName, 
							   std::size_t const  size     )
	{
		// equal to open() with flags = O_CREAT|O_WRONLY|O_TRUNC
		// S_IRWXU - user (file owner) has write only permission
		try
		{
			int fileDescriptor;
			if constexpr (! is_c_str<FileName>)
			{
				fileDescriptor = creat(fileName.c_str(), S_IWUSR);
			}
			else
			{
				fileDescriptor = creat(fileName, S_IWUSR);
			}
			if(fileDescriptor > 0)
			{
				std::size_t count = 0;
				ssize_t     writeToFile;
				if constexpr (is_matrix<Data>)
				{
					using DataType = typename MatrixTraits<Data>::DataType;
					Matrix dat_T;
					dat_T          = dat.transpose();
					count 		   = size*sizeof(DataType);
	      			writeToFile    = write(fileDescriptor, dat_T.data(), count);
				}
				else
				{
					using DataType = typename TensorTraits<Data>::DataType;
					count          = size*sizeof(DataType);
					writeToFile    = write(fileDescriptor, dat.data(), count);
				}
				if (writeToFile <= 0)
				{
					throw -1;
				}
			}
			if(close(fileDescriptor) < 0)
			{
				throw 0;
			}
		}
		catch(...)
		{
			std::cerr << "Exception opening/writing/closing file: " << fileName << std::endl;
		}
	}
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS

	/**
    * @brief Write a Struct of data in a file, in a serial manner.
    * 
    * If the data are stored in a struct of data, then this function
    * can be used to write them in a file.
	* 
    * @tparam Data          Input data type. It can be either 
    *                       @c Matrix or @c Tensor.
    * @tparam FileName  	Input file type. Either @c std::string or @c char*.
    * @param  dat      [in] @c Data struct that will be written in file.                       
    * @param  fileName [in] Specify the path to the file, where the data will be
    * 					    written.
    * @param  size     [in] Number of elements to write. (e.g. For a Matrix with
    *                       rows and columns then the size should be rows*columns)
    */ 
    template<typename Data, typename FileName>
    void write( Data 		const &dat, 
			    FileName    const &fileName, 
			    std::size_t const  size     )
	{
		writeToFile_cppStream(dat, fileName, size);	
	}

	// =====================================================
	// =================== READ FUNCTIONS ==================
	// =====================================================

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Use @c stl @c ifstream to read from a file a struct of data.
	 * 
	 * In case the data to be read is @c Matrix type, then
	 * first creates a temporary variable to store the data and 
	 * then copies the transpose of this variable to @c dat.
	 * 
	 * @tparam Data          	 Type of input data, either @c Matrix or 
	 *                           @c Tensor.
	 * @tparam FileName  	 	 Type of input file, either @c std::string 
	 *                       	 or @c char*.
	 * @param  fileName [in] 	 Path to the file.
	 * @param  size     [in] 	 Number of elements to read.
	 * @param  skip     [in] 	 Number of elements to skip.
	 * @param  dat      [in,out] Data read from file. @c dat MUST be 
	 *                           initialized before called.
	 */ 
	template<typename Data, typename FileName>
	void readFromFile_cppStream( FileName 	 const &fileName, 
								 std::size_t const  size, 
								 std::size_t const  skip, 
								 Data 			   &dat      )
	{
		std::ifstream is;
		is.exceptions(std::ifstream::badbit | std::ifstream::failbit);
		try
		{
			is.open(fileName, std::ios::binary);
			std::size_t count = 0;
			if constexpr (is_matrix<Data>)
			{
				using DataType = typename MatrixTraits<Data>::DataType;
				Matrix dat_T(dat.cols(), dat.rows()); 
				count 		   = size*sizeof(DataType);
				is.ignore(skip*sizeof(DataType));  // where to start
				is.read(reinterpret_cast<char *>(dat_T.data()), count);
				dat = dat_T.transpose();
			}
			else
			{
				using DataType = typename TensorTraits<Data>::DataType;
				count 		   = size*sizeof(DataType);
				is.ignore(skip*sizeof(DataType));  // where to start
				is.read(reinterpret_cast<char*>(dat.data()), count);
			}
			is.close();
    	}
		catch (std::ifstream::failure const &ex)
		{
			std::cerr << "Exception opening/reading/closing file: " << fileName << std::endl;
		}
	}
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS 
	
	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Use c-type @c FILE and fread to read from a file a struct of data.
	 * 
	 * In case the data to be read is @c Matrix type, then
	 * first creates a temporary variable to store the data and 
	 * then copies the transpose of this variable to @c dat.
	 * 
	 * @tparam Data          	 Type of input data, either @c Matrix or 
	 *                           @c Tensor.
	 * @tparam FileName  	 	 Type of input file, either @c std::string 
	 *                       	 or @c char*.
	 * @param  fileName [in] 	 Path to the file.
	 * @param  size     [in] 	 Number of elements to read.
	 * @param  skip     [in] 	 Number of elements to skip.
	 * @param  dat      [in,out] Data read from file. @c dat MUST be 
	 *                           initialized before called.
	 */ 
	template<typename Data, typename FileName>
	void readFromFile_cStream( FileName    const &fileName, 
							   std::size_t const  size, 
							   std::size_t const  skip, 
							   Data              &dat      )
	{
		try
		{
			FILE *inputStream;
			if constexpr (! is_c_str<FileName>)
			{
				inputStream = fopen(fileName.c_str(), "rb");
			}
			else
			{
				inputStream = fopen(fileName, "rb");
			}
			std::size_t fr = 0;
			if constexpr (is_matrix<Data>)
			{
				using DataType = typename MatrixTraits<Data>::DataType;
				fseek(inputStream, skip*sizeof(DataType), SEEK_SET); // where to start
				
				Matrix dat_T(dat.cols(), dat.rows()); 
	      		fr  = fread(dat_T.data(), sizeof(DataType), size, inputStream);
				dat = dat_T.transpose();
			}
			else
			{
				using DataType = typename TensorTraits<Data>::DataType;
				fseek(inputStream, skip*sizeof(DataType), SEEK_SET); // where to start
	      		fr = fread(dat.data(), sizeof(DataType), size, inputStream);
			}
			if(fr!=size)
					{
				throw 0;
			}
			fclose(inputStream);
		}
		catch(...)
		{
			std::cerr << "Exception opening/writing/closing file: " << fileName << std::endl;
		}
  	}
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Use @c system @c calls to read from a file a struct of data.
	 * 
	 * In case the data to be read is @c Matrix type, then
	 * first creates a temporary variable to store the data and 
	 * then copies the transpose of this variable to @c dat.
	 * 
	 * @tparam Data          	 Type of input data, either @c Matrix or 
	 *                           @c Tensor.
	 * @tparam FileName  	 	 Type of input file, either @c std::string 
	 *                       	 or @c char*.
	 * @param  fileName [in] 	 Path to the file.
	 * @param  size     [in] 	 Number of elements to read.
	 * @param  skip     [in] 	 Number of elements to skip.
	 * @param  dat      [in,out] Data read from file. @c dat MUST be 
	 *                           initialized before called.
	 */ 
	template<typename Data, typename FileName>
	void readFromFile_sysCalls( FileName    const &fileName, 
								std::size_t const  size, 
								off_t       const  skip, 
								Data              &dat      )
	{
		// S_IRWXU - user (file owner) has read only permission
		try
		{
			int fileDescriptor;
			if constexpr (! is_c_str<FileName>)
			{
				fileDescriptor = open(fileName.c_str(), S_IRUSR);
			}
			else
			{
				fileDescriptor = open(fileName, S_IRUSR);
			}
	   		if(fileDescriptor > 0)
		 	{
				std::size_t  count = 0;
				off_t        seek  = 0;
				ssize_t      readFromFile;
				if constexpr (is_matrix<Data>)
				{
					using DataType = typename MatrixTraits<Data>::DataType;
					count 		   = size*sizeof(DataType);
				 	seek		   = lseek(fileDescriptor, skip*sizeof(DataType), SEEK_SET);

					Matrix dat_T(dat.cols(), dat.rows()); 
					readFromFile = read(fileDescriptor, dat_T.data(), count);
					dat          = dat_T.transpose();
				}
				else
				{
					using DataType = typename TensorTraits<Data>::DataType;
					count		   = size*sizeof(DataType);
				 	seek		   = lseek(fileDescriptor, skip*sizeof(DataType), SEEK_SET);
					readFromFile   = read(fileDescriptor, dat.data(), count);
				}

			 	if(seek == -1 || readFromFile <= 0)
			 	{
				 	throw -1;
			 	}
	   		}
			if(close(fileDescriptor) < 0)
			{
				throw 0;
			}
	 	}
		catch(...)
		{
			std::cerr << "Exception opening/reading/closing file: " << fileName << std::endl;
		}
  	}
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Use @c boost @c interprocess module to read from a file a struct of data.
	 * 
	 * In case the data to be read is @c Matrix type, then
	 * first creates a temporary variable to store the data and 
	 * then copies the transpose of this variable to @c dat.
	 * 
	 * @tparam Data          	 Type of input data, either  @c Matrix or 
	 *                           @c Tensor.
	 * @tparam FileName  	 	 Type of input file, either @c std::string 
	 *                       	 or @c char*.
	 * @param  fileName [in] 	 Path to the file.
	 * @param  dat      [in,out] Data read from file copied in dat.
	 */
	template<typename Data, typename FileName>
  	void readFromFile_memMap( FileName const &fileName, 
	                          Data           &dat      )
	{
		try
		{
			boost::interprocess::mode_t        mode = boost::interprocess::read_only;
	    	boost::interprocess::file_mapping  fm;
			if constexpr (is_c_str<FileName>)
			{
				fm = boost::interprocess::file_mapping{fileName, mode};
			}
			else
			{
				fm = boost::interprocess::file_mapping{fileName.c_str(), mode};
			}
	    	boost::interprocess::mapped_region region(fm, mode, 0, 0);
			if constexpr (is_matrix<Data>)
			{
				using DataType = typename MatrixTraits<Data>::DataType;
				DataType* addr = static_cast<DataType*>(region.get_address());

				Matrix dat_T(dat.cols(), dat.rows()); 
		    	dat_T 		   = Eigen::Map<Data>(addr, dat.cols(), dat.rows());
				dat            = dat_T.transpose();
			}
			else
			{
				using DataType 	 							  = typename TensorTraits<Data>::DataType;
				using Dimensions 							  = typename TensorTraits<Data>::Dimensions;
				const Dimensions tnsDims = dat.dimensions();
				DataType* addr 	 							  = static_cast<DataType*>(region.get_address());
		        dat                                           = Eigen::TensorMap<Data>(addr, tnsDims);
			}
		}
		catch(boost::interprocess::interprocess_exception &ex)
		{
			std::cerr << ex.what() << std::endl;
		}
		catch(...)
		{
			std::cerr << "Unknown Exception" << std::endl;
		}
	}
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS 

	/**
	 * @brief  Read from a file a Struct, in a serial manner.
	 * 
	 * Use this functions to read from @c fileName, and store @c Eigen
	 * data of @c Matrix or @c Tensor type.
	 * 
	 * @tparam Data          	 Input data type. It can be either @c Matrix or 
	 *                           @c Tensor.
	 * @tparam FileName  	 	 Input file type. Either @c std::string 
	 *                       	 or @c char*.
	 * @param  fileName [in] 	 Specify the path to the file, where the data are
	 * 							 located. Variable can be of @c FileName type.
	 * @param  size     [in] 	 Number of elements to read. (e.g. For a @c Matrix with
   	 *                           rows and columns, then the size should be rows*columns)
	 * @param  skip     [in] 	 Number of elements to skip.
	 * @param  dat      [in,out] Struct where the data will read from file and stored. 
	 *                           
	 * @note @c dat MUST be initialized before the function call called.
	 */ 
	template<typename Data, typename FileName>
	void read( FileName    const &fileName, 
			   std::size_t const  size, 
			   std::size_t const  skip, 
			   Data 			 &dat      )
	{
		readFromFile_cppStream(fileName, size, skip, dat);
	}

	/**
	 * @brief Read @c FMRI data and store in a @c Matrix struct. 
	 * 
	 * Implementation for @c FMRI data, that are stored in files, in a specific way.
	 * In each file,a single row is stored, from the 3D data and the pattern of the file 
	 * names are like : fileName_00000, fileName_00001, etc. Then use @c readFMRI_matrix
	 * to read from all files and store the data in a @c Matrix, equivalent
	 * to the third order Tensor matricization in first mode.
	 * 
	 * @tparam FileName  	 	 Type of input file, either @c std::string 
	 *                       	 or @c char*.
	 * @tparam Dimensions      	 Array type for Tensor dimensions.
	 * @param  fileName [in] 	 Specify the path to the file, where the data are
	 * 							 located. Variable can be of @c FileName type.
	 * @param  tnsDims  [in] 	 @c Eigen or @c stl @c Array with the lengths of each  
	 *                           of Tensor dimension.
	 * @param  skip     [in] 	 Number of elements to skip in @c fileName.
	 * @param  dat      [in,out] Data read from file. 
	 * 
	 * @note @c dat MUST be initialized before the function call called.
	 */
	template<typename FileName, typename Dimensions>
	void readFMRI_matrix( FileName    const &fileName, 
						  Dimensions  const  tnsDims, 
						  std::size_t const  skip, 
						  Matrix           &dat      )
	{
		std::string prefix               = fileName.substr(0,fileName.size()-4);
		std::string extension            = fileName.substr(fileName.size()-4,50);
		std::string newFileName;
		Matrix      _temp(tnsDims[0],1);

		for(auto fn=0; fn<tnsDims[1] * tnsDims[2]; fn++)
		{
			newFileName = prefix + "_";
			newFileName += std::to_string(fn/10000) + std::to_string(fn/1000) + std::to_string(fn/100);
			newFileName += std::to_string(fn/10) + std::to_string(fn%10) + extension;

			readFromFile_cppStream(newFileName, tnsDims[0], skip, _temp);
			dat.col(fn) = _temp;
		}
  	}

	/**
	 * @brief Read @c FMRI data and store in a @c Tensor struct. 
	 * 
	 * Implementation for @c FMRI data, that are stored in files, in a specific way.
	 * In each file,a single row is stored, from the 3D data and the pattern of the file 
	 * names are like : fileName_00000, fileName_00001, etc. Then use @c readFMRI_tensor
	 * to read from all files and store the data in a @c Tensor with order  
	 * equal to 3.
	 * @tparam Data              @c Tensor type. Inside function used to extract
	 *                           Tensor @c DataType and @c Dimensions.
	 * @tparam FileName  	 	 Input file type. Either @c std::string or @c char*.
	 * @param  fileName [in] 	 Specify the path to the file, where the data are
	 * 							 located. Variable can be of @c FileName type.
	 * @param  skip     [in] 	 Number of elements to skip.
	 * @param  dat      [in,out] Data read from file. 
	 * 
	 * @note @c dat MUST be initialized before the function call called.
	 */
	template<typename Data, typename FileName>
	void readFMRI_tensor( FileName 	  const &fileName, 
						  std::size_t const  skip, 
						  Data 				&dat	  )
	{
		using Dimensions = typename TensorTraits<Data>::Dimensions;

		const Dimensions&   	  tnsDims           = dat.dimensions();
		std::string 			  prefix    	 	= fileName.substr(0,fileName.size()-4);
		std::string 			  extension 	 	= fileName.substr(fileName.size()-4,50);
		std::string 			  newFileName;
		int 					  fn				= 0;
		Tensor<1> _temp(tnsDims[0]);

		for(int i=0; i<tnsDims[2]; i++)
		{
			for(int j=0; j<tnsDims[1]; j++, fn++)
			{
				newFileName = prefix + "_";
				newFileName += std::to_string(fn/10000) + std::to_string(fn/1000) + std::to_string(fn/100);
				newFileName += std::to_string(fn/10) + std::to_string(fn%10) + extension;

				readFromFile_cppStream(newFileName, tnsDims[0], skip, _temp);
				dat.chip(i,2).chip(j,1) = _temp;
			}
		}
  	}

	// =============================================================
	// =================== READ FUNCTIONS FOR MPI ==================
	// =============================================================

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Computes @c Tensor 's offsets. Compute for each dimension how 
	 * many elements to skip.
	 * For example in a 4-dimension Tensor, how many rows, matrices, and cubes
	 * to skip in each dimension. It should be called before @c readTensor.
	 * 
	 * @tparam TnsSize             Length of the @c stl array, also Tensor Order.
	 * @param  dimensions [in] 	   @c stl array with the lengths of each of Tensor dimension.
	 * @param  skip       [in] 	   @c stl array with the number of elements to skip.
	 * @param  offset     [in,out] @c stl array containing where to start for each Tensor dimension.
	 */ 
	template<std::size_t TnsSize>
	void offset_calculation( std::array<int, TnsSize> const &dimensions, 
							 std::array<int, TnsSize> const &skip, 
							 std::array<int, TnsSize> 		&offset     )
	{
		for (int i=TnsSize-1; i>1; i--)
		{
			// How many cubes to skip
			offset[i] = std::accumulate(dimensions.begin(), dimensions.end()-(TnsSize-i), 1, std::multiplies<int>()) * skip[i];
		}
		offset[1] =  dimensions[0] * skip[1]; // how many matrices to skip
		offset[0] =  skip[0]; 				  // how many rows to skip
	}
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS

	#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/**
	 * Implementation of @c readTensor function. 
	 * Recursive function that skips rows, matrices, cubes or hypercubes 
	 * until it reach the correct column to read based on @c MPI rank. 
	 * It uses @c offset_calculation for having the number elements to skip 
	 * per dimension and @c stl @c ifsream in order to read from the file.
	 * 
	 * @tparam idx                 	  Identification for the offset array.
	 * @tparam TnsSize                Order of tensor @c dat. Also, the size of the @c stl arrays
	 *                                @c tnsDims, @c local_tnsDims and @c offset.
	 * @param  tnsDims       [in] 	  @c stl array with initial lengths for each Tensor dimension.
	 * @param  local_tnsDims [in]     @c stl array with lengths for each Tensor dimension per processor.
	 * @param  offset        [in]     @c stl array containing where to start for each Tensor dimension,
	 * 								  computed in @c offset_calculation.
	 * @param  inputStream   [in,out] @c stl @c ifstream that points to the file where the data are located.
	 * @param  dat           [in,out] Data read from file. 
	 * 
	 * @note @c dat MUST be initialized before the function call called.
	 */ 
	template<std::size_t idx, std::size_t TnsSize>
	void readTensor_util( std::array<int, TnsSize> const    &tnsDims, 
						  std::array<int, TnsSize> const    &local_tnsDims,
						  std::array<int, TnsSize> const    &offset,
						  std::ifstream       		        &is, 
						  Tensor<static_cast<int>(TnsSize)> &dat           )
	{
		using DT = typename TensorTraits<Tensor<static_cast<int>(TnsSize)>>::DataType;
		if constexpr (idx>0)
		{
			// e.g. I*J*K in 4D tensor
			int dim_product = std::accumulate(tnsDims.begin(), tnsDims.begin()+idx, 1, std::multiplies<int>());
			// ignore
			is.ignore(offset[idx]*sizeof(DT));

			for (int i=0; i<local_tnsDims[idx]; i++)
			{
				readTensor_util<idx-1, TnsSize>(tnsDims, local_tnsDims, offset, is, dat);
			}
			is.ignore( dim_product*(tnsDims[idx] - local_tnsDims[idx] - offset[idx])*sizeof(DT) );
		}
		else
		{
			Tensor<1> tns_col(local_tnsDims[0]);
			// ignore offset[0] rows
			is.ignore(offset[0]*sizeof(DT));
			is.read(reinterpret_cast<char*>(tns_col.data()), local_tnsDims[0]*sizeof(DT));
			// use tensor indexing for reading from file
			for (int i=0; i<local_tnsDims[0]; i++)
			{
				static int idx_read = 0;
				dat(idx_read) = tns_col(i);

				idx_read++;
			}
			// ignore remaining column elements
			is.ignore( (tnsDims[0] - local_tnsDims[0] - offset[0])*sizeof(DT) );
		}
	}
	#endif // end of DOXYGEN_SHOULD_SKIP_THIS 
	
	/**
	 * @brief Read and store an @c Tensor struct with the use of MPI.
	 * 
	 * When the data of a Tensor are stored in a @c fileName, and an @c MPI environent has
	 * been set, then @c readTensor can be used to extract those data from the file.
	 * Passing the correct values in arguments : @c local_tnsDims and @c local_skip for each 
	 * processor, then a sub-tensor is distributed to each one, in @c dat. 
	 * 
	 * @tparam TnsSize                Order of tensor @c dat. Also, the size of the @c stl arrays
	 *                                @c tnsDims, @c local_tnsDims and @c local_skip.
	 * @tparam FileName               Type of input file, either @c std::string 
	 *                                or @c char*.
	 * @param  fileName      [in] 	  Specify the path to the file, where the data are
	 * 							 	  located. Variable can be of @c FileName type.
	 * @param  tnsDims       [in] 	  @c stl array with initial lengths for each Tensor dimension.
	 * @param  local_tnsDims [in]     @c stl array with lengths for each Tensor dimension per processor.
	 * @param  local_skip    [in]     @c stl array containing where to start for each Tensor dimension,
	 *                                computed in @c offset_calculation.
	 * @param  dat           [in,out] Data read from file. 
	 * 
	 * @note @c dat MUST be initialized before the function call called.
	 */
	template<std::size_t TnsSize, typename FileName>
	void readTensor( FileName 				           const &fileName, 
	                 std::array<int, TnsSize>          const &tnsDims, 
					 std::array<int, TnsSize>          const &local_tnsDims,
					 std::array<int, TnsSize>          const &local_skip, 
					 Tensor<static_cast<int>(TnsSize)>	     &dat           )
	{
		std::ifstream  is;
		try
		{
			is.open(fileName, std::ios::binary);
			std::array<int, TnsSize> offset;
			offset_calculation( tnsDims, local_skip, offset );
			readTensor_util<TnsSize-1, TnsSize>( tnsDims, local_tnsDims, offset, is, dat );

			is.close();
		}
		catch (std::ofstream::failure const &ex)
		{
			std::cerr << "Exception opening/reading/closing file: " << fileName << std::endl;
		}
	}

	/**
	 * @brief Read @c FMRI data and store in a @c Tensor struct with
	 *        the use of MPI.
	 * 
	 * Implementation for @c FMRI data, that are stored in files, in a specific way.
	 * In each file,a single row is stored, from the 3D data and the pattern of the file 
	 * names are like : fileName_00000, fileName_00001, etc. Then use @c readFMRIFromFiles
	 * in an @c MPI environment, to read from all files and store the data in an @c Tensor 
	 * with order equal to 3.
	 * 
	 * @tparam TnsSize                Order of tensor @c dat. Also, the size of the @c stl arrays
	 *                                @c tnsDims, @c local_tnsDims and @c local_skip.
	 * @tparam FileName               Input file type. Either @c std::string 
	 *                                or @c char*.
	 * @param  fileName      [in] 	  Specify the path to the file, where the data are
	 * 							 	  located. Variable can be of @c FileName type.
	 * @param  tnsDims       [in] 	  @c stl array with initial lengths for each Tensor dimension.
	 * @param  local_tnsDims [in]     @c stl array with lengths for each Tensor dimension per processor.
	 * @param  local_skip    [in]     @c stl array with number of skip elements per Tensor dimension.
	 * @param  dat           [in,out] Data read from file. 
	 * 
	 * @note @c dat MUST be initialized before the function call called.
	 */
	template<std::size_t TnsSize, typename FileName>
	void readFMRI_mpi(  FileName   				          const &fileName, 
						std::array<int, TnsSize>          const &tnsDims, 
						std::array<int, TnsSize>          const &local_tnsDims,
						std::array<int, TnsSize>          const &local_skip, 
						Tensor<static_cast<int>(TnsSize)>       &dat          )
	{
		std::string  			  prefix       = fileName.substr(0,fileName.size()-4);
		std::string  			  extension    = fileName.substr(fileName.size()-4,50);
		std::string               newFileName;
		int 					  fn 		   = local_skip[2] * tnsDims[1];
		Tensor<1> _temp(local_tnsDims[0]);

		for(int i=0; i<local_tnsDims[2]; i++)
		{
			fn += local_skip[1];
			for(int j=0; j<local_tnsDims[1]; j++)
			{
				newFileName = prefix + "_";
				newFileName += std::to_string(fn/10000) + std::to_string(fn/1000) + std::to_string(fn/100);
				newFileName += std::to_string(fn/10) + std::to_string(fn%10) + extension;
				readFromFile_cppStream(newFileName.c_str(), local_tnsDims[0], local_skip[0], _temp);

				dat.chip(i,2).chip(j,1) = _temp;
				fn++;
			}
			fn += tnsDims[1] - local_tnsDims[1] - local_skip[1];
		}
    }

} // end namespace partensor

#endif // end of PARTENSOR_READ_WRITE_HPP
