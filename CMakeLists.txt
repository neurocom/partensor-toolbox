CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

# -DDEVEL_BUILD=1
# -DBOOST_LAYOUT_OPT=tagged
# -DDEVEL_BUILD=1 -DBOOST_LAYOUT_OPT=tagged -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ -DCMAKE_C_COMPILER=/usr/local/bin/gcc
# -DDEVEL_BUILD=1 -DBOOST_LAYOUT_OPT=tagged -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ -DCMAKE_C_COMPILER=/usr/local/bin/gcc
# -DDEVEL_BUILD=1 -DBOOST_LAYOUT_OPT=tagged -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=/opt/rh/devtoolset-8/root/usr/bin/g++ -DCMAKE_C_COMPILER=/opt/rh/devtoolset-8/root/usr/bin/gcc


PROJECT(partensor)

########################################################################################################################
###	CMAKE INCLUDES
########################################################################################################################
###	@brief		Required includes
## include(ConfigCmake.cmake REQUIRED)
###	@brief		Optional includes
INCLUDE(CMakeToolsHelpers OPTIONAL)

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/include)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

########################################################################################################################
###    SET VARIABLES
########################################################################################################################
SET(CMAKE_MINIMUM_REQUIRED_VERSION                           "3.5")

#################
##### DEVEL_BUILD parameter 
##### ONLY FOR DEVELOPMENT REASONS!
##### MUST be 1 in order to use TBB and Google Test/Benchmark
#################
#################

IF (APPLE)
  SET(CPP_LIBS                                                 "/usr/local/SDKS/CPP_LIBS")
ELSE (APPLE)
  SET(CPP_LIBS                                                 "/SDKS/CPP_LIBS")
ENDIF (APPLE)

SET(BOOST_ROOT                                               ${CPP_LIBS}/boost)
SET(EIGEN_ROOT_DIR                                           ${CPP_LIBS}/eigen)
SET(MPI_ROOT_DIR                                             ${CPP_LIBS}/mpi)
SET(SPDLOG_ROOT_DIR                                          ${CPP_LIBS}/spdlog)

SET(CMAKE_SRC                                                ${CMAKE_CURRENT_LIST_DIR}/cmake/include)
SET(INCLUDE_DIR                                              ${CMAKE_CURRENT_LIST_DIR}/include)
SET(TEST_DIR                                                 ${CMAKE_CURRENT_LIST_DIR}/test)
SET(INCLUDE_DIR_NCOM                                         ${INCLUDE_DIR}/ncom)
SET(INCLUDE_DIR_TUC                                          ${INCLUDE_DIR}/tuc)

IF (DEFINED ENV{PARTENSOR_DEV_ENV})
	SET(DEVEL_BUILD $ENV{PARTENSOR_DEV_ENV} CACHE STRING "Partensor development" FORCE)
ELSEIF (DEFINED PARTENSOR_DEV_OPT)
	SET(DEVEL_BUILD $ENV{PARTENSOR_DEV_OPT} CACHE STRING "Partensor development" FORCE)
ENDIF()


IF(NOT DEFINED DEVEL_BUILD)
  SET(INCLUDE_DIR_NCOM ${INCLUDE_DIR})
  SET(INCLUDE_DIR_TUC ${INCLUDE_DIR})
ELSE(NOT DEFINED DEVEL_BUILD)
  SET(TEST_DIR         ${TEST_DIR}/ncom)
ENDIF(NOT DEFINED DEVEL_BUILD)
SET(DEVEL_BUILD                                              ${DEVEL_BUILD})

SET(BIN_DIR                                                  ${CMAKE_CURRENT_LIST_DIR}/bin)
SET(DISPLAY_VERSIONS                                         0)
########################################################################################################################

########################################################################################################################
###     Get build platform information
########################################################################################################################
cmake_host_system_information(RESULT platform.hostname              QUERY HOSTNAME)
cmake_host_system_information(RESULT platform.os.name               QUERY OS_NAME)
cmake_host_system_information(RESULT platform.processor.description QUERY PROCESSOR_DESCRIPTION)
cmake_host_system_information(RESULT platform.processor.cores       QUERY NUMBER_OF_LOGICAL_CORES)
cmake_host_system_information(RESULT platform.memory.size           QUERY TOTAL_PHYSICAL_MEMORY)
########################################################################################################################

OPTION (USE_OpenMP "Use OpenMP to enamble <omp.h>" ON)

IF (APPLE)
  IF (CMAKE_C_COMPILER_ID MATCHES "Clang")
    MESSAGE( STATUS  "[ PARTENSOR ] Clang C toolset found." )
    SET(OpenMP_C "${CMAKE_C_COMPILER}")
    SET(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
    SET(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
    SET(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
    SET(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
    SET(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
  ENDIF()

  IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    MESSAGE( STATUS  "[ PARTENSOR ] Clang C++ toolset found." )
    SET(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    SET(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
    SET(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
    SET(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    SET(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    SET(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
  ENDIF()
ELSE ()
  ADD_DEFINITIONS(-fopenmp)
ENDIF()

ADD_DEFINITIONS(-Wall -std=c++17)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

SET(CMAKE_CXX_FLAGS              "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wextra -std=c++17")
SET(CMAKE_CXX_FLAGS              "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wno-deprecated-copy")
# SET(CMAKE_CXX_FLAGS_RELEASE      "-O3")
set(CMAKE_CXX_FLAGS_RELEASE      "-O3 -funroll-loops -DEIGEN_NO_DEBUG -march=native -mtune=native ")
SET(CMAKE_CXX_FLAGS_DEBUG        "-g")

MESSAGE(STATUS "[ PARTENSOR ] Build type: ${CMAKE_BUILD_TYPE}")

########################################################################################################################
# Eigen
########################################################################################################################
MESSAGE( STATUS  "[ PARTENSOR ] Configuring Eigen" )

IF (EIGEN_ROOT_DIR)
	 FILE(READ "${EIGEN_ROOT_DIR}/Eigen/src/Core/util/Macros.h" _eigen_version_file)
   STRING(REGEX REPLACE ".*#define EIGEN_WORLD_VERSION ([0-9]+).*" "\\1"
                EIGEN_VERSION_WORLD "${_eigen_version_file}")
	 STRING(REGEX REPLACE ".*#define EIGEN_MAJOR_VERSION ([0-9]+).*" "\\1"
                EIGEN_VERSION_MAJOR "${_eigen_version_file}")
   STRING(REGEX REPLACE ".*#define EIGEN_MINOR_VERSION ([0-9]+).*" "\\1"
                EIGEN_VERSION_MINOR "${_eigen_version_file}")
   SET(EIGEN_VERSION "${EIGEN_VERSION_WORLD}.${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}")
ENDIF()

MESSAGE(STATUS "[ PARTENSOR ] EIGEN_VERSION                 : ${EIGEN_VERSION}")
MESSAGE(STATUS "[ PARTENSOR ] Config EIGEN END...")
########################################################################################################################

########################################################################################################################
# SPDLOG
########################################################################################################################
IF (SPDLOG_ROOT_DIR)
  FILE(READ "${SPDLOG_ROOT_DIR}/include/spdlog/version.h" _spdlog_version_file)
  STRING(REGEX REPLACE ".*#define SPDLOG_VER_MAJOR ([0-9]+).*" "\\1"
               SPDLOG_VER_MAJOR "${_spdlog_version_file}")
  STRING(REGEX REPLACE ".*#define SPDLOG_VER_MINOR ([0-9]+).*" "\\1"
               SPDLOG_VER_MINOR "${_spdlog_version_file}")
  STRING(REGEX REPLACE ".*#define SPDLOG_VER_PATCH ([0-9]+).*" "\\1"
               SPDLOG_VER_PATCH "${_spdlog_version_file}")
  SET(SPDLOG_VERSION "${SPDLOG_VER_MAJOR}.${SPDLOG_VER_MINOR}.${SPDLOG_VER_PATCH}")
ENDIF()

MESSAGE(STATUS "[ PARTENSOR ] SPDLOG_VERSION                : ${SPDLOG_VERSION}")
########################################################################################################################

########################################################################################################################
###  BOOST
########################################################################################################################
MESSAGE(STATUS   "[ PARTENSOR] Configuring BOOST")
SET(BOOST_LAYOUT "system")

IF (DEFINED ENV{BOOST_LAYOUT_ENV})
	SET(BOOST_LAYOUT $ENV{BOOST_LAYOUT_ENV} CACHE STRING "Boost layout" FORCE)
ELSEIF (DEFINED BOOST_LAYOUT_OPT)
	SET(BOOST_LAYOUT ${BOOST_LAYOUT_OPT} CACHE STRING "Boost layout" FORCE)
ELSE()
	SET(BOOST_LAYOUT "system" CACHE STRING "tagged boost layout" FORCE)
ENDIF()

SET(BOOST_COMPONENTS_NEEDED           mpi
                                      serialization)
SET(USE_PIC_LIBS                      OFF)
INCLUDE(ConfigBoost)

MESSAGE(STATUS   "[ PARTENSOR ] Boost_LIBRARIES : ${Boost_LIBRARIES}")

MESSAGE(STATUS   "[ PARTENSOR ] Config BOOST END...")
########################################################################################################################

########################################################################################################################
###  MPI
########################################################################################################################
MESSAGE(STATUS   "[ PARTENSOR] Configuring MPI")
MESSAGE(STATUS   "[ PARTENSOR ] Settting MPI_ROOT_DIR         : ${MPI_ROOT_DIR}")
IF (NOT EXISTS ${MPI_ROOT_DIR})
  MESSAGE(SEND_ERROR "[ PARTENSOR ] Cannot find MPI_ROOT_DIR    : ${MPI_ROOT_DIR}")
ELSE (NOT EXISTS ${MPI_ROOT_DIR})
  MESSAGE(STATUS "[ PARTENSOR ] Using MPI_ROOT_DIR            : ${MPI_ROOT_DIR}")
  MESSAGE(STATUS "[ PARTENSOR ] Processing                    : ${CMAKE_CMAKE_SRC}/FindMPI.cmake")

  INCLUDE(ConfigMPI)
  INCLUDE(FindMPI)
  SET(MPI_VERSION "${MPI_C_VERSION_MAJOR}.${MPI_C_VERSION_MINOR}")

  MESSAGE(STATUS "[ PARTENSOR ] MPI_FOUND                     : ${MPI_FOUND}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_VERSION                   : ${MPI_VERSION}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_C_FOUND                   : ${MPI_C_FOUND}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_CXX_FOUND                 : ${MPI_CXX_FOUND}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_C_COMPILE_OPTIONS         : ${MPI_C_COMPILE_OPTIONS}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_CXX_COMPILE_OPTIONS       : ${MPI_CXX_COMPILE_OPTIONS}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_C_COMPILE_DEFINITIONS     : ${MPI_C_COMPILE_DEFINITIONS}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_CXX_COMPILE_DEFINITIONS   : ${MPI_CXX_COMPILE_DEFINITIONS}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_C_INCLUDE_DIRS            : ${MPI_C_INCLUDE_DIRS}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_CXX_INCLUDE_DIRS          : ${MPI_CXX_INCLUDE_DIRS}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_C_LINK_FLAGS              : ${MPI_C_LINK_FLAGS}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_CXX_LINK_FLAGS            : ${MPI_CXX_LINK_FLAGS}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_C_LIBRARIES               : ${MPI_C_LIBRARIES}")
  MESSAGE(STATUS "[ PARTENSOR ] MPI_CXX_LIBRARIES             : ${MPI_CXX_LIBRARIES}")
  MESSAGE(STATUS "[ PARTENSOR ] MPIEXEC_EXECUTABLE            : ${MPIEXEC_EXECUTABLE}")
  MESSAGE(STATUS "[ PARTENSOR ] MPIEXEC_NUMPROC_FLAG          : ${MPIEXEC_NUMPROC_FLAG}")
  MESSAGE(STATUS "[ PARTENSOR ] MPIEXEC_MAX_NUMPROCS          : ${MPIEXEC_MAX_NUMPROCS}")
  MESSAGE(STATUS "[ PARTENSOR ] Setting MPI include and library directories" )
  MESSAGE(STATUS "[ PARTENSOR ] Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

  LINK_DIRECTORIES(${MPI_LIBRARY_DIR})
ENDIF (NOT EXISTS ${MPI_ROOT_DIR})
MESSAGE(STATUS "[ PARTENSOR ] Config MPI END...")
########################################################################################################################

INCLUDE_DIRECTORIES(${INCLUDE_DIR_NCOM})
INCLUDE_DIRECTORIES(${INCLUDE_DIR_TUC})
INCLUDE_DIRECTORIES(${BOOST_ROOT}/include)
INCLUDE_DIRECTORIES(${EIGEN_ROOT_DIR})
INCLUDE_DIRECTORIES(${MPI_ROOT_DIR}/include)
INCLUDE_DIRECTORIES(${SPDLOG_ROOT_DIR}/include)

########################################################################################################################
# TBB (Threading Building Blocks) Libraries && Google Libraries (Test - Benchmark)
########################################################################################################################
IF(DEFINED DEVEL_BUILD)
  INCLUDE(ConfigGoogleLibs)
  INCLUDE(ConfigTBB)
ENDIF(DEFINED DEVEL_BUILD)
########################################################################################################################

########################################################################################################################
###    SUB DIRECTORIES
########################################################################################################################
###    NCOM
IF(EXISTS "${TEST_DIR}" AND DEFINED DEVEL_BUILD)
  ADD_SUBDIRECTORY(test/ncom/Cpd)
  ADD_SUBDIRECTORY(test/ncom/CpdDimTree)
  ADD_SUBDIRECTORY(test/ncom/CpdVsDimTree)
  ADD_SUBDIRECTORY(test/ncom/DataGeneration)
  ADD_SUBDIRECTORY(test/ncom/dimTree)
  ADD_SUBDIRECTORY(test/ncom/ReadWrite)
  ADD_SUBDIRECTORY(test/ncom/KhatriRao)
  ADD_SUBDIRECTORY(test/ncom/Kronecker)
  ADD_SUBDIRECTORY(test/ncom/Matricization)
  ADD_SUBDIRECTORY(test/ncom/CWiseProd)
  ADD_SUBDIRECTORY(test/ncom/InitMPI)
  # ADD_SUBDIRECTORY(test/ncom/TerminationConditions)
  # ADD_SUBDIRECTORY(test/ncom/TensorKhatriRaoProduct)
  # ADD_SUBDIRECTORY(test/ncom/OpenMP_TBB)
  # ADD_SUBDIRECTORY(test/ncom/Server/KhatriRao)
  # ADD_SUBDIRECTORY(test/ncom/TensorSize)
  # ADD_SUBDIRECTORY(test/ncom/RowMajorMatrix)
  # ADD_SUBDIRECTORY(test/ncom/MPIntf)
  ###    TUC
  # IF(EXISTS "test/tuc")
  # # ADD_SUBDIRECTORY(test/tuc/dimTree)
  #   INCLUDE_DIRECTORIES(${INCLUDE_DIR_TUC}/openmp )
  #   ADD_SUBDIRECTORY(test/tuc/CPD)
  #   ADD_SUBDIRECTORY(test/tuc/ReadWrite)
  #   ADD_SUBDIRECTORY(test/tuc/KhatriRaoProduct)
  # ENDIF()
  # ADD_SUBDIRECTORY(test/tuc/MPIGTC)
  # ADD_SUBDIRECTORY(test/tuc/GTC)
  ### DOXYGEN
  ADD_SUBDIRECTORY(documentation)
  ADD_SUBDIRECTORY(documentation/doxygen/examples)
ELSE(EXISTS "${TEST_DIR}" AND DEFINED DEVEL_BUILD)
  ADD_SUBDIRECTORY(test)
ENDIF(EXISTS "${TEST_DIR}" AND DEFINED DEVEL_BUILD)
########################################################################################################################

########################################################################################################################
###     Get build compiler information
########################################################################################################################
SET(compiler.vendor              ${CMAKE_CXX_COMPILER_ID})
SET(compiler.version             ${CMAKE_CXX_COMPILER_VERSION})
SET(compiler.target.architecture ${CMAKE_HOST_SYSTEM_PROCESSOR})

ADD_DEFINITIONS(
   -DCOMPILER_VENDOR="${compiler.vendor}"
   -DCOMPILER_VERSION="${compiler.version}"
   -DCOMPILER_TARGET_ARCHITECTURE="${compiler.target.architecture}"
)
########################################################################################################################

########################################################################################################################
###    CMAKE ATTRIBUTES
########################################################################################################################
MESSAGE(STATUS "[ PARTENSOR ] >>>>>>>>>>>>>>>>>>>>>>>> CMAKE ATTRIBUTES >>>>>>>>>>>>>>>>>>>>>>>>")
MESSAGE(STATUS "[ PARTENSOR ] CMake minimum required version : <${CMAKE_MINIMUM_REQUIRED_VERSION}>")
MESSAGE(STATUS "[ PARTENSOR ] CMake runtime version          : <${CMAKE_VERSION}>")
MESSAGE(STATUS "[ PARTENSOR ] CPP_LIBS                       : <${CPP_LIBS}>")
MESSAGE(STATUS "[ PARTENSOR ] CMAKE_CURRENT_SOURCE_DIR       : <${CMAKE_CURRENT_SOURCE_DIR}>")
MESSAGE(STATUS "[ PARTENSOR ] <<<<<<<<<<<<<<<<<<<<<<<< CMAKE ATTRIBUTES <<<<<<<<<<<<<<<<<<<<<<<<")
########################################################################################################################

MESSAGE(STATUS "[ PARTENSOR ] C++ Compiler:     {vendor=\"${compiler.vendor}\", version=\"${compiler.version}\", target.architecture=\"${compiler.target.architecture}\"}")
MESSAGE(STATUS "[ PARTENSOR ] Build Platform:   {hostname=\"${platform.hostname}\", os.name=\"${platform.os.name}\", processor.description=\"${platform.processor.description}\", processor.cores=\"${platform.processor.cores}\", memory.size=\"${platform.memory.size}MB\"}")
