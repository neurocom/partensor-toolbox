#######################################################
#### MPI
#######################################################
MESSAGE(STATUS " [NEURO<ConfigMPI>] Start of ConfigMPI.cmake")

IF (NOT CPP_LIBS)
    MESSAGE(FATAL_ERROR " [NEURO] CPP_LIBS must be set...")
ENDIF (NOT CPP_LIBS)

SET(MPI_ROOT_DIR         ${MPI_ROOT_DIR})

IF (NOT MPI_ROOT_DIR)
	 MESSAGE(FATAL_ERROR " [NEURO] MPI_ROOT_DIR must be set")
ENDIF (NOT MPI_ROOT_DIR)


IF (WIN32)
	SET(MPI_LIB_SUFFIX ".lib")
	SET(MPI_LIB_PREFIX "")
ELSEIF (UNIX)
	SET(MPI_LIB_PREFIX ".lib")
	IF (APPLE)
		SET(MPI_LIB_SUFFIX ".dylib")
	ELSE (APPLE)
		SET(MPI_LIB_SUFFIX ".so")
	ENDIF (APPLE)
	SET(MPI_LIB_PREFIX "lib")
ELSE (WIN32)
	MESSAGE(SEND_ERROR "[NEURO<ConfigMPI>] Unknown OS")
ENDIF(WIN32)

MESSAGE(STATUS " [NEURO<ConfigMPI>] Using MPI_LIB_PREFIX : ${MPI_LIB_PREFIX}")
MESSAGE(STATUS " [NEURO<ConfigMPI>] Using MPI_LIB_SUFFIX : ${MPI_LIB_SUFFIX}")

SET(MPI_INCLUDE_DIR      ${MPI_ROOT_DIR}/include)
SET(MPI_LIBRARY_DIR      ${MPI_ROOT_DIR}/lib)
SET(MPI_BIN_DIR          ${MPI_ROOT_DIR}/bin)

SET(MPI_C_HEADER_DIR     ${MPI_INCLUDE_DIR})
SET(MPI_CXX_HEADER_DIR   ${MPI_INCLUDE_DIR})

SET(MPI_C_COMPILER       ${MPI_BIN_DIR}/mpicc)
SET(MPI_CXX_COMPILER     ${MPI_BIN_DIR}/mpicxx)
SET(MPIEXEC_EXECUTABLE   ${MPI_BIN_DIR}/exec)

##################################
# Find MPI components
##################################

SET(MPI_SEARCH_COMPOMPONENTS mca_common_dstore mca_common_dstore mca_common_monitoring mca_common_ompio mca_common_sm mpi mpi_usempi_ignore_tkr ompitrace open-pal open-rte)
MESSAGE(STATUS " [NEURO<ConfigMPI>] MPI_SEARCH_COMPOMPONENTS : ${MPI_SEARCH_COMPOMPONENTS}")

FOREACH (_comp ${MPI_SEARCH_COMPOMPONENTS})
	MESSAGE(STATUS " [NEURO<ConfigMPI>] _comp : ${_comp}")

  IF (EXISTS ${MPI_ROOT_DIR}/lib/${MPI_LIB_PREFIX}${_comp}${MPI_LIB_SUFFIX})
  	 SET(MPI_${_comp}_LIBRARY_RELEASE ${MPI_ROOT_DIR}/lib/${MPI_LIB_PREFIX}${_comp}${MPI_LIB_SUFFIX})
     SET(MPI_FOUND 1)
  ELSE ()
		 MESSAGE(STATUS " [NEURO<ConfigMPI>] ${MPI_ROOT_DIR}/lib/${MPI_LIB_PREFIX}${_comp}${MPI_LIB_SUFFIX} ==> NOT FOUND")
  ENDIF ()

  IF (MPI_${_comp}_LIBRARY_RELEASE)
  	 LIST(APPEND MPI_LIBRARIES_RELEASE "${MPI_${_comp}_LIBRARY_RELEASE}")
  ENDIF()

  IF (MPI_${_comp}_LIBRARY_${MPI_BUILD_TYPE} AND NOT MPI_${_comp}_LIBRARY)
  	 SET(MPI_${_comp}_LIBRARY "${MPI_${_comp}_LIBRARY_${MPI_BUILD_TYPE}}")
  ENDIF()

  IF (MPI_${_comp}_LIBRARY AND EXISTS "${MPI_${_comp}_LIBRARY}")
  	 SET(MPI_${_comp}_FOUND TRUE)
  ELSE()
		 SET(MPI_${_comp}_FOUND FALSE)
  ENDIF()

  MESSAGE(STATUS " [NEURO<ConfigMPI>] MPI_${_comp}_LIBRARY_RELEASE : ${MPI_${_comp}_LIBRARY_RELEASE}")
ENDFOREACH()

SET(MPI_LIBRARIES "${MPI_LIBRARIES_${MPI_BUILD_TYPE}}")
MESSAGE(STATUS " [NEURO<ConfigMPI>] End of ConfigMPI.cmake")
