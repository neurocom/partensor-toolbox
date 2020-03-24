# The MIT License (MIT)
#
# Copyright (c) 2017 Neurocom S.A.
#
# FindTBB
# -------
#
# Find TBB include directories and libraries.
#
# Usage:
#  INCLUDE(<path>/ConfigTBB.cmake)
#
# Users must specify the TBB_ROOT_DIR
#
# Variables set by module
# * TBB_INCLUDE_DIR       - The directory that contains the TBB headers files.
# * TBB_LIBRARY           - The directory that contains the TBB library files.
# * TBB_FOUND             - Set to false, or undefined, if we haven’t found, or
#                           don’t want to use TBB.
# * TBB_<component>_FOUND - If False, optional <c"omponent> part of TBB sytem is
#                           not available.
# * TBB_VERSION           - The full version string
# * TBB_VERSION_MAJOR     - The major version
# * TBB_VERSION_MINOR     - The minor version
# * TBB_INTERFACE_VERSION - The interface version number defined in
#                           tbb/tbb_stddef.h.
# * TBB_<library>_LIBRARY_RELEASE - The path of the TBB release version of
#                           <library>, where <library> may be tbb, tbb_debug,
#                           tbbmalloc, tbbmalloc_debug, tbb_preview, or
#                           tbb_preview_debug.
# * TBB_<library>_LIBRARY_DEGUG - The path of the TBB release version of
#                           <library>, where <library> may be tbb, tbb_debug,
#                           tbbmalloc, tbbmalloc_debug, tbb_preview, or
#                           tbb_preview_debug.
#
# The following varibles should be used to build and link with TBB:
#
# * TBB_INCLUDE_DIRS        - The include directory for TBB.
# * TBB_LIBRARIES           - The libraries to link against to use TBB.
# * TBB_LIBRARIES_RELEASE   - The release libraries to link against to use TBB.
# * TBB_LIBRARIES_DEBUG     - The debug libraries to link against to use TBB.
# * TBB_DEFINITIONS         - Definitions to use when compiling code that uses
#                             TBB.
# * TBB_DEFINITIONS_RELEASE - Definitions to use when compiling release code that
#                             uses TBB.
# * TBB_DEFINITIONS_DEBUG   - Definitions to use when compiling debug code that
#                             uses TBB.
#

MESSAGE(STATUS "[NEURO<ConfigTBB>] Start of ConfigTBB.cmake")

SET(PSTLROOT           ${CPP_LIBS}/tbb/pstl_current  CACHE PATH "PSTL Root Folder"         FORCE)
SET(PSTL_TARGET_ARCH   intel64                       CACHE PATH "PSTL Target Architecture" FORCE)
SET(TBBROOT            ${CPP_LIBS}/tbb/tbb_current   CACHE PATH "TBB Root Folder"          FORCE)
SET(TBB_TARGET_ARCH    intel64                       CACHE PATH "TBB Target Architecture"  FORCE)
SET(TBB_ROOT_DIR       ${TBBROOT}                    CACHE PATH "TBB_Root Folder"          FORCE)

MESSAGE(STATUS     "[NEURO<ConfigTBB>] Settting TBB_ROOT_DIR         : ${TBB_ROOT_DIR}")
IF (NOT EXISTS ${TBB_ROOT_DIR})
  	MESSAGE(SEND_ERROR "[NEURO<ConfigTBB>] Cannot find TBB_ROOT_DIR    : ${TBB_ROOT_DIR}")
ELSE (NOT EXISTS ${TBB_ROOT_DIR})
	MESSAGE(STATUS "[NEURO<ConfigTBB>] Using TBB_ROOT_DIR            : ${TBB_ROOT_DIR}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] Processing                    : ${CMAKE_CMAKE_SRC}/ConfigTBB.cmake")
	# INCLUDE(ConfigTBB)

	IF (NOT TBB_ROOT_DIR)
		MESSAGE(FATAL_ERROR "TBB_ROOT_DIR must be set")
	ENDIF (NOT TBB_ROOT_DIR)

	IF (WIN32)
		SET(TBB_LIB_SUFFIX ".lib")
		SET(TBB_LIB_PREFIX "")
	ELSEIF (UNIX)
		SET(TBB_LIB_PREFIX ".lib")
		IF (APPLE)
			SET(TBB_LIB_SUFFIX ".dylib")
		ELSE (APPLE)
			SET(TBB_LIB_SUFFIX ".so")
		ENDIF (APPLE)

		SET(TBB_LIB_PREFIX "lib")
	ELSE (WIN32)
		MESSAGE(SEND_ERROR "[NEURO<ConfigTBB>] Unknown OS")
	ENDIF(WIN32)

	MESSAGE(STATUS "[NEURO<ConfigTBB>] Using TBB_LIB_PREFIX : ${TBB_LIB_PREFIX}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] Using TBB_LIB_SUFFIX : ${TBB_LIB_SUFFIX}")

	IF (NOT DEFINED TBB_USE_DEBUG_BUILD)
		IF (CMAKE_BUILD_TYPE MATCHES "(Debug|DEBUG|debug|RelWithDebInfo|RELWITHDEBINFO|relwithdebinfo)")
			SET(TBB_BUILD_TYPE DEBUG)
	ELSE()
		SET(TBB_BUILD_TYPE RELEASE)
	ENDIF()
	ELSEIF(TBB_USE_DEBUG_BUILD)
		SET(TBB_BUILD_TYPE DEBUG)
	ELSE()
	SET(TBB_BUILD_TYPE RELEASE)
	ENDIF()

	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_BUILD_TYPE : ${TBB_BUILD_TYPE}")

	SET(TBB_INCLUDE_DIR ${TBB_ROOT_DIR}/include)
	SET(TBB_LIBRARY_DIR ${TBB_ROOT_DIR}/lib)
	SET(TBB_INCLUDE_DIRS ${TBB_ROOT_DIR}/include)

	IF (TBB_INCLUDE_DIRS)
		FILE(READ "${TBB_INCLUDE_DIRS}/tbb/tbb_stddef.h" _tbb_version_file)
	STRING(REGEX REPLACE ".*#define TBB_VERSION_MAJOR ([0-9]+).*" "\\1"
									TBB_VERSION_MAJOR "${_tbb_version_file}")
		STRING(REGEX REPLACE ".*#define TBB_VERSION_MINOR ([0-9]+).*" "\\1"
							TBB_VERSION_MINOR "${_tbb_version_file}")
	STRING(REGEX REPLACE ".*#define TBB_INTERFACE_VERSION ([0-9]+).*" "\\1"
									TBB_INTERFACE_VERSION "${_tbb_version_file}")
	SET(TBB_VERSION "${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}")
	ENDIF()

	SET(TBB_SEARCH_COMPOMPONENTS tbbmalloc_proxy tbbmalloc tbb)

	##################################
	# Find TBB components
	##################################

	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_VERSION : ${TBB_VERSION}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_SEARCH_COMPOMPONENTS : ${TBB_SEARCH_COMPOMPONENTS}")

	# Find each component
	FOREACH (_comp ${TBB_SEARCH_COMPOMPONENTS})
		MESSAGE(STATUS "[NEURO<ConfigTBB>] _comp : ${_comp}")

		IF (EXISTS ${TBB_ROOT_DIR}/lib/${TBB_LIB_PREFIX}${_comp}_debug${TBB_LIB_SUFFIX})
		SET(TBB_${_comp}_LIBRARY_DEBUG ${TBB_ROOT_DIR}/lib/${TBB_LIB_PREFIX}${_comp}_debug${TBB_LIB_SUFFIX})
		SET(TBB_FOUND 1)
	#		 MESSAGE(STATUS " [NEURO<ConfigTBB>] ${TBB_ROOT_DIR}/lib/${TBB_LIB_PREFIX}${_comp}_debug${TBB_LIB_SUFFIX} ==> FOUND")
		ELSE ()
		MESSAGE(STATUS " [NEURO<ConfigTBB>] ${TBB_ROOT_DIR}/lib/${TBB_LIB_PREFIX}${_comp}_debug${TBB_LIB_SUFFIX} ==> NOT FOUND")
	ENDIF ()

	IF (EXISTS ${TBB_ROOT_DIR}/lib/${TBB_LIB_PREFIX}${_comp}${TBB_LIB_SUFFIX})
		SET(TBB_${_comp}_LIBRARY_RELEASE ${TBB_ROOT_DIR}/lib/${TBB_LIB_PREFIX}${_comp}${TBB_LIB_SUFFIX})
		SET(TBB_FOUND 1)
	#		 MESSAGE(STATUS "[NEURO<ConfigTBB>] ${TBB_ROOT_DIR}/lib/${TBB_LIB_PREFIX}${_comp}${TBB_LIB_SUFFIX} ==> FOUND")
		ELSE ()
			MESSAGE(STATUS "[NEURO<ConfigTBB>] ${TBB_ROOT_DIR}/lib/${TBB_LIB_PREFIX}${_comp}${TBB_LIB_SUFFIX} ==> NOT FOUND")
	ENDIF ()

		IF (TBB_${_comp}_LIBRARY_DEBUG)
		LIST(APPEND TBB_LIBRARIES_DEBUG "${TBB_${_comp}_LIBRARY_DEBUG}")
	ENDIF()

	IF (TBB_${_comp}_LIBRARY_RELEASE)
		LIST(APPEND TBB_LIBRARIES_RELEASE "${TBB_${_comp}_LIBRARY_RELEASE}")
	ENDIF()

	IF (TBB_${_comp}_LIBRARY_${TBB_BUILD_TYPE} AND NOT TBB_${_comp}_LIBRARY)
		SET(TBB_${_comp}_LIBRARY "${TBB_${_comp}_LIBRARY_${TBB_BUILD_TYPE}}")
	ENDIF()

	IF (TBB_${_comp}_LIBRARY AND EXISTS "${TBB_${_comp}_LIBRARY}")
		SET(TBB_${_comp}_FOUND TRUE)
	ELSE()
			SET(TBB_${_comp}_FOUND FALSE)
	ENDIF()

	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_${_comp}_LIBRARY_DEBUG : ${TBB_${_comp}_LIBRARY_DEBUG}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_${_comp}_LIBRARY_RELEASE : ${TBB_${_comp}_LIBRARY_RELEASE}")

	# Mark internal variables as advanced
	mark_as_advanced(TBB_${_comp}_LIBRARY_RELEASE)
	mark_as_advanced(TBB_${_comp}_LIBRARY_DEBUG)
	mark_as_advanced(TBB_${_comp}_LIBRARY)

	ENDFOREACH()

	SET(TBB_LIBRARIES "${TBB_LIBRARIES_${TBB_BUILD_TYPE}}")

	##################################
	# Set compile flags
	##################################

	set(TBB_DEFINITIONS_RELEASE "")
	set(TBB_DEFINITIONS_DEBUG "-DTBB_USE_DEBUG=1")
	set(TBB_DEFINITIONS "${TBB_DEFINITIONS_${TBB_BUILD_TYPE}}")

	IF (DEBUG)
		SET (TBB_LIBRARIES ${TBB_LIBRARIES_DEBUG})
	ELSE (DEBUG)
		SET (TBB_LIBRARIES ${TBB_LIBRARIES_RELEASE})
	ENDIF(DEBUG)

	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_FOUND                     : ${TBB_FOUND}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_VERSION                   : ${TBB_VERSION}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_INCLUDE_DIRS              : ${TBB_INCLUDE_DIRS}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_LIBRARIES                 : ${TBB_LIBRARIES}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_LIBRARIES_RELEASE         : ${TBB_LIBRARIES_RELEASE}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_LIBRARIES_DEBUG           : ${TBB_LIBRARIES_DEBUG}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_DEFINITIONS               : ${TBB_DEFINITIONS}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_DEFINITIONS_RELEASE       : ${TBB_DEFINITIONS_RELEASE}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_DEFINITIONS_DEBUG         : ${TBB_DEFINITIONS_DEBUG}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_tbb_LIBRARY               : ${TBB_tbb_LIBRARY}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_tbbmalloc_LIBRARY         : ${TBB_tbbmalloc_LIBRARY}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_tbb_LIBRARY_RELEASE       : ${TBB_tbb_LIBRARY_RELEASE}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_tbbmalloc_LIBRARY_RELEASE : ${TBB_tbbmalloc_LIBRARY_RELEASE}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_tbb_LIBRARY_DEGUG         : ${TBB_tbb_LIBRARY_DEGUG}")
	MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_tbbmalloc_LIBRARY_DEGUG   : ${TBB_tbbmalloc_LIBRARY_DEGUG}")

	MESSAGE(STATUS "[NEURO<ConfigTBB>] Setting TBB inlcude and library directories" )
	INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIR})
	LINK_DIRECTORIES(${TBB_LIBRARY_DIR})
		MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_LIBRARY_DIR               : ${TBB_LIBRARY_DIR}")
		MESSAGE(STATUS "[NEURO<ConfigTBB>] TBB_INCLUDE_DIRS              : ${TBB_INCLUDE_DIRS}")

	
	INCLUDE_DIRECTORIES(${PSTLROOT}/include)
	INCLUDE_DIRECTORIES(${TBBROOT}/include)	
ENDIF (NOT EXISTS ${TBB_ROOT_DIR})
MESSAGE(STATUS "[NEURO<ConfigTBB>] Config TBB END...")
