#######################################################
#### Google Libraries
#######################################################
MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] Start of ConfigGoogleLibs.cmake")

SET(GOOGLETEST_ROOT_DIR            ${CPP_LIBS}/googletest)
SET(GOOGLEBENCHMARK_ROOT_DIR       ${CPP_LIBS}/benchmark)

########################################################################################################################
### GOOGLE TEST LIBRARY
########################################################################################################################
MESSAGE( STATUS  "[NEURO<ConfigGoogleLibs>] Configuring GOOGLE TEST" )

find_package(Threads REQUIRED)

SET(GOOGLETEST_ROOT_DIR                                                 ${GOOGLETEST_ROOT_DIR})
MESSAGE(STATUS     "[NEURO<ConfigGoogleLibs>] Settting GOOGLETEST_ROOT_DIR      : " ${GOOGLETEST_ROOT_DIR})
IF (NOT EXISTS ${GOOGLETEST_ROOT_DIR})
  MESSAGE(SEND_ERROR "[NEURO<ConfigGoogleLibs>] Cannot find GOOGLETEST_ROOT_DIR : " ${GOOGLETEST_ROOT_DIR})
ELSE (NOT EXISTS ${GOOGLETEST_ROOT_DIR})
  IF (${DISPLAY_VERSIONS} AND EXISTS ${GOOGLETEST_ROOT_DIR}/CMakeLists.txt)
    FILE(READ ${GOOGLETEST_ROOT_DIR}/CMakeLists.txt _googletest_version_file)
    STRING(REGEX REPLACE ".*GOOGLETEST_VERSION ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1"
                GOOGLETEST_VERSION_WORLD "${_googletest_version_file}")
    SET(GOOGLETEST_VERSION "${GOOGLETEST_VERSION_WORLD}")
  ENDIF (${DISPLAY_VERSIONS} AND EXISTS ${GOOGLETEST_ROOT_DIR}/CMakeLists.txt)
  MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] Using GOOGLETEST_ROOT_DIR         : ${GOOGLETEST_ROOT_DIR}")
  MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] GOOGLETEST_VERSION                : ${GOOGLETEST_VERSION}")
  SET(GOOGLETEST_INCLUDE_DIR                                        ${GOOGLETEST_ROOT_DIR}/include)
  SET(GOOGLETEST_LIBRARY_DIR                                        ${GOOGLETEST_ROOT_DIR}/lib)
  MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] GOOGLETEST_INCLUDE_DIR  ==>       " ${GOOGLETEST_INCLUDE_DIR})
  MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] GOOGLETEST_LIBRARY_DIR  ==>       " ${GOOGLETEST_LIBRARY_DIR})
  SET(GOOGLETEST_LIBRARIES                                          ${GOOGLETEST_LIBRARY_DIR}/libgmock.a
                                                                    ${GOOGLETEST_LIBRARY_DIR}/libgmock_main.a
                                                                    ${GOOGLETEST_LIBRARY_DIR}/libgtest.a
                                                                    ${GOOGLETEST_LIBRARY_DIR}/libgtest_main.a)

  INCLUDE_DIRECTORIES(${GOOGLETEST_INCLUDE_DIR})
ENDIF (NOT EXISTS ${GOOGLETEST_ROOT_DIR})
MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] Config GOOGLE TEST END...")
########################################################################################################################

########################################################################################################################
### GOOGLE BENCHMARK LIBRARY
########################################################################################################################
MESSAGE( STATUS  "[NEURO<ConfigGoogleLibs>] Configuring GOOGLE BENCHMARK" )

SET(GOOGLEBENCHMARK_ROOT_DIR                                               ${GOOGLEBENCHMARK_ROOT_DIR})
MESSAGE(STATUS     "[NEURO<ConfigGoogleLibs>] Settting GOOGLEBENCHMARK_ROOT_DIR : "    ${GOOGLEBENCHMARK_ROOT_DIR})
IF (NOT EXISTS ${GOOGLEBENCHMARK_ROOT_DIR})
  MESSAGE(SEND_ERROR "[NEURO<ConfigGoogleLibs>] Cannot find GOOGLETEST_ROOT_DIR : "    ${GOOGLEBENCHMARK_ROOT_DIR})
ELSE (NOT EXISTS ${GOOGLEBENCHMARK_ROOT_DIR})
  SET(GOOGLEBENCHMARK_CONFIG_CMAKE                                         ${GOOGLEBENCHMARK_ROOT_DIR}/cmake/GetGitVersion.cmake)
  MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] Processing                        : "      ${GOOGLEBENCHMARK_CONFIG_CMAKE})
  # Read the git tags to determine the project version
  IF (EXISTS ${GOOGLEBENCHMARK_CONFIG_CMAKE})
    INCLUDE(${GOOGLEBENCHMARK_CONFIG_CMAKE})

    IF (${DISPLAY_VERSIONS})
      get_git_version(GOOGLEBENCHMARK_VERSION)
      # Tell the user what versions we are using
      STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" VERSION ${GOOGLEBENCHMARK_VERSION})
    ENDIF (${DISPLAY_VERSIONS})
  ENDIF (EXISTS ${GOOGLEBENCHMARK_CONFIG_CMAKE})
  IF (${DISPLAY_VERSIONS})
    MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] GOOGLEBENCHMARK_VERSION           : "      ${VERSION})
  ENDIF (${DISPLAY_VERSIONS})

  MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] Using GOOGLEBENCHMARK_ROOT_DIR    : "      ${GOOGLEBENCHMARK_ROOT_DIR})
  SET(GOOGLEBENCHMARK_INCLUDE_DIR                                          ${GOOGLEBENCHMARK_ROOT_DIR}/include)
  SET(GOOGLEBENCHMARK_LIBRARY_DIR                                          ${GOOGLEBENCHMARK_ROOT_DIR}/lib)
  MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] GOOGLEBENCHMARK_INCLUDE_DIR  ==>  "        ${GOOGLEBENCHMARK_INCLUDE_DIR})
  MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] GOOGLEBENCHMARK_LIBRARY_DIR  ==>  "        ${GOOGLEBENCHMARK_LIBRARY_DIR})
  SET(GOOGLEBENCHMARK_LIBRARIES                                            ${GOOGLEBENCHMARK_LIBRARY_DIR}/libbenchmark.a
                                                                           ${GOOGLEBENCHMARK_LIBRARY_DIR}/libbenchmark_main.a)

ENDIF (NOT EXISTS ${GOOGLEBENCHMARK_ROOT_DIR})
MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] Config GOOGLE BENCHMARK END...")
########################################################################################################################

INCLUDE_DIRECTORIES(${GOOGLEBENCHMARK_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${GOOGLETEST_INCLUDE_DIR})
MESSAGE(STATUS "[NEURO<ConfigGoogleLibs>] End of ConfigGoogleLibs.cmake")