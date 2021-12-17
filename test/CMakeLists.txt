CMAKE_MINIMUM_REQUIRED(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

Project(examples)

SET(TEST_SRC_FILE_NAMES QuickStart_example
                        cpd
                        cpd_mpi
                        cpd_omp##
                        cpd_file
                        cpd_mpi_file
                        cpd_omp_file##
                        cpd_options
                        cpd_mpi_options
                        cpd_omp_options##
                        cpd_options_file
                        cpd_mpi_options_file
                        cpd_omp_options_file##
                        cpd_factorsinit
                        cpd_mpi_factorsinit
                        cpd_omp_factorsinit##
                        cpd_factorsinit_file
                        cpd_mpi_factorsinit_file
                        cpd_omp_factorsinit_file##
                        cpd_full
                        cpd_mpi_full
                        cpd_omp_full##
                        cpd_full_file
                        cpd_mpi_full_file
                        cpd_omp_full_file##
                        cpd_dimtrees
                        cpd_dimtrees_mpi##
                        cpd_dimtrees_file
                        cpd_dimtrees_mpi_file##
                        cpd_dimtrees_options
                        cpd_dimtrees_mpi_options##
                        cpd_dimtrees_options_file
                        cpd_dimtrees_mpi_options_file##
                        cpd_dimtrees_factorsinit
                        cpd_dimtrees_mpi_factorsinit##
                        cpd_dimtrees_factorsinit_file
                        cpd_dimtrees_mpi_factorsinit_file##
                        cpd_dimtrees_full
                        cpd_dimtrees_mpi_full##
                        cpd_dimtrees_full_file
                        cpd_dimtrees_mpi_full_file##
                        gtc
                        gtc_mpi
                        gtc_omp##
                        gtc_file
                        gtc_mpi_file
                        gtc_omp_file##
                        gtc_factorsinit
                        gtc_mpi_factorsinit
                        gtc_omp_factorsinit##
                        gtc_stochastic
                        gtc_stochastic_mpi
                        gtc_stochastic_omp##
                        cwiseprod
                        khatrirao
                        kronecker
                        matricization
                        matrix_generation
                        random_tensor_generation
                        cpd_gen
                        tensor_generation
                        matr_tensor_generation
                        factors_gen
                        make_tensor
                        write
                        read
                        )

########################################################################################################################
###      SET Variables
########################################################################################################################
STRING(TOUPPER ${PROJECT_NAME} PROJECT_NAME_TOUPPER)  
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] Processing in ${CMAKE_BUILD_TYPE} mode.")
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] SOURCE DIR     ==> ${CMAKE_CURRENT_LIST_DIR}")
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] HEADER DIR     ==> ${INCLUDE_DIR_NCOM}")
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] BOOST LIB DIR  ==> ${Boost_LIBRARY_DIR}")
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] BOOST LIBS     ==> ${Boost_LIBRARIES}")
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] MPI LIBS       ==> ${MPI_CXX_LIBRARIES}")
MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] EIGEN_ROOT     ==> " ${EIGEN_ROOT_DIR})
########################################################################################################################

########################################################################################################################
### OpenMP
########################################################################################################################
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
########################################################################################################################
IF(EXISTS "${TEST_DIR}" AND DEFINED DEVEL_BUILD)
    SET(LIBRARIES_TO_LINK   ${MPI_CXX_LIBRARIES}
                            ${GOOGLETEST_LIBRARIES}
                            ${GOOGLEBENCHMARK_LIBRARIES} 
                            ${CMAKE_THREAD_LIBS_INIT} 
                            ${TBB_LIBRARIES}
                            ${Boost_LIBRARIES}
                            OpenMP::OpenMP_CXX)
ELSE(EXISTS "${TEST_DIR}" AND DEFINED DEVEL_BUILD)
    SET(LIBRARIES_TO_LINK   ${MPI_CXX_LIBRARIES}
                            ${Boost_LIBRARIES}
                            OpenMP::OpenMP_CXX)
ENDIF(EXISTS "${TEST_DIR}" AND DEFINED DEVEL_BUILD)
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
########################################################################################################################
### CREATE EXECUTABLE FROM SOURCE AND HEADER FILES
########################################################################################################################
INCLUDE_DIRECTORIES(${INCLUDE_DIR_NCOM})
FOREACH(TEST IN LISTS TEST_SRC_FILE_NAMES)

    MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] Creating executable for ${TEST}.cpp")

    ADD_EXECUTABLE(${TEST} ${CMAKE_CURRENT_LIST_DIR}/${TEST}.cpp)
    TARGET_INCLUDE_DIRECTORIES(${TEST} PUBLIC ${INCLUDE_DIR_NCOM})
    TARGET_LINK_LIBRARIES(${TEST} ${LIBRARIES_TO_LINK})

    ######################
    ### EXECUTABLE DIR ###
    ######################
    SET_TARGET_PROPERTIES(
        ${TEST}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
    )
ENDFOREACH()