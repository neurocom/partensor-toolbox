#--- requires cuda v. 11.0 or later (in order to add -std=c++17)---.

nvcc --gpu-architecture=sm_75 -lcudart -lcuda -lcublas ../../include/CUDA/add_vectors.cu -O3 -c -o ../../bin/add_vectors.o

# nvcc -I${CPP_LIBS}/mpi/sources/openmpi-4.0.1/include  -L${CPP_LIBS}/mpi/sources/openmpi-4.0.1/lib\
#    -I${Boost_LIBRARIES} -L${Boost_LIBRARIES} \
#    -I${CPP_LIBS}/eigen \
#    -I${CPP_LIBS}/spdlog/include \
#    --gpu-architecture=sm_75 -lcudart -lcuda -lcublas -lmpi -std=c++17\
#     -I../../include/ -I../../include/CUDA ../../bin/add_vectors.o test_batched_mttkrp.cpp -Xcompiler "-pthread -fopenmp -O3 -DEIGEN_NO_DEBUG -march=native -mtune=native" -o ../../bin/test_batched_mttkrp

nvcc -I${CPP_LIBS}/mpi/sources/openmpi-4.0.1/include  -L${CPP_LIBS}/mpi/sources/openmpi-4.0.1/lib\
    -I${Boost_LIBRARIES} -L${Boost_LIBRARIES} \
    -I${CPP_LIBS}/eigen \
    -I${CPP_LIBS}/spdlog/include \
    --gpu-architecture=sm_75 -lcudart -lcuda -lcublas -lmpi -std=c++17\
    -I../../include/ -I../../include/CUDA ../../bin/add_vectors.o test_batched_mttkrp.cpp -Xcompiler "-pthread -fopenmp -O3 -DEIGEN_NO_DEBUG -march=native -mtune=native" -o ../../bin/cpd_cuda

# nvcc -I${CPP_LIBS}/mpi/sources/openmpi-4.0.1/include  -L${CPP_LIBS}/mpi/sources/openmpi-4.0.1/lib\
#     -I${Boost_LIBRARIES} -L${Boost_LIBRARIES} \
#     -I${CPP_LIBS}/eigen \
#     -I${CPP_LIBS}/spdlog/include \
#     -L/usr/local/cuda-11.2/lib64 --gpu-architecture=sm_75 -lcudart -lcuda -lcublas -lmpi -std=c++17\
#     -I../../include/ -I../../include/CUDA ../../bin/add_vectors.o test_batched_mttkrp.cpp -Xcompiler "-pthread -fopenmp -O3 -DEIGEN_NO_DEBUG -march=native -mtune=native" -o ../../bin/cpd_cuda_full_file

# nvcc -I${CPP_LIBS}/mpi/sources/openmpi-4.0.1/include  -L${CPP_LIBS}/mpi/sources/openmpi-4.0.1/lib\
#     -I${Boost_LIBRARIES} -L${Boost_LIBRARIES} \
#     -I${CPP_LIBS}/eigen \
#     -I${CPP_LIBS}/spdlog/include \
#     -L/usr/local/cuda-11.2/lib64 --gpu-architecture=sm_75 -lcudart -lcuda -lcublas -lmpi -std=c++17\
#     -I../../include/ -I../../include/CUDA ../../bin/add_vectors.o test_batched_mttkrp.cpp -Xcompiler "-pthread -fopenmp -O3 -DEIGEN_NO_DEBUG -march=native -mtune=native" -o ../../bin/cpd_cuda_file

# nvcc -I${CPP_LIBS}/mpi/sources/openmpi-4.0.1/include  -L${CPP_LIBS}/mpi/sources/openmpi-4.0.1/lib\
#     -I${Boost_LIBRARIES} -L${Boost_LIBRARIES} \
#     -I${CPP_LIBS}/eigen \
#     -I${CPP_LIBS}/spdlog/include \
#     --gpu-architecture=sm_75 -lcudart -lcuda -lcublas -lmpi -std=c++17\
#     -I../../include/ -I../../include/CUDA ../../bin/add_vectors.o test_batched_mttkrp.cpp -Xcompiler "-pthread -fopenmp -O3 -DEIGEN_NO_DEBUG -march=native -mtune=native" -o ../../bin/test_mttkrp
