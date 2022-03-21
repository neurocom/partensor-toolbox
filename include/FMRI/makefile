MPICC = /SDKS/CPP_LIBS/mpi/bin/mpiCC
CC = /usr/local/bin/g++
#DEBUG = -g
CXXFLAGS = -c -O3 -I /SDKS/CPP_LIBS/eigen

#LFLAGS = -Wall# $(DEBUG)
OBJS = ntf_mpi_eigen.o compute_grid_dimensions.o dis_count.o create_grid.o compute_svd.o g_lambda.o get_objective_value.o get_objective_value_accel.o line_search_accel.o normalize.o nesterov_matrix_nnls.o read_data.o read_from_file.o set_info.o update_alpha.o update_A_n_Z_factors.o update_S_factor.o
DEPS = ntf_functions.h

all: ntf

ntf: $(OBJS)
	$(MPICC) $(OBJS) -o ntf

ntf_mpi_eigen.o: ntf_mpi_eigen.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) ntf_mpi_eigen.cpp

compute_grid_dimensions.o: compute_grid_dimensions.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) compute_grid_dimensions.cpp

create_grid.o: create_grid.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) create_grid.cpp

compute_svd.o: compute_svd.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) compute_svd.cpp
			
dis_count.o: dis_count.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) dis_count.cpp
		
g_lambda.o: g_lambda.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) g_lambda.cpp
	
get_objective_value.o: get_objective_value.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) get_objective_value.cpp
	
get_objective_value_accel.o: get_objective_value_accel.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) get_objective_value_accel.cpp

line_search_accel.o: line_search_accel.cpp get_objective_value_accel.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) line_search_accel.cpp
	
normalize.o: normalize.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) normalize.cpp
	
nesterov_matrix_nnls.o: nesterov_matrix_nnls.cpp compute_svd.cpp update_alpha.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) nesterov_matrix_nnls.cpp

read_data.o: read_data.cpp dis_count.cpp read_from_file.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) read_data.cpp
	
read_from_file.o: read_from_file.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) read_from_file.cpp
			
set_info.o: set_info.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) set_info.cpp
		
update_alpha.o: update_alpha.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) update_alpha.cpp

update_S_factor.o: update_S_factor.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) update_S_factor.cpp
	
update_A_n_Z_factors.o: update_A_n_Z_factors.cpp nesterov_matrix_nnls.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) update_A_n_Z_factors.cpp

create_data:
	$(CC) -O3 create_data.cpp -o create_data

clean:
	rm *.o *~ ntf create_data
	
clean_data:
	rm create_data
