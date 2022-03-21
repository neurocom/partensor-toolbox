/*--------------------------------------------------------------------------------------------------*/
/* 					Function for the creation of the 3 dimensional Communication grid				*/
/*    																							   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 6/4/2017                                             											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"
#include "Eigen/Dense"

using namespace Eigen;

//	<--------------------------------------		Create Grid		--------------------------------------->	//
void Create_Grid(MPI_Comm *grid_comm, MPI_Comm *column_comm, MPI_Comm *row_comm, int *my_rank, int *column_rank, int *row_rank, int *coordinates, int p_A, int p_B){

	int dim_sizes[2], wrap_around[2];		// |
	int free_coords[2];									// | Variables for the Communication Grid
	int reorder = 1;										// |

	dim_sizes[0] = p_A;	dim_sizes[1] = p_B;
	wrap_around[0] = wrap_around[1] = 1;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sizes, wrap_around, reorder, grid_comm);
	MPI_Comm_rank(*grid_comm, my_rank);
	MPI_Cart_coords(*grid_comm, *my_rank, 2, coordinates);

	free_coords[0] = 0; free_coords[1] = 1;
	MPI_Cart_sub(*grid_comm, free_coords, column_comm);

	free_coords[1] = 0; free_coords[0] = 1;
	MPI_Cart_sub(*grid_comm, free_coords, row_comm);

	MPI_Comm_rank(*column_comm, column_rank);
	MPI_Comm_rank(*row_comm, row_rank);

}
