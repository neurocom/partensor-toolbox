/*--------------------------------------------------------------------------------------------------------------*/
/* 		    Solves a distributed semi orthogonal and nonnegative matrix factorization  problem in C++				 			*/
/*    (The implementation uses MPI for distributed communication and the Eigen Library for Linear Algebra)   		*/
/*                                                                           																		*/
/* 1 - Create Data: make create_data																																						*/
/*              	./create_data	                                             																		*/
/*                                                                           																		*/
/* 2 - Compile: make							 																																							*/
/*                                                                     									      									*/
/* 3 - Execute: mpirun -np #np pm_onmf  	 	                    		   																					*/
/*																																																							*/
/* A. P. Liavas																																																	*/
/* P. Karakasis																																																	*/
/* 16/3/2020													                                              														*/
/*--------------------------------------------------------------------------------------------------------------*/
#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <string>
#include "ntf_functions.h"
#include <limits>

using namespace std;
using namespace Eigen;

void Write_to_File(int nrows, int ncols, Ref<MatrixXd> Mat, const char *file_name){
	ofstream my_file(file_name, ios::out | ios::binary | ios::trunc);
	if (my_file.is_open()){
		my_file.write((char *) Mat.data(), nrows*ncols*sizeof(double));
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}

void Write_info_to_File(int* R, int* K, int* N, int* M, const char *file_name){
	ofstream my_file(file_name, ios::out | ios::binary | ios::trunc);

	if (my_file.is_open()){
		my_file.write((char *) R, sizeof(int));
		my_file.write((char *) K, sizeof(int));
		my_file.write((char *) N, sizeof(int));
		my_file.write((char *) M, sizeof(int));
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}

/* <----------------------------------------------------------------------------------------------------------> */
/* < ---------------------------------------		MAIN		----------------------------------------------> */
int main(int argc, char **argv){
	int N, M;														    // Dimensions of the matrices
	int K;																	// Number of matrices
	int R;																	// Rank of the factorization
	double c_k;

	const double AO_tol = 1e-3;							// Tolerance for AO Algorithm
	const double delta_1 = 1e-5;						// | Tolerance for Nesterov Algorithm
	const double delta_2 = 1e-5;						// |
	int max_iter = 1000;											  // Maximum Number of iterations

	int my_world_rank,my_rank;							// | The ids of each processor
	int column_rank, row_rank;							// |
	int comm_sz;														// Total number of processors
	int p_A, p_B;														// Number of processor per dimension
	int coordinates[2];

	double f_value, a;													// Objective Value

	// int acc_coeff = 3;										// | Variables
	// int acc_fail = 0;										// | for
	int k_0 = 1;														// | Acceleration step

	double t_end_max, t_end_min;						// | Time variables
	double t_start, t_end, total_t;					// |
	clock_t start_t, end_t ;								// |
	int AO_iter;														// |

	Set_Info(&R, &K, &N, &M, &c_k, "Data_cpp/info.bin");													// Initialize factors' sizes and rank from file

	MPI_Init(&argc, &argv);  																											// | Initialize the MPI execution environment
	MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank); 																// | Determine current running process
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 																			// | Total number of processes

	//	<-----------------------------		Create Communication Grid		------------------------------------>	//
	MPI_Comm grid_comm, column_comm, row_comm;

	Compute_Grid_Dimensions(&p_A, &p_B, &comm_sz, 1);

	Create_Grid(&grid_comm, &column_comm, &row_comm, &my_rank, &column_rank, &row_rank, coordinates, p_A, p_B);

	if (my_rank == 0)
			cout << "R=" << R << ", K=" << K << ", N=" << N << ", M=" << M << ", c_k=" << c_k << ", p_A=" << p_A << ", p_B=" << p_B << endl;

	//	<-----------------------		Define arrays for the Data Submatricess Dimensions		---------------------------->	//
	int n_N[p_A], n_M[p_B];
	int skip_N[p_A], skip_M[p_B];

	//	<-----------------------		Define arrays for Subtensors Dimensions		---------------------------->	//
	Dis_Count(skip_N, n_N, p_A, N, 1);
	Dis_Count(skip_M, n_M, p_B, K*M, 1);

	srand(0);

	MatrixXd A_true = (MatrixXd::Random(N, R) + MatrixXd::Ones(N ,R))/2;			// A~uniform[0,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))

	for (int n=0; n<N; n++){																		// | Meet the orthnormality constraints A'*A = I
			a = A_true.row(n).maxCoeff();																// |
			for (int r=0; r<R; r++){																// |
					if (A_true(n,r) < a)																			// |
							A_true(n,r) = 0;																			// |
			}																												// |
	}																														// |
																															// |
	for (int r=0; r<R; r++){																		// |
			a = A_true.col(r).norm();																		// |
			A_true.col(r) *= 1/a;																				// |
	}																														// |

	MatrixXd S_full = MatrixXd::Random(K*M, R);																// S~uniform[-1,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	MatrixXd X_A_full = A_true * S_full.transpose(); 																		// Create matrix data X_A (concatenation of X_k matrices)

	//  <-------------------------------------------   Create Initial Factors   ----------------------------------------------------->  //

	MatrixXd A_full_init = (MatrixXd::Random(N, R) + MatrixXd::Ones(N,R))/2;	// A_init~uniform[0,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	// Z_init = MatrixXd::Zero(N, R);		                        	//
	MatrixXd S_full_init = MatrixXd::Random(K*M, R);	                        // S_init~uniform[-1,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))


	//	<--------------------------------		Matrix Initializations		--------------------------------------->	//
	MatrixXd A(n_N[row_rank],R), S(n_M[column_rank],R);																				// Factors A, S
	MatrixXd A_old(n_N[row_rank],R), S_old(n_M[column_rank],R);																// Old Factors A, S
	MatrixXd X_A(n_N[row_rank],n_M[column_rank]);																							// Data Matrix X_A

	//	<----------------------		Read Initial Factors and Matrices from file		--------------------------->	//
	
	// A = A_full_init.block(skip_N[row_rank], 0, n_N[row_rank], R);
	// S = S_full_init.block(skip_M[column_rank], 0, n_M[column_rank], R);
	// X_A = X_A_full.block(skip_N[row_rank], skip_M[column_rank], n_N[row_rank], n_M[column_rank]);
	std::cout << "Reading data..." << std::endl;
	Read_Data(A, S, X_A, N, n_N[row_rank], skip_N[row_rank], n_M[column_rank], skip_M[column_rank]);
	
	//	<-------------------------------------------	Begin AO Algorithm		------------------------------------------>	//
	if (my_rank==0)
		 cout << " BEGIN ALGORITHM " << endl;

	//	<----------------------------------------------	Start Timers	------------------------------------------------>	//
	MPI_Barrier(grid_comm);
	t_start = MPI_Wtime();
	start_t = clock();

	AO_iter = 0;

	//	<----------------------------------------	Cost Function Computation		---------------------------------------->	//
	f_value = Get_Objective_Value(S, X_A, A, (c_k / (double) p_B ), grid_comm);
	
	while(1){
		
		AO_iter++;
		A_old = A;

		//	<------------------------		 Print the message		------------------------------->	//
		if (my_rank == 0)
				cout << AO_iter << " -- " << f_value << " -- " <<  endl;

		//	<------------------------		 Update A and Z		------------------------------->	//
		Update_A_n_Z_factors(A, S, X_A, delta_1, delta_2, AO_tol, c_k, p_B, column_comm);

		//	<-----------------------	Normalization step		--------------------------->	//
		Normalize(A, S, row_comm);

		//	<------------------------		 Update S		------------------------------->	//
		Update_S_factor( X_A, A, S, row_comm);

		//	<-----------------------	Cost Function Computation		---------------------->	//
		f_value = Get_Objective_Value( S, X_A, A, (c_k / (double) p_B ), grid_comm);

		//	<-----------------------	Terminating condition		----------------------->	//
		if ( (A - A_old).norm() < AO_tol * A_old.norm() || AO_iter > max_iter )
			 break;

		/*
		//	<-----------------------	Acceleration step		--------------------------->	//
		if (AO_iter>k_0)
			Line_Search_Accel(A_old_N, B_old_N, C_old_N, A, B, C, A_T_A, B_T_B, C_T_C, KhatriRao_BA, X_C, &acc_fail, &acc_coeff, AO_iter, f_value, frob_X, grid_comm, grid_comm, grid_comm, grid_comm);

		A_old_N = A_N; B_old_N = B_N;	C_old_N = C_N;
		*/

	}
	//	<-------------------------------------------	End of AO Algorithm		------------------------------------------->	//

	//	<-----------------------------------------------	End of timers	--------------------------------------------->	//
	t_end = MPI_Wtime();
	end_t = clock();
	total_t = (end_t - start_t);

	t_end -= t_start;
	MPI_Allreduce(&t_end, &t_end_max, 1, MPI_DOUBLE, MPI_MAX, grid_comm);					// Communication through all processes
	MPI_Allreduce(&t_end, &t_end_min, 1, MPI_DOUBLE, MPI_MIN, grid_comm);					// Communication through all processes
	cout.precision(15);

	//	<----------------	Processor 0 prints some	results in the terminal	------------------------->	//
	if (my_rank == 0) {
		cout << " MPI_Wtime_max = " << t_end_max << endl;
		cout << " MPI_Wtime_min = " << t_end_min << endl;
		cout << " CPU time = " << ((float)total_t)/CLOCKS_PER_SEC << endl;
		cout << " AO_iter = " << AO_iter << endl;
		cout << " relative f_value = " << f_value << endl << endl;
	}
	//	<----------------	Processor 0 writes some results in Results.txt	------------------------->	//
	if (my_rank == 0) {
		ofstream my_file("Data_cpp/Results.txt", ios::out | ios::app);
		if (my_file.is_open()){
			my_file << comm_sz << " " << K << " " << N << " " << M << " " << R << " " << t_end_max << " " << ((float)total_t)/CLOCKS_PER_SEC << " " << AO_iter << " " << f_value << " " << comm_sz << endl;
			my_file << endl;
			my_file.close();
		}
		else
			cout << "Unable to open file";
	}

	
	//  <-----------------   All Processor write their B part  --------------------------------------->  //
/*
	stringstream ss;
    ss << my_rank;
    string str = ss.str();
	str = "Data_cpp/B_final_" + str + ".bin";
	MatrixXd B_T(R,n_M[row_rank]);
	B_T.noalias() = B.transpose();
	Write_to_File(R, n_M[row_rank], B_T, str.c_str());

	// <----------------   Processor 0 writes produced factors in Data_cpp folder -------------------->  //
	if (my_rank == 0) {

		//Write_to_File(N, R, A, "Data_cpp/A_final.bin");

		Write_to_File(N, R, Z, "Data_cpp/Z_final.bin");

		/*MatrixXd B_all_T(R,M);

		int i,j;
		j = 0;

		for (i=0; i < comm_sz; i++){
			ss.str("");
			ss << i;
			string str = ss.str();
	        str = "Data_cpp/B_final_" + str + ".bin";
	       	Read_From_File(R, n_M[i],B_all_T.block(0,j,R,n_M[i]) ,str.c_str(), 0);
			j = j + n_M[i];
		}
		MatrixXd B_all(M,R);
		B_all = B_all_T.transpose();
    	Write_to_File(M, R, B_all, "Data_cpp/B_final.bin");
    }

	if (my_rank == 0)
		cout << "Machine Epsilon is: " << numeric_limits<double>::epsilon() << endl;
*/

	MPI_Comm_free( &row_comm );							// |
	MPI_Comm_free( &column_comm );							// | Free the new communicators
	MPI_Comm_free( &grid_comm );						// |
	MPI_Finalize();													// | Shut down the MPI execution environment

	return 0;
}														// End of main
