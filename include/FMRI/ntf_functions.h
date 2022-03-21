/*--------------------------------------------------------------------------------------------------*/
/* 							HEADER FILE ntf_functions.h			    																								*/
/*    		(Containes declerations of functions used by ntf_eigen_mpi.cpp)														*/
/*                                                                           												*/
/* A. P. Liavas																																											*/
/* Georgios Kostoulas																																								*/
/* Georgios Lourakis																																								*/
/* 6/4/2017                                              																					  */
/*--------------------------------------------------------------------------------------------------*/
#ifndef NTF_FUNCTIONS_H		// if ntf_functions.h hasn't been included yet...
#define NTF_FUNCTIONS_H		// #define this so the compiler knows it has been included

#include "Eigen/Dense"
#include <mpi.h>

using namespace Eigen;

void Compute_Grid_Dimensions(int *p_A, int *p_B, int *comm_sz, int flag);

void Create_Grid(MPI_Comm *grid_comm, MPI_Comm *column_comm, MPI_Comm *row_comm, int *my_rank, int *column_rank, int *row_rank, int *coordinates, int p_A, int p_B);

void Compute_SVD(double *L, double *mu, const Ref<const MatrixXd> Z);

void Dis_Count(int *dis, int *Count, int size, int Dim, int R);

void G_Lambda(double *lambda, double *q, double *L, double mu);

double Get_Objective_Value( const Ref<const MatrixXd> S, const Ref<const MatrixXd> X_A, const Ref<const MatrixXd> A, double c_k, MPI_Comm grid_comm);

double Get_Objective_Value_Accel(const Ref<const MatrixXd> C, const Ref<const MatrixXd> X_C, const Ref<const MatrixXd> Kr, const Ref<const MatrixXd> A_T_A,
								const Ref<const MatrixXd> B_T_B, const Ref<const MatrixXd> C_T_C, double frob_X, MPI_Comm grid_comm);

void Line_Search_Accel(const Ref<const MatrixXd> A_old_N, const Ref<const MatrixXd> B_old_N, const Ref<const MatrixXd> C_old_N, Ref<MatrixXd> A,
						Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> A_T_A, Ref<MatrixXd> B_T_B, Ref<MatrixXd> C_T_C, Ref<MatrixXd> KhatriRao_BA,
						const Ref<const MatrixXd> X_C, int *acc_fail, int *acc_coeff, int iter, double f_value, double frob_X,
						MPI_Comm column_comm, MPI_Comm row_comm, MPI_Comm tube_comm, MPI_Comm grid_comm);

void Normalize(Ref<MatrixXd> A, Ref<MatrixXd> S, MPI_Comm comm);

void Normalize_Init(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, MPI_Comm row_comm, MPI_Comm tube_comm);

void Nesterov_Matrix_Nnls(Ref<MatrixXd> Z, Ref<MatrixXd> W, Ref<MatrixXd> A_init, double delta_1, double delta_2);

void Read_Data(Ref<MatrixXd> A_init, Ref<MatrixXd> S_init, Ref<MatrixXd> X_A, int N, int n_N, int skip_N, int n_M, int skip_M);

void Read_From_File(int nrows, int ncols, Ref<MatrixXd> Mat, const char *file_name, int skip);

void Set_Info(int* R, int* K, int* N, int* M, double* c_k, const char *file_name);

void update_alpha(double alpha, double q, double *new_alpha);

void Update_S_factor(const Ref<const MatrixXd> X_A, const Ref<const MatrixXd> A, Ref<MatrixXd> S, MPI_Comm comm);

void Update_A_n_Z_factors( Ref<MatrixXd> A, const Ref<const	MatrixXd> S, const Ref<const	MatrixXd> X_A,
                         double delta_1, double delta_2, double AO_tol, double c_k, int comm_sz, MPI_Comm comm);


#endif
