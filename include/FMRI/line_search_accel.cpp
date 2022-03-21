/*--------------------------------------------------------------------------------------------------*/
/* 						Function that implements the acceleration step								*/
/*    			(calls Khatri_Rao_Product and Get_Objective_Value_Accel functions)				   	*/
/*    																							   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 6/4/2017                   				                           								*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"
#include "Eigen/Dense"

using namespace Eigen;

void Line_Search_Accel(const Ref<const MatrixXd> A_old_N, const Ref<const MatrixXd> B_old_N, const Ref<const MatrixXd> C_old_N, Ref<MatrixXd> A,
						Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> A_T_A, Ref<MatrixXd> B_T_B, Ref<MatrixXd> C_T_C, Ref<MatrixXd> KhatriRao_BA,
						const Ref<const MatrixXd> X_C, int *acc_fail, int *acc_coeff, int iter, double f_value, double frob_X,
						MPI_Comm column_comm, MPI_Comm row_comm, MPI_Comm tube_comm, MPI_Comm grid_comm){

	int R = A.cols();
	MatrixXd A_accel, B_accel, C_accel;												// Factors A_accel, B_accel, C_accel
	MatrixXd A_T_A_accel(R, R), B_T_B_accel(R, R), C_T_C_accel(R, R);
	double f_accel;																	// Objective Value after the acceleration step
	double acc_step;

	acc_step = pow(iter+1,(1.0/(*acc_coeff)));
	A_accel.noalias() = A_old_N + acc_step * (A - A_old_N);
	B_accel.noalias() = B_old_N + acc_step * (B - B_old_N);
	C_accel.noalias() = C_old_N + acc_step * (C - C_old_N);

	A_T_A_accel.noalias() = A_accel.transpose() * A_accel;											// Communicate A^T*A
	//MPI_Allreduce(MPI_IN_PLACE, A_T_A_accel.data(), R*R, MPI_DOUBLE, MPI_SUM, column_comm);			// Communication through columns

	B_T_B_accel.noalias() = B_accel.transpose() * B_accel;											// Communicate B^T*B
	MPI_Allreduce(MPI_IN_PLACE, B_T_B_accel.data(), R*R, MPI_DOUBLE, MPI_SUM, row_comm);			// Communication through rows

	C_T_C_accel.noalias() = C_accel.transpose() * C_accel;											// Communicate C^T*C
	//MPI_Allreduce(MPI_IN_PLACE, C_T_C_accel.data(), R*R, MPI_DOUBLE, MPI_SUM, tube_comm);			// Communication through tubes

	//Khatri_Rao_Product(B_accel, A_accel, KhatriRao_BA);
	f_accel = Get_Objective_Value_Accel(C_accel, X_C, KhatriRao_BA, A_T_A_accel, B_T_B_accel, C_T_C_accel, frob_X, grid_comm);

	if (f_value>f_accel){
		A = A_accel;
		B = B_accel;
		C = C_accel;
		A_T_A = A_T_A_accel;
		B_T_B = B_T_B_accel;
		C_T_C = C_T_C_accel;
	}
	else
		(*acc_fail)++;

	if (*acc_fail==5){
		*acc_fail=0;
		(*acc_coeff)++;
	}

}
