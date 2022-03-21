/*--------------------------------------------------------------------------------------------------*/
/* 				Function for the computation of the Objective Value	of the NTF problem	 			*/
/*    							(used in the acceleration step)									   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/ 
/* Georgios Kostoulas																				*/
/* 6/4/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

using namespace Eigen;

double Get_Objective_Value_Accel(const Ref<const MatrixXd> C, const Ref<const MatrixXd> X_C, const Ref<const MatrixXd> Kr, const Ref<const MatrixXd> A_T_A, 
								const Ref<const MatrixXd> B_T_B, const Ref<const MatrixXd> C_T_C, double frob_X, MPI_Comm grid_comm){
	double local_sum, global_sum;
	MatrixXd temp_X1;	
	MatrixXd temp_X2;
	
	temp_X1.noalias() = (Kr.transpose() * X_C.transpose()) * C;
	temp_X2.noalias() = A_T_A.cwiseProduct(B_T_B) * C_T_C;
	local_sum = temp_X1.trace();
	
	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, grid_comm);		// Communication through all processes
	
	return sqrt(frob_X - 2 * global_sum + temp_X2.trace());
}
