/*--------------------------------------------------------------------------------------------------*/
/* 											Function for the computation of the Objective Value		 											*/
/*    																of the semi ONMF problem 																			*/
/*                                                                           												*/
/* A. P. Liavas																																											*/
/* P. Karakasis																																											*/
/* 23/3/2020                                              																					*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

using namespace Eigen;

double Get_Objective_Value( const Ref<const MatrixXd> S, const Ref<const MatrixXd> X_A, const Ref<const MatrixXd> A, double c_k, MPI_Comm grid_comm){

	double local_sum, global_sum, local_sum_2;
	int R = A.cols();
	MatrixXd D(R,R);
	MatrixXd Q(R,R);

	Q = MatrixXd::Ones(R, R) - MatrixXd::Identity(R, R);
	D = Q * (A.transpose() * A);

	local_sum = (c_k / 2) * (D.trace());

	MPI_Allreduce(&local_sum, &local_sum_2, 1, MPI_DOUBLE, MPI_SUM, grid_comm);		// Communication through all processes

	local_sum = 0.5 * (X_A -  A * S.transpose()).squaredNorm();

	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, grid_comm);		// Communication through all processes

	return sqrt(global_sum) + local_sum_2;
}
