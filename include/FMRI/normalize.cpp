/*--------------------------------------------------------------------------------------------------*/
/* 				Function for the Normalization of the columns of the Factors A and Z											*/
/*    					All the weights of factor Z go to the columns of Factor S		   											*/
/*       			 (Used throughout the Algorithm, Doesn't require communication)												*/
/*   																																																*/
/*                                                                           												*/
/* A. P. Liavas																																											*/
/* P. Karakasis																																											*/
/* G. Lourakis																																											*/
/* 22/3/2020 											                                             											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

using namespace Eigen;

void Normalize(Ref<MatrixXd> A, Ref<MatrixXd> S, MPI_Comm comm){
	int R = A.cols();
	double a;
	VectorXd lambda_A(R);

  lambda_A = A.colwise().squaredNorm();

	MPI_Allreduce(MPI_IN_PLACE, lambda_A.data(), R, MPI_DOUBLE, MPI_SUM, comm);	// | Communication through all processes

	for (int i=0; i<R; i++){

		a = sqrt( lambda_A(i) );

		if (a>0) {
			A.col(i) *= 1/a;
			S.col(i) *= a;
		}

	}

}
