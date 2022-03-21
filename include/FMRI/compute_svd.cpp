/*--------------------------------------------------------------------------------------------------*/
/* 					Function for the Computation of maximum and minimum 							*/
/*    							Singular Value of a matrix Z									   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/ 
/* Georgios Kostoulas																				*/
/* 6/4/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

using namespace Eigen;

void Compute_SVD(double *L, double *mu, const Ref<const MatrixXd> Z){
	JacobiSVD<MatrixXd> svd(Z, ComputeThinU | ComputeThinV);
	*L = svd.singularValues().maxCoeff();
	*mu = svd.singularValues().minCoeff();
}
