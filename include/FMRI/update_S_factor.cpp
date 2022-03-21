/*--------------------------------------------------------------------------------------------------*/
/* 							                    Function for Updating Factor S               										*/
/*    						                  (calls Nesterov_Matrix_Nnls function)	  		                   	*/
/*    																							   	                                            */
/*                                                                           						            */
/* A. P. Liavas																						                                          */
/* P. Karakasis																				                                              */
/* 22/3/2020                                              											                    */
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

using namespace Eigen;

void Update_S_factor(const Ref<const MatrixXd> X_A, const Ref<const MatrixXd> A, Ref<MatrixXd> S, MPI_Comm comm){

  int R = A.cols();
  int n_M = S.rows();

  MatrixXd X_A_T_A(n_M, R);
  MatrixXd A_T_A(R,R);

  X_A_T_A = X_A.transpose() * A;
  MPI_Allreduce(MPI_IN_PLACE, X_A_T_A.data(), n_M*R, MPI_DOUBLE, MPI_SUM, comm);	// | Communication through all processes

  A_T_A = A.transpose() * A;
  MPI_Allreduce(MPI_IN_PLACE, A_T_A.data(), R*R, MPI_DOUBLE, MPI_SUM, comm);	// | Communication through all processes

  S.noalias() = X_A_T_A * A_T_A.inverse();

}
