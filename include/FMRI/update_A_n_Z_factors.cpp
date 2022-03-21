/*--------------------------------------------------------------------------------------------------*/
/* 							                    Function for Updating Factors A and Z        										*/
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

void Update_A_n_Z_factors(Ref<MatrixXd> A, const Ref<const	MatrixXd> S, const Ref<const	MatrixXd> X_A,
                         double delta_1, double delta_2, double AO_tol, double c_k, int comm_sz, MPI_Comm comm){
	int iter;
	int R = A.cols();
  int N = A.rows();

	MatrixXd H(R, R);
	MatrixXd Q(R, R);
  MatrixXd Q_c_k(R, R);
  MatrixXd W(N, R);

  Q = MatrixXd::Ones(R, R) - MatrixXd::Identity(R, R);                          // | Initializations
  Q_c_k = (c_k /(double) comm_sz) * Q;                                          //

  W = - X_A * S;                                                                  //
  H = S.transpose() * S + Q_c_k;                                                // | Update A_k factors
  MPI_Allreduce(MPI_IN_PLACE, W.data(), N*R, MPI_DOUBLE, MPI_SUM, comm);	      // | Communication through
  MPI_Allreduce(MPI_IN_PLACE, H.data(), R*R, MPI_DOUBLE, MPI_SUM, comm);	      // | Communication through

  Nesterov_Matrix_Nnls(H, W, A, delta_1, delta_2);                              // |

}
