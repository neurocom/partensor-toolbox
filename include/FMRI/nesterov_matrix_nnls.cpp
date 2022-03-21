/*--------------------------------------------------------------------------------------------------*/
/* 					Function for the optimal Nesterov algorithm for the 							*/
/*    						Non-Negative Least Squares problem									   	*/
/*                (calls Compute_SVD, G_Lambda and update_alpha functions)      					*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 6/4/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

using namespace Eigen;

void Nesterov_Matrix_Nnls(Ref<MatrixXd> Z, Ref<MatrixXd> W, Ref<MatrixXd> A_init, double delta_1, double delta_2){
	int m = A_init.rows();
	int r = A_init.cols();
	double L, mu, lambda, q, alpha, new_alpha, beta;

	MatrixXd grad_Y(m, r);
	MatrixXd Y(m, r);
	MatrixXd new_A(m, r);
	MatrixXd A(m, r);
	MatrixXd Zero_Matrix = MatrixXd::Zero(m, r);

	Compute_SVD(&L, &mu, Z);

	// G_Lambda(&lambda, &q, &L, mu);
	//
	// Z += lambda * MatrixXd::Identity(r, r);
	// W -= lambda * A_init;

	alpha = 1;

	A = A_init;
	Y = A_init;

	while(1){
		grad_Y = W;										// |
		grad_Y.noalias() += Y * Z;						// | grad_Y = W + Y * Z.transpose();

		if (grad_Y.cwiseProduct(Y).cwiseAbs().maxCoeff() <= delta_1 && grad_Y.minCoeff() >= -delta_2)
			break;

		new_A = (Y - grad_Y/L).cwiseMax(Zero_Matrix);

		update_alpha(alpha, q, &new_alpha);
		beta = alpha * (1 - alpha) / (alpha*alpha + new_alpha);

		Y = (1 + beta) * new_A - beta * A;

		A = new_A;
		alpha = new_alpha;

	}
	A_init = A;

}
