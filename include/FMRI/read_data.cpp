/*--------------------------------------------------------------------------------------------------*/
/* 					Function for reading initial Factors A, Z, S and concatenated matrices X_k  						*/
/*    																	(calls Read_From_File)  						   											*/
/*    																							   																							*/
/*                                                                           												*/
/* A. P. Liavas																																											*/
/* P. Karakasis																																										  */
/* Georgios Lourakis																																								*/
/* Georgios Kostoulas																																								*/
/* 24/3/2020                                             																						*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"
#include "Eigen/Dense"
#include <fstream>
#include <iostream>
using namespace std;

using namespace Eigen;

void Read_Data(Ref<MatrixXd> A_init, Ref<MatrixXd> S_init, Ref<MatrixXd> X_A, int N, int n_N, int skip_N, int n_M, int skip_M){

	int skip;
	int R = A_init.cols();
	MatrixXd tmp(n_N, 1);
	MatrixXd A_init_T(R, n_N);
	MatrixXd S_init_T(R, n_M);

	//	<---------------------------		Read Initial Factors from file		--------------------------->	//
	skip =  skip_N * R;
	Read_From_File(n_N, R, A_init_T, "Data_cpp/A_init.bin", skip);								// Read initial factor A
	A_init = A_init_T.transpose();

	skip =  skip_M * R;
	Read_From_File(n_M, R, S_init_T, "Data_cpp/S_init.bin", skip);								// Read initial factor S
	S_init = S_init_T.transpose();

	//	<---------------------------		Read Concatenated Matrices from file		--------------------------->	//
	skip =  skip_M * N + skip_N;
	for  (int kk=0; kk<n_M; kk++){
		Read_From_File(n_N, 1, tmp, "Data_cpp/X_A.bin", skip );
		X_A.col(kk) = tmp;
		skip = skip + N;
	}


}
