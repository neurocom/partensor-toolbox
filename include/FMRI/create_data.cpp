/*--------------------------------------------------------------------------------------------------*/
/* 				Write Factors A, S, A_init, S_init, Z_init and matrix X_A	in different files 							*/
/*                       (Saves Data in Data_cpp Folder)       	                   									*/
/*				 	                              																													*/
/*         																																													*/
/* A. P. Liavas																																											*/
/* P. Karakasis																																											*/
/* G. Lourakis																																											*/
/* 22/3/2020                                             																						*/
/*--------------------------------------------------------------------------------------------------*/
#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include <time.h>

using namespace std;
using namespace Eigen;

void Write_to_File(int nrows, int ncols, Ref<MatrixXd> Mat, const char *file_name){
	ofstream my_file(file_name, ios::out | ios::binary | ios::trunc);
	if (my_file.is_open()){
		my_file.write((char *) Mat.data(), nrows*ncols*sizeof(double));
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}

void Write_info_to_File(int* R, int* K, int* N, int* M, double* c_k, const char *file_name){
	ofstream my_file(file_name, ios::out | ios::binary | ios::trunc);

	if (my_file.is_open()){
		my_file.write((char *) R, sizeof(int));
		my_file.write((char *) K, sizeof(int));
		my_file.write((char *) N, sizeof(int));
		my_file.write((char *) M, sizeof(int));
		my_file.write((char *) c_k, sizeof(double));
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}

/* <-----------------------------------------------------------------------------------------------> */
/* < ------------------------------------		MAIN		-----------------------------------------------> */
int main(int argc, char **argv){
	int K = 20;																																		// | Dimensions
	int N = 30000;																																// | and rank
	int M = 100;																																	// | of the
	int R = 10;							             																					// | problem
	double c_k = 10;
	double a;

	MatrixXd A(N, R);
	MatrixXd S(M*K, R);
	MatrixXd A_init(N, R);
	MatrixXd Z_init(N, R);
	MatrixXd S_init(K*M, R);
	MatrixXd A_T(R, N);
	MatrixXd S_T(R, K*M);
	MatrixXd A_init_T(R, N);
	MatrixXd Z_init_T(R, N);
	MatrixXd S_init_T(R, K*M);
	MatrixXd X_A(N, K*M);

	//	<---------------------------		Print Dimensions and Rank		------------------------------------------->	//
	cout << "R=" << R << ", K=" << K << ", N=" << N << ", M=" << M << ", c_k=" << c_k << endl;

	//  <-----------------------------   Create directory Data_cpp/   ------------------------------------------------->    //
  system("mkdir -p Data_cpp/");

	//	<-------------------------		Write Dimensions and Rank in one file each		------------------------------------->	//
	Write_info_to_File(&R, &K, &N, &M, &c_k, "Data_cpp/info.bin");

	//  <----------------------------------   Create true Factors   ----------------------------------------------------->  //
	A = (MatrixXd::Random(N, R) + MatrixXd::Ones(N ,R))/2;			// A~uniform[0,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))

	for (int n=0; n<N; n++){																		// | Meet the orthnormality constraints A'*A = I
			a = A.row(n).maxCoeff();																// |
			for (int r=0; r<R; r++){																// |
					if (A(n,r) < a)																			// |
							A(n,r) = 0;																			// |
			}																												// |
	}																														// |
																															// |
	for (int r=0; r<R; r++){																		// |
			a = A.col(r).norm();																		// |
			A.col(r) *= 1/a;																				// |
	}																														// |

	S = MatrixXd::Random(K*M, R);																// S~uniform[-1,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	X_A = A * S.transpose(); 																		// Create matrix data X_A (concatenation of X_k matrices)

	//  <-------------------------------------------   Create Initial Factors   ----------------------------------------------------->  //

	A_init = (MatrixXd::Random(N, R) + MatrixXd::Ones(N,R))/2;	// A_init~uniform[0,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))
	Z_init = MatrixXd::Zero(N, R);		                        	//
	S_init = MatrixXd::Random(K*M, R);	                        // S_init~uniform[-1,1] (MatrixXd::Random returns uniform random numbers in (-1, 1))

	//	<-------------------	Write A_init, Z_init, S_init, A, S, and X_A in different files		------------------------------------->	//
	A_init_T = A_init.transpose();
	Write_to_File(N, R, A_init_T, "Data_cpp/A_init.bin");

	Z_init_T = Z_init.transpose();
	Write_to_File(N, R, Z_init_T, "Data_cpp/Z_init.bin");

	S_init_T = S_init.transpose();
	Write_to_File(K*M, R, S_init_T, "Data_cpp/S_init.bin");

	A_T = A.transpose();
	Write_to_File(N, R, A_T, "Data_cpp/A.bin");

	S_T = S.transpose();
	Write_to_File(K*M, R, S_T, "Data_cpp/S.bin");

	Write_to_File(N, K*M, X_A, "Data_cpp/X_A.bin");

	return 0;
}
