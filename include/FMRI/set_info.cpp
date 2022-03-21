/*--------------------------------------------------------------------------------------------------*/
/* 								Function for reading the dimensions of   							*/
/*    								the problem from a binary file									*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Kostoulas																				*/
/* Georgios Lourakis																				*/
/* 6/4/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include <iostream>
#include <fstream>
#include "ntf_functions.h"

using namespace std;

void Set_Info(int* R, int* K, int* N, int* M, double* c_k, const char *file_name){
	ifstream my_file(file_name, ios::in | ios::binary);
	if (my_file.is_open()){
		my_file.read((char *) R, sizeof(int));
		my_file.read((char *) K, sizeof(int));
		my_file.read((char *) N, sizeof(int));
		my_file.read((char *) M, sizeof(int));
		my_file.read((char *) c_k, sizeof(double));
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}
