/*--------------------------------------------------------------------------------------------------*/
/* 				Function for the computation of the dimensions of the communication grid			*/
/*       				(if flag==1, we use a cubic partitioning of the grid,						*/
/*                        otherwise, we set grid diensions manually) 								*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 6/4/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

void Compute_Grid_Dimensions(int *p_A, int *p_B, int *comm_sz, int flag){

	//	<----------------------	Cubic Partitioning (Default choice)		------------------->	//
	if(flag == 1)
	{
		if (*comm_sz==1){
			*p_A = 1;
			*p_B = 1;
		}
		else if (*comm_sz==4){
			*p_A = 2;
			*p_B = 2;
		}
		else if (*comm_sz==9){
			*p_A = 3;
			*p_B = 3;
		}
		else if (*comm_sz==16){
			*p_A = 4;
			*p_B = 4;
		}
		else if (*comm_sz==25){
			*p_A = 5;
			*p_B = 5;
		}
		else if (*comm_sz==36){
			*p_A = 6;
			*p_B = 6;
		}
		else if (*comm_sz==42){
			*p_A = 7;
			*p_B = 7;
		}
		else if (*comm_sz==64){
			*p_A = 8;
			*p_B = 8;
		}
		else if (*comm_sz==81){
			*p_A = 9;
			*p_B = 9;
		}
		else if (*comm_sz==100){
			*p_A = 10;
			*p_B = 10;
		}
	}
	else
	{
		if (*comm_sz==1){
			*p_A = 1;
			*p_B = 1;
		}
		else if (*comm_sz==2){
			*p_A = 2;
			*p_B = 1;
		}
		else if (*comm_sz==3){
			*p_A = 3;
			*p_B = 1;
		}
		else if (*comm_sz==4){
			*p_A = 4;
			*p_B = 1;
		}
		else if (*comm_sz==10){
			*p_A = 10;
			*p_B = 1;
		}
		else if (*comm_sz==12){
			*p_A = 12;
			*p_B = 1;
		}
		else if (*comm_sz==18){
			*p_A = 18;
			*p_B = 1;
		}
		else if (*comm_sz==21){
			*p_A = 21;
			*p_B = 1;
		}
		else if (*comm_sz == 25){
			*p_A = 25;
			*p_B = 1;
		}
		else if (*comm_sz == 30){
			*p_A = 30;
			*p_B = 1;
		}
		else if (*comm_sz == 45){
			*p_A = 45;
			*p_B = 1;
		}
		else if (*comm_sz == 60){
			*p_A = 60;
			*p_B = 1;
		}
	}


	/*
	//	<----------------------------	Manual Partitioning		---------------------------->	//
	else{
		if (comm_sz==64){
			*p_A = 1;
			*p_B = 1;
			*p_C = 64;
		}
		else if (comm_sz==9){
			*p_A = 3;
			*p_B = 1;
			*p_C = 3;
		}
		else if (comm_sz==36){
			*p_A = 6;
			*p_B = 1;
			*p_C = 6;
		}
		else if (comm_sz==121){
			*p_A = 11;
			*p_B = 1;
			*p_C = 11;
		}
		else if (comm_sz==225){
			*p_A = 15;
			*p_B = 1;
			*p_C = 15;
		}
		else if (comm_sz==361){
			*p_A = 19;
			*p_B = 1;
			*p_C = 19;
		}
		else if (comm_sz==529){
			*p_A = 23;
			*p_B = 1;
			*p_C = 23;
		}
		else if (comm_sz==512){
			*p_A = 2;
			*p_B = 2;
			*p_C = 128;
		}
		else if (comm_sz==400){
			*p_A = 1;
			*p_B = 1;
			*p_C = 400;
		}
		else if (comm_sz==343){
			*p_A = 1;
			*p_B = 1;
			*p_C = 343;
		}
		else if (comm_sz==344){
			*p_A = 2;
			*p_B = 2;
			*p_C = 86;
		}
		else if (comm_sz==216){
			*p_A = 1;
			*p_B = 1;
			*p_C = 216;
		}
		else if (comm_sz==125){
			*p_A = 1;
			*p_B = 1;
			*p_C = 125;
		}
		else if (comm_sz==124){
			*p_A = 2;
			*p_B = 2;
			*p_C = 31;
		}
		else if (comm_sz==27){
			*p_A = 1;
			*p_B = 1;
			*p_C = 27;
		}
		else if (comm_sz==8){
			*p_A = 1;
			*p_B = 1;
			*p_C = 8;
		}
		else{
			*p_A = 1;
			*p_B = 1;
			*p_C = 1;
		}
	}
	*/
}
