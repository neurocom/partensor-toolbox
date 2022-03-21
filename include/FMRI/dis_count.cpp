/*--------------------------------------------------------------------------------------------------*/
/* 						Function for the partitioning of Dim rows to size groups															*/
/*                                                                           		     					   		*/
/* A. P. Liavas																																											*/
/* Georgios Lourakis																																								*/
/* Georgios Kostoulas																																								*/
/* 6/4/2017                                             																						*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

void Dis_Count(int *dis, int *Count, int size, int Dim, int R){
	int i, j;
	int x = Dim / size;
	int y = Dim % size;

	for (i=0; i<size; i++){
		Count[i] = i >= y ? x*R : (x+1)*R;
		dis[i] = 0;
		for (j=0; j<i; j++)
			dis[i] = dis[i] + Count[j];
	}
}
