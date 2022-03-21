/*--------------------------------------------------------------------------------------------------*/
/* 			Function for the computation of the weight, lambda, of the proximal term				*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/ 
/* Georgios Kostoulas																				*/
/* 6/4/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

void G_Lambda(double *lambda, double *q, double *L, double mu){
	*q = mu/(*L);
	if (1/(*q)>1e6)
		*lambda = 10 * mu;
	else if (1/(*q)>1e3)
		*lambda = mu;
	else
		*lambda = mu/10;
	
	*L += (*lambda);
	mu += (*lambda);
	*q = mu/(*L);
}
