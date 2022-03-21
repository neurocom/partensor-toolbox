/*--------------------------------------------------------------------------------------------------*/
/* 					Function for the update of alpha in the Nesterov algorithm						*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/ 
/* Georgios Kostoulas																				*/
/* 6/4/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

void update_alpha(double alpha, double q, double *new_alpha){
	double a, b, c, D;
	a = 1;
	b = alpha*alpha - q;
	c = -alpha*alpha;
	D = b*b - 4*a*c;
	
	*new_alpha = (-b+sqrt(D))/2;
}
