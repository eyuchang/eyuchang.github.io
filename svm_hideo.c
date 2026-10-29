/***********************************************************************/
/*                                                                     */
/*   svm_hideo.c                                                       */
/*                                                                     */
/*   The Hildreth and D'Espo solver specialized for SVMs.              */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 19.07.99                                                    */
/*                                                                     */
/*   Copyright (c) 1999  Universitaet Dortmund - All rights reserved   */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

# include <math.h>
# include "svm_common.h"

/* Common Block Declarations */

long verbosity;

# define PROGRESS                   1
# define NO_PROGRESS_MAXITER        2
# define NO_PROGRESS_PRIMAL_OPTIMAL 3
# define NO_PROGRESS_DUAL_OPTIMAL   4
# define NAN_SOLUTION               5

# define PRIMAL_OPTIMAL      1
# define DUAL_OPTIMAL        2
# define MAXITER_EXCEEDED    3

/* /////////////////////////////////////////////////////////////// */

# define DEF_PRECISION          1E-5
# define DEF_MAX_ITERATIONS     200
# define DEF_LINDEP_SENSITIVITY 1E-8

double *optimize_qp(QP *, double *, long, double *, LEARN_PARM *);
double *primal=0,*dual=0;
long   precision_violations=0;
double opt_precision=DEF_PRECISION;
long   maxiter=DEF_MAX_ITERATIONS;
double lindep_sensitivity=DEF_LINDEP_SENSITIVITY;
double *buffer;
long   *nonoptimal;

/* /////////////////////////////////////////////////////////////// */

void *my_malloc();

int optimize_hildreth_despo(long,long,double,double,long,double,double *,
			    double *,double *,double *,double *,double *,
			    double *,double *,double *,long *,double *);
int solve_dual(long,long,double,double,long,double *,double *,double *,
	       double *,double *,double *,double *,double *,double *,
	       double *,double *,double *,double *);

void linvert_matrix(double *, long, double *, double, long *);
void lprint_matrix(double *, long);
void ladd_matrix(double *, long, double);
void lcopy_matrix(double *, long, double *);
void lswitch_rows_matrix(double *, long, long, long);
void lswitchrk_matrix(double *, long, long, long);

double calculate_qp_objective(long, double *, double *, double *);

double *optimize_qp(qp,epsilon_crit,nx,threshold,learn_parm)
QP *qp;
double *epsilon_crit;
long nx; /* Maximum number of variables in QP */
double *threshold; 
LEARN_PARM *learn_parm;
/* start the optimizer and return the optimal values */
/* The HIDEO optimizer does not necessarily fully solve the problem. */
/* Since it requires a strictly positive definite hessian, the solution */
/* is restricted to a linear independent subset in case the matrix is */
/* only semi-definite. */
{
  long i,j;
  int result;

  if(!primal) { /* allocate memory at first call */
    primal=(double *)my_malloc(sizeof(double)*nx);
    dual=(double *)my_malloc(sizeof(double)*((nx+1)*2));
    nonoptimal=(long *)my_malloc(sizeof(long)*(nx));
    buffer=(double *)my_malloc(sizeof(double)*((nx+1)*2*(nx+1)*2+
					       nx*nx+2*(nx+1)*2+2*nx+1+2*nx+
					       nx));
  }

  if(verbosity>=4) { /* really verbose */
    printf("\n\n");
    for(i=0;i<qp->opt_n;i++) {
      printf("%f: ",qp->opt_g0[i]);
      for(j=0;j<qp->opt_n;j++) {
	printf("%f ",qp->opt_g[i*qp->opt_n+j]);
      }
      printf(": a=%.30f",qp->opt_xinit[i]);
      printf(": y=%f\n",qp->opt_ce[i]);
    }
    printf("\n");
  }


  result=optimize_hildreth_despo(qp->opt_n,qp->opt_m,
				 opt_precision,(*epsilon_crit),maxiter,
				 lindep_sensitivity,
				 qp->opt_g,qp->opt_g0,qp->opt_ce,qp->opt_ce0,
				 qp->opt_low,qp->opt_up,primal,qp->opt_xinit,
				 dual,nonoptimal,buffer);
  /* printf("result=%d\n",result); */

  if(result == NAN_SOLUTION) {
    lindep_sensitivity*=2;  /* throw out linear dependent examples more */
                            /* generously */
    if(learn_parm->svm_maxqpsize>2) {
      learn_parm->svm_maxqpsize--;  /* decrease size of qp-subproblems */
    }
    while(result == NAN_SOLUTION) {
      /* Shaking things up should help in this unlikely case. */
      for(i=0;i<qp->opt_n;i++) {
	qp->opt_g[i*qp->opt_n+i]+=1;
      }
      result=optimize_hildreth_despo(qp->opt_n,qp->opt_m,opt_precision,
				     (*epsilon_crit),maxiter,
				     lindep_sensitivity,
				     qp->opt_g,qp->opt_g0,qp->opt_ce,
				     qp->opt_ce0,qp->opt_low,qp->opt_up,
				     primal,qp->opt_xinit,dual,
				     nonoptimal,buffer);
    }
    result=NAN_SOLUTION;
    precision_violations++;
  }

  if((result == NO_PROGRESS_MAXITER) || 
     (result == NO_PROGRESS_PRIMAL_OPTIMAL)) {
      /* the problem could be, that all examples in the working set */
      /* are linear dependent. I have never seen this happen on my data. */
      /* But shaking things up should help in this unlikely case. */
      for(i=0;i<qp->opt_n;i++) {
	for(j=0;j<qp->opt_n;j++) {
	  qp->opt_g[i*qp->opt_n+j]=0;
	}
	qp->opt_g[i*qp->opt_n+i]=0.5/qp->opt_up[i];
      }
      optimize_hildreth_despo(qp->opt_n,qp->opt_m,opt_precision,
			      (*epsilon_crit),maxiter,lindep_sensitivity,
			      qp->opt_g,qp->opt_g0,qp->opt_ce,qp->opt_ce0,
			      qp->opt_low,qp->opt_up,primal,qp->opt_xinit,dual,
			      nonoptimal,buffer);
  }

  if(result == NO_PROGRESS_DUAL_OPTIMAL) {
    if(learn_parm->svm_maxqpsize>2) {
      learn_parm->svm_maxqpsize--;  /* decrease size of qp-subproblems */
    }
  }

  if(result == NO_PROGRESS_MAXITER) {
    /* in any case, increase the precision */
    maxiter+=50;
    precision_violations++;
    if(verbosity>=2) {
      printf("NOTICE: Increasing Precision of HIDEO.\n");
    }
  }

  if(precision_violations > 50) {
    precision_violations=0;
    (*epsilon_crit)*=10.0; 
    if(verbosity>=1) {
      printf("\nWARNING: Relaxing epsilon on KT-Conditions (%f).\n",
	     (*epsilon_crit));
    }
  }	  

  if(qp->opt_m>0) 
    (*threshold)=dual[1]-dual[0];
  else
    (*threshold)=0;
  if(isnan(*threshold))
    (*threshold)=0;
  return(primal);
}


int optimize_hildreth_despo(n,m,precision,epsilon_crit,maxiter,
			    lindep_sensitivity,g,g0,ce,ce0,low,up,
			    primal,init,dual,lin_dependent,buffer)
     long   n;            /* number of variables */
     long   m;            /* number of linear equality constraints [0,1] */
     double precision;    /* solve at least to this dual precision */
     double epsilon_crit; /* stop, if KT-Conditions approx fulfilled */
     long   maxiter;      /* stop after this many iterations */
     double lindep_sensitivity; /* epsilon for detecting linear dependent ex */
     double *g;           /* hessian of objective */
     double *g0;          /* linear part of objective */
     double *ce,*ce0;     /* linear equality constraints */
     double *low,*up;     /* box constraints */
     double *primal;      /* primal variables */
     double *init;        /* initial values of primal */
     double *dual;        /* dual variables */
     long   *lin_dependent;
     double *buffer;
{
  long i,j,k,from,to,n_indep;
  double sum,bmin,bmax,w,maxviol,viol,epsilon_hideo,dist,model_b;
  double *d,*d0,*ig,*dual_old,*temp;       
  double *g0_new,*ce_new,*ce0_new,*low_new,*up_new;
  double add;
  int result;
  double obj_before,obj_after; 
  long b1,b2;

  g0_new=&(buffer[0]);    /* claim regions of buffer */
  d=&(buffer[n]);
  d0=&(buffer[n+(n+m)*2*(n+m)*2]);
  ce_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2]);
  ce0_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n]);
  ig=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m]);
  dual_old=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n]);
  low_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2]);
  up_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2+n]);
  temp=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2+n+n]);

  b1=-1;
  b2=-1;
  for(i=0;i<n;i++) {   /* get variables with steepest descent and */
    sum=g0[i];         /* opposite sign */
    for(j=0;j<n;j++) 
      sum+=init[j]*g[i*n+j];
    sum=sum-ce[i];
    if((sum<bmin) || (b1==-1)) {
      bmin=sum;
      b1=i;
    }
    if((sum>bmax) || (b2==-1)) {
      bmax=sum;
      b2=i;
    }
  }

  lcopy_matrix(g,n,d);
  add=0;
  if(m==1) {
    /* if we have a biased hyperplane, then adding a constant to the */
    /* hessian does not change the solution. So that is done for examples */
    /* with zero diagonal entry, since HIDEO cannot handle them. */
    for(i=0;i<n;i++) {
      if(d[i*n+i]==0) {
	for(j=0;j<n;j++) {
	  for(k=0;k<n;k++) {
	    d[j*n+k]+=1.0*ce[j]*ce[k];
	  }
	}
	add+=1.0;
	i=0;
      }
    }
  }

  lswitchrk_matrix(d,n,b1,b2); /* switch, so that variables are better mixed */
  linvert_matrix(d,n,ig,lindep_sensitivity,lin_dependent);
  lswitchrk_matrix(ig,n,b1,b2);
  i=lin_dependent[b1];  /* now switch back */
  lin_dependent[b1]=lin_dependent[b2];
  lin_dependent[b2]=i;
  /* lprint_matrix(d,n);   */
  /* lprint_matrix(ig,n);  */
  lcopy_matrix(g,n,d);   /* restore d matrix */
  if(add>0)
    for(j=0;j<n;j++) {
      for(k=0;k<n;k++) {
	d[j*n+k]+=add*ce[j]*ce[k];
      }
    }

  for(i=0;i<n;i++) {  /* fix linear dependent vectors */
    g0_new[i]=g0[i]+add*ce0[0]*ce[i];
  }
  if(m>0) ce0_new[0]=-ce0[0];
  for(i=0;i<n;i++) {  /* fix linear dependent vectors */
    if(lin_dependent[i]) {
      for(j=0;j<n;j++) {
	if(!lin_dependent[j]) {
	  g0_new[j]+=init[i]*d[i*n+j];
	}
      }
      if(m>0) ce0_new[0]-=(init[i]*ce[i]);
    }
  }
  from=0;   /* remove linear dependent vectors */
  to=0;
  n_indep=0;
  for(i=0;i<n;i++) {
    if(!lin_dependent[i]) {
      g0_new[n_indep]=g0_new[i];
      ce_new[n_indep]=ce[i]; 
      low_new[n_indep]=low[i];
      up_new[n_indep]=up[i];
      primal[n_indep]=init[i];
      n_indep++;
    }
    for(j=0;j<n;j++) {
      if((!lin_dependent[i]) && (!lin_dependent[j])) {
        ig[to]=ig[from];
	to++;
      }
      from++;
    }
  }

  if(n_indep<=1) /* cannot optimize with only one variable */
    return((int)NO_PROGRESS_PRIMAL_OPTIMAL);

  obj_before=calculate_qp_objective(n,g,g0,init);
  
  result=solve_dual(n_indep,m,precision,epsilon_crit,maxiter,g,g0_new,ce_new,
		    ce0_new,low_new,up_new,primal,d,d0,ig,dual,dual_old,temp);
  
  if(m>0) 
    model_b=dual[1]-dual[0];
  else
    model_b=0;
  
  j=n_indep;
  for(i=n-1;i>=0;i--) {
    if(!lin_dependent[i]) {
      j--;
      primal[i]=primal[j];
    }
    else if((m==0) && (g[i*n+i]==0)) {
      /* if we use a biased hyperplane, each example with a zero diagonal */
      /* entry must have an alpha at the upper bound. Doing this */
      /* is essential for the HIDEO optimizer, since it cannot handle zero */
      /* diagonal entries in the hessian for the unbiased hyperplane case. */
      primal[i]=up[i];  
    }
    else {
      primal[i]=init[i];  /* leave as is */
    }
    temp[i]=primal[i];
  }

  obj_after=calculate_qp_objective(n,g,g0,primal);
  /* printf("before=%f after=%f result=%d\n",obj_before,obj_after,result); */

  if(isnan(obj_after))
      return((int)NAN_SOLUTION);
  if(obj_after >= obj_before) 
    if(result == MAXITER_EXCEEDED) 
      return((int)NO_PROGRESS_MAXITER);
    else if(result == DUAL_OPTIMAL) 
      return((int)NO_PROGRESS_DUAL_OPTIMAL);
    else
      return((int)NO_PROGRESS_PRIMAL_OPTIMAL);

  /* Check the precision of the alphas. If results of current optimization */
  /* violate KT-Conditions, relax the epsilon on the bounds on alphas. */
  epsilon_hideo=1E-20;
  for(i=0;i<n;i++) {
    if(!lin_dependent[i]) {
      dist=-model_b*ce[i]; 
      dist+=(g0[i]+1.0);
      for(j=0;j<i;j++) {
	dist+=(primal[j]*g[j*n+i]);
      }
      for(j=i;j<n;j++) {
	dist+=(primal[j]*g[i*n+j]);
      }
      if((primal[i]<(up[i]-epsilon_hideo)) && (dist < (1.0-epsilon_crit))) {
	epsilon_hideo=(up[i]-primal[i])*2.0;
      }
      else if((primal[i]>(low[i]+epsilon_hideo)) && (dist > (1.0+epsilon_crit))) {
	epsilon_hideo=(primal[i]-low[i])*2.0;
      }
      /* printf("HIDEO: a[%d]=%f, dist=%f, b=%f, epsilon_hideo=%f\n",i,
	 primal[i],dist,model_b,epsilon_hideo);  */
    }
  }
  for(i=0;i<n;i++) {  /* clip alphas to bounds */
    if(primal[i]<=(low[i]+epsilon_hideo)) {
      primal[i]=low[i];
    }
    else if(primal[i]>=(up[i]-epsilon_hideo)) {
      primal[i]=up[i];
    }
  }

  obj_after=calculate_qp_objective(n,g,g0,primal);

  if(obj_after >= obj_before) {
    for(i=0;i<n;i++) {  /* use unclipped values */
      primal[i]=temp[i];
    }
  }
  return((int)PROGRESS);
}


solve_dual(n,m,precision,epsilon_crit,maxiter,g,g0,ce,ce0,low,up,primal,
	   d,d0,ig,dual,dual_old,temp)
     /* Solves the dual using the method of Hildreth and D'Espo. */
     /* Can only handle problems with zero or exactly one */
     /* equality constraints. */

     long   n;            /* number of variables */
     long   m;            /* number of linear equality constraints */
     double precision;    /* solve at least to this dual precision */
     double epsilon_crit; /* stop, if KT-Conditions approx fulfilled */
     long   maxiter;      /* stop after that many iterations */
     double *g;
     double *g0;          /* linear part of objective */
     double *ce,*ce0;     /* linear equality constraints */
     double *low,*up;     /* box constraints */
     double *primal;      /* variables (with initial values) */
     double *d,*d0,*ig,*dual,*dual_old,*temp;       /* buffer  */
{
  long i,j,k,iter;
  double sum,w,maxviol,viol,temp1,temp2;
  double model_b,dist;
  long retrain,maxfaktor,primal_optimal;
  double epsilon_a=1E-15;
  double obj_before,obj_after; 

  if((m<0) || (m>1)) 
    perror("SOLVE DUAL: inappropriate number of eq-constrains!");

  obj_before=calculate_qp_objective(n,g,g0,primal);

  for(i=0;i<2*(n+m);i++) {
    dual[i]=0;
    dual_old[i]=0;
  }
  for(i=0;i<n;i++) {   
    for(j=0;j<n;j++) {   /* dual hessian for box constraints */
      d[i*2*(n+m)+j]=ig[i*n+j];
      d[(i+n)*2*(n+m)+j]=-ig[i*n+j];
      d[i*2*(n+m)+j+n]=-ig[i*n+j];
      d[(i+n)*2*(n+m)+j+n]=ig[i*n+j];
    }
    if(m>0) {
      sum=0;              /* dual hessian for eq constraints */
      for(j=0;j<n;j++) {
	sum+=(ce[j]*ig[i*n+j]);
      }
      d[i*2*(n+m)+2*n]=sum;
      d[i*2*(n+m)+2*n+1]=-sum;
      d[(n+i)*2*(n+m)+2*n]=-sum;
      d[(n+i)*2*(n+m)+2*n+1]=sum;
      d[(n+n)*2*(n+m)+i]=sum;
      d[(n+n+1)*2*(n+m)+i]=-sum;
      d[(n+n)*2*(n+m)+(n+i)]=-sum;
      d[(n+n+1)*2*(n+m)+(n+i)]=sum;
      
      sum=0;
      for(j=0;j<n;j++) {
	for(k=0;k<n;k++) {
	  sum+=(ce[k]*ce[j]*ig[j*n+k]);
	}
      }
      d[(n+n)*2*(n+m)+2*n]=sum;
      d[(n+n)*2*(n+m)+2*n+1]=-sum;
      d[(n+n+1)*2*(n+m)+2*n]=-sum;
      d[(n+n+1)*2*(n+m)+2*n+1]=sum;
    } 
  }

  for(i=0;i<n;i++) {   /* dual linear component for the box constraints */
    w=0;
    for(j=0;j<n;j++) {
      w+=(ig[i*n+j]*g0[j]); 
    }
    d0[i]=up[i]+w;
    d0[i+n]=-low[i]-w;
  }

  if(m>0) {  
    sum=0;             /* dual linear component for eq constraints */
    for(j=0;j<n;j++) {
      for(k=0;k<n;k++) {
	sum+=(ce[k]*ig[k*n+j]*g0[j]); 
      }
    }
    d0[2*n]=ce0[0]+sum;
    d0[2*n+1]=-ce0[0]-sum;
  }

  maxviol=999999;
  iter=0;
  retrain=1;
  maxfaktor=1;
  while((retrain) && (maxviol > 0) && (iter < (maxiter*maxfaktor))) {
    iter++;
    
    while((maxviol > precision) && (iter < (maxiter*maxfaktor))) {
      iter++;
      maxviol=0;
      for(i=0;i<2*(n+m);i++) {
	sum=d0[i];
	for(j=0;j<2*(n+m);j++) {
	  sum+=d[i*2*(n+m)+j]*dual_old[j];
	}
	sum-=d[i*2*(n+m)+i]*dual_old[i];
	dual[i]=-sum/d[i*2*(n+m)+i];
	if(dual[i]<0) dual[i]=0;
	
	viol=fabs(dual[i]-dual_old[i]);
	if(viol>maxviol) 
	  maxviol=viol;
	dual_old[i]=dual[i];
      }
      /*
      printf("%d) maxviol=%20f precision=%f\n",iter,maxviol,precision); 
      */
    }
  
    if(m>0) {
      for(i=0;i<n;i++) {
	temp[i]=dual[i]-dual[i+n]+ce[i]*(dual[n+n]-dual[n+n+1])+g0[i];
      }
    } 
    else {
      for(i=0;i<n;i++) {
	temp[i]=dual[i]-dual[i+n]+g0[i];
      }
    }
    for(i=0;i<n;i++) {
      primal[i]=0;             /* calc value of primal variables */
      for(j=0;j<n;j++) {
	primal[i]+=ig[i*n+j]*temp[j];
      }
      primal[i]*=-1.0;
      if(primal[i]<=(low[i]+epsilon_a)) {  /* clip to feasible region */
	primal[i]=low[i];
      }
      else if(primal[i]>=(up[i]-epsilon_a)) {
	primal[i]=up[i];
      }
    }

    if(m>0) 
      model_b=dual[n+n+1]-dual[n+n];
    else
      model_b=0;
    retrain=0;
    primal_optimal=1;
    for(i=0;i<n;i++) {     /* check primal KT-Conditions */
      dist=-model_b*ce[i]; 
      dist+=(g0[i]+1.0);
      for(j=0;j<i;j++) {
	dist+=(primal[j]*g[j*n+i]);
      }
      for(j=i;j<n;j++) {
	dist+=(primal[j]*g[i*n+j]);
      }
      /* printf("HIDEOtemp: a[%d]=%f, dist=%f, b=%f\n",i,primal[i],dist,model_b); */
      if((primal[i]<(up[i]-epsilon_a)) && (dist < (1.0-epsilon_crit))) {
	retrain=1;
	primal_optimal=0;
      }
      else if((primal[i]>(low[i]+epsilon_a)) && (dist > (1.0+epsilon_crit))) {
	retrain=1;
	primal_optimal=0;
      }
    }
    if(retrain) 
      precision/=10;
    if(obj_before <= calculate_qp_objective(n,g,g0,primal)) { 
      maxfaktor=10;
      if(!retrain) {
	retrain=1;
	precision/=10;
      }
    }
    else {
      maxfaktor=1;
    }
  }

  /* printf("iter=%d\n",iter); */

  if(m>0) {
    temp1=dual[n+n+1];   /* copy the dual variables for the eq */
    temp2=dual[n+n];     /* constraints to a handier location */
    for(i=n+n+1;i>=2;i--) {
      dual[i]=dual[i-2];
    }
    dual[0]=temp2;
    dual[1]=temp1;
  }

  if(primal_optimal) {
    return((int)PRIMAL_OPTIMAL);
  }
  else if(maxviol == 0.0) {
    return((int)DUAL_OPTIMAL);
  }
  else {
    return((int)MAXITER_EXCEEDED);
  }
}


void linvert_matrix(matrix,depth,inverse,lindep_sensitivity,lin_dependent)
double *matrix;
long depth;
double *inverse,lindep_sensitivity;
long *lin_dependent;
{
  long i,j,k,bestrow;
  double factor,maxdiv;

  for(i=0;i<depth;i++) {
    lin_dependent[i]=0;
    for(j=0;j<depth;j++) {
      if(i==j) inverse[i*depth+j]=1.0;
      else inverse[i*depth+j]=0.0;
    }
  }
  for(i=0;i<depth;i++) {
    if(fabs(matrix[i*depth+i])<lindep_sensitivity) {
      lin_dependent[i]=1;
    }
    else {
      for(j=i+1;j<depth;j++) {
	factor=matrix[j*depth+i]/matrix[i*depth+i];
	for(k=i;k<depth;k++) {
	  matrix[j*depth+k]-=(factor*matrix[i*depth+k]);
	}
	for(k=0;k<depth;k++) {
	  inverse[j*depth+k]-=(factor*inverse[i*depth+k]);
	}
      }
    }
  }
  for(i=depth-1;i>=0;i--) {
    if(!lin_dependent[i]) {
      factor=1/matrix[i*depth+i];
      for(k=0;k<depth;k++) {
	inverse[i*depth+k]*=factor;
      }
      matrix[i*depth+i]=1;
      for(j=i-1;j>=0;j--) {
	factor=matrix[j*depth+i];
	matrix[j*depth+i]=0;
	for(k=0;k<depth;k++) {
	  inverse[j*depth+k]-=(factor*inverse[i*depth+k]);
	}
      }
    }
  }
}

void lprint_matrix(matrix,depth)
double *matrix;
long depth;
{
  long i,j;
  for(i=0;i<depth;i++) {
    for(j=0;j<depth;j++) {
      printf("%5.2f ",(double)(matrix[i*depth+j]));
    }
    printf("\n");
  }
  printf("\n");
}

void ladd_matrix(matrix,depth,scalar)
double *matrix;
long depth;
double scalar;
{
  long i,j;
  for(i=0;i<depth;i++) {
    for(j=0;j<depth;j++) {
      matrix[i*depth+j]+=scalar;
    }
  }
}

void lcopy_matrix(matrix,depth,matrix2) 
double *matrix;
long depth;
double *matrix2;
{
  long i;
  
  for(i=0;i<(depth)*(depth);i++) {
    matrix2[i]=matrix[i];
  }
}

void lswitch_rows_matrix(matrix,depth,r1,r2) 
double *matrix;
long depth,r1,r2;
{
  long i,j,k;
  double temp;

  for(i=0;i<depth;i++) {
    temp=matrix[r1*depth+i];
    matrix[r1*depth+i]=matrix[r2*depth+i];
    matrix[r2*depth+i]=temp;
  }
}

void lswitchrk_matrix(matrix,depth,rk1,rk2) 
double *matrix;
long depth,rk1,rk2;
{
  long i,j,k;
  double temp;

  for(i=0;i<depth;i++) {
    temp=matrix[rk1*depth+i];
    matrix[rk1*depth+i]=matrix[rk2*depth+i];
    matrix[rk2*depth+i]=temp;
  }
  for(i=0;i<depth;i++) {
    temp=matrix[i*depth+rk1];
    matrix[i*depth+rk1]=matrix[i*depth+rk2];
    matrix[i*depth+rk2]=temp;
  }
}

double calculate_qp_objective(opt_n,opt_g,opt_g0,alpha)
long opt_n;
double *opt_g,*opt_g0,*alpha;
{
  double obj;
  long i,j;
  obj=0;  /* calculate objective  */
  for(i=0;i<opt_n;i++) {
    obj+=(opt_g0[i]*alpha[i]);
    obj+=(0.5*alpha[i]*alpha[i]*opt_g[i*opt_n+i]);
    for(j=0;j<i;j++) {
      obj+=(alpha[j]*alpha[i]*opt_g[j*opt_n+i]);
    }
  }
  return(obj);
}
