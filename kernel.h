/************************************************************************/
/*                                                                      */
/*   kernel.h                                                           */
/*                                                                      */
/*   User defined kernel function. Feel free to plug in your own.       */
/*                                                                      */
/*   Copyright: Thorsten Joachims                                       */
/*   Date: 16.12.97                                                     */
/*                                                                      */
/************************************************************************/

/* KERNEL_PARM is defined in svm.h The field 'custom' is reserved for */
/* parameters of the user defined kernel. You can also access and use */
/* the parameters of the other kernels. */

double custom_kernel(kernel_parm,a,b) /* plug in you favorite kernel */
KERNEL_PARM *kernel_parm;      
DOC *a,*b;
{
  return((double)(1.0));
}
