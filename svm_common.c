/************************************************************************/
/*                                                                      */
/*   svm_common.c                                                       */
/*                                                                      */
/*   Definitions and functions used in both svm_learn and svm_classify. */
/*                                                                      */
/*   Author: Thorsten Joachims                                          */
/*   Date: 19.07.99                                                     */
/*                                                                      */
/*   Copyright (c) 1999  Universitaet Dortmund - All rights reserved    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

# include "svm_common.h"
# include "kernel.h"         /* this contains a user supplied kernel */

double classify_example(model,ex,kernel_parm) /* classifies example */
MODEL *model;
DOC *ex;
KERNEL_PARM *kernel_parm;
{
  register long i;
  register double dist;

  dist=0;
  for(i=1;i<model->sv_num;i++) {  
    dist+=kernel(kernel_parm,model->supvec[i],ex)*model->alpha[i];
  }
  return(dist-model->b);
}

CFLOAT kernel(kernel_parm,a,b) /* calculate the kernel function */
KERNEL_PARM *kernel_parm;
DOC *a,*b;
{
  kernel_cache_statistic++;
  switch(kernel_parm->kernel_type) {
    case 0: /* linear */ 
            return((CFLOAT)sprod_ss(a->words,b->words)); 
    case 1: /* polynomial */
            return((CFLOAT)pow(kernel_parm->coef_lin*sprod_ss(a->words,b->words)+kernel_parm->coef_const,(double)kernel_parm->poly_degree)); 
    case 2: /* radial basis function */
            return((CFLOAT)exp(-kernel_parm->rbf_gamma*(a->twonorm_sq-2*sprod_ss(a->words,b->words)+b->twonorm_sq)));
    case 3: /* sigmoid neural net */
            return((CFLOAT)tanh(kernel_parm->coef_lin*sprod_ss(a->words,b->words)+kernel_parm->coef_const)); 
    case 4: /* custom-kernel supplied in file kernel.h*/
            return((CFLOAT)custom_kernel(kernel_parm,a,b)); 
    default: printf("Error: Unknown kernel function\n"); exit(1);
  }
}

double sprod_ss(a,b) /* compute the inner product of two sparse vectors */
WORD *a,*b;          
{
    register FVAL sum=0;
    register WORD *ai,*bj;
    ai=a;
    bj=b;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	ai++;
      }
      else {
	sum+=ai->weight * bj->weight;
	ai++;
	bj++;
      }
    }
    return((double)sum);
}

double model_length_s(model,kernel_parm) /* compute length of weight vector */
MODEL *model;
KERNEL_PARM *kernel_parm;
{
  register long i,j;
  register double sum=0,alphai;
  register DOC *supveci;

  for(i=1;i<model->sv_num;i++) {  
    alphai=model->alpha[i];
    supveci=model->supvec[i];
    for(j=1;j<model->sv_num;j++) {
      sum+=alphai*model->alpha[j]
	   *kernel(kernel_parm,supveci,model->supvec[j]);
    }
  }
  return(sqrt(sum));
}

void clear_vector_n(vec,n)
double *vec;
long n;
{
  register long i;
  for(i=0;i<=n;i++) vec[i]=0;
}

void add_vector_ns(vec_n,vec_s,faktor) 
double *vec_n;
WORD *vec_s;
double faktor;
{
  register WORD *ai;
  ai=vec_s;
  while (ai->wnum) {
    vec_n[ai->wnum]+=(faktor*ai->weight);
    ai++;
  }
}

double sprod_ns(vec_n,vec_s) 
double *vec_n;
WORD *vec_s;
{
  register double sum=0;
  register WORD *ai;
  ai=vec_s;
  while (ai->wnum) {
    sum+=(vec_n[ai->wnum]*ai->weight);
    ai++;
  }
  return(sum);
}

void read_documents(docfile,docs,label,max_words_doc,ll,totwords,totdoc)
char *docfile;
DOC *docs;
long *label,max_words_doc,ll,*totwords,*totdoc;
{
  char *line;
  DOC doc;
  long dnum=0,wpos,i;
  long doc_label;
  FILE *docfl;

  line = (char *)my_malloc(sizeof(char)*ll);

  if ((docfl = fopen (docfile, "r")) == NULL)
  { perror (docfile); exit (1); }

  while(fscanf(docfl,"#%[^\n]\n", line));/* first lines may contain comments */

  doc.words = (WORD *)my_malloc(sizeof(WORD)*(max_words_doc+10));
  if(verbosity>=1) {
    printf("Reading examples into memory..."); fflush(stdout);
  }
  dnum=0;
  (*totwords)=0;
  while(fscanf(docfl,"%[^\n]\n", line) != EOF) {
    if(!parse_document(line,&doc,&doc_label,&wpos,max_words_doc)) {
      printf("\nParsing error in line %d!\n",dnum);
      exit(1);
    }
    label[dnum]=doc_label;
    if((wpos>1) && ((doc.words[wpos-2]).wnum>(*totwords))) 
      (*totwords)=(doc.words[wpos-2]).wnum;
    docs[dnum].words = (WORD *)my_malloc(sizeof(WORD)*wpos);
    docs[dnum].docnum=dnum;
    for(i=0;i<wpos;i++) { 
      docs[dnum].words[i]=doc.words[i];
    }
    docs[dnum].twonorm_sq=doc.twonorm_sq;
    dnum++;  
    if(verbosity>=1) {
      if((dnum % 100) == 0) {
	printf("%ld..",dnum); fflush(stdout);
      }
    }
  }
  fclose(docfl);
  free(line);
  free(doc.words);
  if(verbosity>=1) {
    fprintf(stdout, "OK. (%ld examples read)\n", dnum);
  }
  (*totdoc)=dnum;
}

int parse_document(line,doc,label,numwords,max_words_doc)
char *line;
DOC *doc;
long *label,*numwords,max_words_doc;
{
  register long wpos,pos;
  long wnum;
  double weight;

  pos=0;
  while(line[pos]) {      /* cut off comments */
    if(line[pos] == '#') {
      line[pos]=0;
    }
    else {
      pos++;
    }
  }
  wpos=0;
  if(sscanf(line,"%ld",label) == EOF) return(0);
  pos=0;
  while(line[pos]==' ') pos++;
  while(line[pos]>' ') pos++;
  while((sscanf(line+pos,"%ld:%lf",&wnum,&weight) != EOF) && 
	(wpos<max_words_doc)) {
    while(line[pos++]==' ');
    while(line[++pos]>' ');
    if(wnum<=0) { 
      perror ("Feature numbers must be larger or equal to 1!!!\n"); 
      exit (1); 
    }
    (doc->words[wpos]).wnum=wnum;
    (doc->words[wpos]).weight=weight; 
    /*    (doc->words[wpos]).weight=weight;  */
    wpos++;
  }
  (doc->words[wpos]).wnum=0;
  (*numwords)=wpos+1;
  doc->docnum=-1;
  doc->twonorm_sq=sprod_ss(doc->words,doc->words);
  return(1);
}

void nol_ll(file,nol,wol,ll) /* grep through file and count number of lines, */
char *file;                  /* maximum number of spaces per line, and */
long *nol,*wol,*ll;          /* longest line. */
{
  FILE *fl;
  char c;
  long current_length,current_wol;

  if ((fl = fopen (file, "r")) == NULL)
  { perror (file); exit (1); }
  current_length=0;
  current_wol=0;
  (*ll)=0;
  (*nol)=1;
  (*wol)=0;
  while((c=getc(fl)) != EOF) {
    current_length++;
    if(c == ' ') {
      current_wol++;
    }
    if(c == '\n') {
      (*nol)++;
      if(current_length>(*ll)) {
	(*ll)=current_length;
      }
      if(current_wol>(*wol)) {
	(*wol)=current_wol;
      }
      current_length=0;
      current_wol=0;
    }
  }
  fclose(fl);
}


long min(a,b)
long a,b;
{
  if(a<b)
    return(a);
  else
    return(b);
}

long max(a,b)
long a,b;
{
  if(a>b)
    return(a);
  else
    return(b);
}

long get_runtime() 
{
  struct tms buffer;
  times(&buffer);
  return((long)(buffer.tms_utime+buffer.tms_stime));
}

int pairComp (const void* e1, const void* e2)
{
  PAIR *p1, *p2;
  p1 = (PAIR*)e1;
  p2 = (PAIR*)e2;

  if (p1->value > p2->value) {
    return (1);
  } else if (p1->value < p2->value) {
    return (-1);
  } else {
    return (0);
  }
}
 
void *my_malloc(size) 
long size;
{
  void *ptr;
  ptr=(void *)malloc(size);
  if(!ptr) { 
    perror ("Out of memory!\n"); 
    exit (1); 
  }
  return(ptr);
}

void copyright_notice()
{
  printf("\nCopyright: Thorsten Joachims, thorsten@ls8.cs.uni-dortmund.de\n\n");
  printf("This software is available for non-commercial use only. It must not\n");
  printf("be modified and distributed without prior permission of the author.\n");
  printf("The author is not responsible for implications from the use of this\n");
  printf("software.\n\n");
}
