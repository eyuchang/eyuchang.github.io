/***********************************************************************/
/*                                                                     */
/*   svm_classify.c                                                    */
/*                                                                     */
/*   Classification module of Support Vector Machine.                  */
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
/************************************************************************/

# include "svm_common.h"

#define EPS 1e-10
/* maximum number of documents that we can ignore */
#define MAX_IGNORE 10000
/* maximum number of documents that we already know the labels of */
#define MAX_QUERY 10000

char docfile[200];
char modelfile[200];
char predictionsfile[200];
char ignorelistfile[200];
char querylistfile[200];

void read_model(char *, MODEL *, long, long, KERNEL_PARM *);
void read_list_file(char*, long* , int*);
void read_input_parameters(int, char **, char *, char *, char *, char *,
			   char*, long *, long *);
void print_help();
double ComputeBreakEven(PAIR *pairs, long totdoc);
/* Compute precision and recall. Return 1 if there's been a change 
   in the sign of (precision - recall) since the last time */
int ComputePrecisionRecall(double precision[2], double recall[2], 
			   long i, PAIR *pairs, long totdoc);

/* Computes the accuracy of the top top_limit documents that we
   think are positive. */
float ComputeLimitedPrecision(int top_limit, PAIR *pairs, long totdoc);

main (argc, argv)
int argc;
char *argv[];
{
  int i,j;
  DOC doc;   /* test example */
  long max_docs,max_words_doc,lld,llsv;
  long max_sv,max_words_sv,totdoc=0,doc_label;
  long correct=0,incorrect=0,no_accuracy=0;
  long res_a=0,res_b=0,res_c=0,res_d=0,wnum,pred_format;
  long ignorelist[MAX_IGNORE]; /* documents to ignore when classifying */
  long querylist[MAX_QUERY]; /* documents to boost confidence on when classifying since we know their labels for certain */
  int ignorelist_size;
  int querylist_size;
  long doc_index;
  int ignore_this;
  double t1,runtime=0;
  double dist;
  char *line; 
  FILE *predfl,*docfl;
  KERNEL_PARM kernel_parm;
  MODEL model; 
  PAIR *pairs;

  read_input_parameters(argc,argv,docfile,modelfile,predictionsfile,
			ignorelistfile, querylistfile, 
			&verbosity,&pred_format);

  nol_ll(docfile,&max_docs,&max_words_doc,&lld); /* scan size of input file */
  max_words_doc+=2;
  lld+=2;
  nol_ll(modelfile,&max_sv,&max_words_sv,&llsv); /* scan size of model file */
  max_words_sv+=2;
  llsv+=2;

  model.supvec = (DOC **)my_malloc(sizeof(DOC *)*max_sv);
  model.alpha = (double *)my_malloc(sizeof(double)*max_sv);

  line = (char *)my_malloc(sizeof(char)*lld);
  doc.words = (WORD *)my_malloc(sizeof(WORD)*(max_words_doc+10));

  pairs = (PAIR *)my_malloc(sizeof(PAIR)*(max_docs));

  read_model(modelfile,&model,max_words_sv,llsv,&kernel_parm);

  read_list_file(ignorelistfile,ignorelist,&ignorelist_size);
  read_list_file(querylistfile,querylist,&querylist_size);
  
  if(verbosity>=2) {
    printf("Classifying test examples.."); fflush(stdout);
  }

  if ((docfl = fopen (docfile, "r")) == NULL)
  { perror (docfile); exit (1); }
  if ((predfl = fopen (predictionsfile, "w")) == NULL)
  { perror (predictionsfile); exit (1); }

  while(fscanf(docfl,"#%[^\n]\n", line));/* first lines may contain comments */

  doc_index = 0;

  while(fscanf(docfl,"%[^\n]\n", line) != EOF) {
    
    /* check to see if we should ignore this document */
    ignore_this = 0;
    for(j=0; j< ignorelist_size; j++) {
      if(ignorelist[j] == doc_index) {
       ignore_this = 1;
      }
    }
    if(ignore_this == 1) {
      if(verbosity>=2) {
	printf("Ignoring document: %ld\n",doc_index);
      }      
      doc_index++;
      continue;
    }

    parse_document(line,&doc,&doc_label,&wnum,max_words_doc);
    t1=get_runtime();
    dist=classify_example(&model,&doc,&kernel_parm);

    /* check to see if we already know the label of this document */
    for(j=0; j< querylist_size; j++) {
      if(querylist[j] == doc_index) {
	if(verbosity>=2) {
	  printf("Boosting confidence of document: %ld from %f to %f\n",doc_index, dist, doc_label*INFINITY);
	}
	dist = doc_label*INFINITY;
      }
    }

    pairs[totdoc].value = dist;
    pairs[totdoc].label = doc_label;


    totdoc++;
    runtime+=(get_runtime()-t1);
    if(dist>0) {
      if(pred_format==0) { /* old weired output format */
	fprintf(predfl,"%.8g:+1 %.8g:-1\n",dist,-dist);
      }
      if(doc_label==1) correct++; else incorrect++;
      if(doc_label==1) res_a++; else res_b++;
    }
    else {
      if(pred_format==0) { /* old weired output format */
	fprintf(predfl,"%.8g:-1 %.8g:+1\n",-dist,dist);
      }
      if(doc_label==-1) correct++; else incorrect++;
      if(doc_label==1) res_c++; else res_d++;
    }
    if(pred_format==1) { /* output the value of decision function */
      fprintf(predfl,"%.8g\n",dist);
    }
    if((doc_label*doc_label) != 1) { no_accuracy=1; }
    if(verbosity>=2) {
      if(totdoc % 100 == 0) {
	printf("%ld..",totdoc); fflush(stdout);
      }
    }
    doc_index++;
  }  

  free(line);
  free(doc.words);
  free(model.supvec);
  free(model.alpha);
  if(verbosity>=2) {
    printf("done\n");
    printf("Runtime (without IO) in cpu-seconds: %.2f\n",
	   (float)(runtime/100.0));
    
  }

  qsort (pairs, totdoc, sizeof(PAIR), pairComp);
  
  if((!no_accuracy) && (verbosity>=1)) {
    printf("Accuracy on test set: %.2f%% (%ld correct, %ld incorrect, %ld total)\n",(float)(correct)*100.0/totdoc,correct,incorrect,totdoc);
    printf("Precision/recall on test set: %.2f%%/%.2f%%\n",(float)(res_a)*100.0/(res_a+res_b),(float)(res_a)*100.0/(res_a+res_c));
    printf("Precision/recall breakeven point: %.2f%%\n", 100.0 * (float)ComputeBreakEven(pairs, totdoc));
    for(i=10; i< 201; i=i+10) {
      if(i<totdoc) {
      printf("Precision on top %d docs: %.2f%%\n", i, 100.0 * (float)ComputeLimitedPrecision(i, pairs, totdoc));
      }
    }
  }
}

void read_model(modelfile,model,max_words,ll,kernel_parm)
char *modelfile;
MODEL *model;
long max_words,ll;
KERNEL_PARM *kernel_parm;
{
  FILE *modelfl;
  long j,i;
  char *line;
  WORD *words;
  register long wpos;
  long wnum,pos;
  double weight;
  char version_buffer[100];

  if(verbosity>=1) {
    printf("Reading model..."); fflush(stdout);
  }
  words = (WORD *)my_malloc(sizeof(WORD)*(max_words+10));
  line = (char *)my_malloc(sizeof(char)*ll);

  if ((modelfl = fopen (modelfile, "r")) == NULL)
  { perror (modelfile); exit (1); }

  fscanf(modelfl,"SVM-light Version %s\n",version_buffer);
  if(strcmp(version_buffer,VERSION)) {
    perror ("Version of model-file does not match version of svm_classify!"); 
    exit (1); 
  }
  fscanf(modelfl,"%ld%*[^\n]\n", &kernel_parm->kernel_type);  
  fscanf(modelfl,"%ld%*[^\n]\n", &kernel_parm->poly_degree);
  fscanf(modelfl,"%lf%*[^\n]\n", &kernel_parm->rbf_gamma);
  fscanf(modelfl,"%lf%*[^\n]\n", &kernel_parm->coef_lin);
  fscanf(modelfl,"%lf%*[^\n]\n", &kernel_parm->coef_const);
  fscanf(modelfl,"%s%*[^\n]\n", kernel_parm->custom);

  fscanf(modelfl,"%ld%*[^\n]\n", &model->sv_num);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->b);

  for(i=1;i<model->sv_num;i++) {
    fscanf(modelfl,"%[^\n]\n", line);
    pos=0;
    wpos=0;
    sscanf(line,"%lf",&model->alpha[i]);
    while(line[++pos]>' ');
    while((sscanf(line+pos,"%ld:%lf",&wnum,&weight) != EOF) 
	  && (wpos<max_words)) {
      while(line[++pos]>' ');
      words[wpos].wnum=wnum;
      words[wpos].weight=weight; 
      wpos++;
    } 
    model->supvec[i] = (DOC *)my_malloc(sizeof(DOC));
    (model->supvec[i])->words = (WORD *)my_malloc(sizeof(WORD)*(wpos+1));
    for(j=0;j<wpos;j++) {
      (model->supvec[i])->words[j]=words[j]; 
    }
    ((model->supvec[i])->words[wpos]).wnum=0;
    (model->supvec[i])->twonorm_sq = sprod_ss((model->supvec[i])->words,
					      (model->supvec[i])->words);
    (model->supvec[i])->docnum = -1;
  }
  fclose(modelfl);
  free(line);
  free(words);
  if(verbosity>=1) {
    fprintf(stdout, "OK. (%d support vectors read)\n",(int)(model->sv_num-1));
  }
}

void read_list_file(listfile,list,list_size_ptr)
char* listfile;
long* list;
int* list_size_ptr;
{
  FILE *listfl;
  long list_doc;

  *list_size_ptr=0;

  if(verbosity>=1) {
    printf("Reading list file..."); fflush(stdout);
  }

  if(strncmp("***none***",listfile,10) == 0) {
    list = NULL;
    *list_size_ptr=0;
    return;
  }

  if ((listfl = fopen (listfile, "r")) == NULL)
  { perror (modelfile); exit (1); }

  
  while(fscanf(listfl,"%ld%*[^\n]\n", &list_doc) != EOF) {
    printf("doc: %ld\n", list_doc);
    list[*list_size_ptr] = list_doc;  
    (*list_size_ptr) = (*list_size_ptr)+1;
  }
  
  fclose(listfl);
  if(verbosity>=1) {
    fprintf(stdout, "OK. (%d documents read)\n",*list_size_ptr);
  }
}

double ComputeBreakEven(PAIR *pairs, long totdoc) 
{
  long i;
  int signChange=0;
  double precision[2], recall[2];
  double currentBreakEven=1.0, newBreakEven;

  for(i=-1; i<totdoc-1; i++) {
    signChange=ComputePrecisionRecall(precision, recall, i, pairs, totdoc);
    if (signChange==1) {
      signChange = 0;
      newBreakEven = (precision[0]*recall[1] - precision[1]*recall[0])/(precision[0]-precision[1]-recall[0]+recall[1]);
      if(newBreakEven < currentBreakEven) {
	currentBreakEven = newBreakEven;
      }
    }
  }
  return (currentBreakEven);
}

/* Computes the accuracy of the top top_limit documents that we
   think are positive. */

float ComputeLimitedPrecision(int top_limit, 
			      PAIR *pairs, 
			      long totdoc)
{
  int correct=0, num_docs=0;
  long j;

  /*
  for(j=0; j< totdoc; j++) {
        printf("distance: %f\t value: %d\n", pairs[j].value, pairs[j].label);
  }
  */

  /* The docs are sorted in order of their (signed) distance from the margin. */

  for (j=totdoc-1; j>totdoc-1-top_limit; j--) {
      num_docs++;
      if(pairs[j].label==1) {
        printf("distance: %f\t value: %d\n", pairs[j].value, pairs[j].label);
	correct++;
      }

  }
  return(correct * 1.0 / num_docs);
}

int ComputePrecisionRecall(double precision[2], double recall[2], 
			   long i, PAIR *pairs, long totdoc)
{
  long j, res_a=0, res_b=0, res_c=0, res_d=0, correct=0, incorrect=0;
  
  /* The docs are sorted in order of there (signed) distance from the margin. 
     Classify the first i+1 docs as class -1 and all the others as class 1 */
  
  for (j=0; j<totdoc; j++) {
    if(j>i) {
      if(pairs[j].label==1) {
	correct++;
	res_a++;
      }
      else {
	incorrect++;
	res_b++;
      }
    }
    else {
      if(pairs[j].label==-1) {
	correct++; 
	res_d++;
      }
      else {
	incorrect++;
	res_c++;
      }
    }
  }

  precision[1] = precision[0];
  recall[1] = recall[0];
  
  precision[0] = (double)res_a/(res_a+res_b);
  recall[0] = (double)res_a/(res_a+res_c);

  /* If we have a swap and we are not just starting out then we have a change in sign */
  
  if (((precision[0] >= recall[0] - EPS && precision[1] < recall[1]) 
      || (precision[1] >= recall[1] - EPS && precision[0] < recall[0]))
      && i != -1) {
    return (1);
  } else {
    return(0);
  }
}

void read_input_parameters(argc,argv,docfile,modelfile,predictionsfile,
			   ignorelistfile, querylistfile,
			   verbosity,pred_format)
int argc;
char *argv[],*docfile,*modelfile,*predictionsfile;
char *ignorelistfile, *querylistfile;
long *verbosity,*pred_format;
{
  long i;
  
  /* set default */
  strcpy (modelfile, "svm_model");
  strcpy (predictionsfile, "svm_predictions"); 
  strcpy (ignorelistfile, "***none***"); 
  strcpy (querylistfile, "***none***"); 
  (*verbosity)=2;
  (*pred_format)=1;

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) 
      { 
      case 'h': print_help(); exit(0);
      case 'v': i++; (*verbosity)=atol(argv[i]); break;
      case 'f': i++; (*pred_format)=atol(argv[i]); break;
      case 'i': i++; strcpy(ignorelistfile,argv[i]); break;
      case 'q': i++; strcpy(querylistfile,argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n",argv[i]);
	       print_help();
	       exit(0);
      }
  }
  if((i+1)>=argc) {
    printf("\nNot enough input parameters!\n\n");
    print_help();
    exit(0);
  }
  strcpy (docfile, argv[i]);
  strcpy (modelfile, argv[i+1]);
  if((i+2)<argc) {
    strcpy (predictionsfile, argv[i+2]);
  }
  if(((*pred_format) != 0) && ((*pred_format) != 1)) {
    printf("\nOutput format can only take the values 0 or 1!\n\n");
    print_help();
    exit(0);
  }
}

void print_help()
{
  printf("\nSVM-light %s: Support Vector Machine, classification module     %s\n",VERSION,VERSION_DATE);
  copyright_notice();
  printf("   usage: svm_classify [options] example_file model_file output_file\n\n");
  printf("options: -h            -> this help\n");
  printf("         -i filename   -> list of document indicies to ignore\n");
  printf("         -q filename   -> list of document indicies that we know
        \n                          the labels of already\n");
  printf("         -v [0..3]     -> verbosity level (default 2)\n");
  printf("         -f [0,1]      -> 0: old output format of V1.0\n");
  printf("                       -> 1: output the value of decision function (default)\n\n");
}


