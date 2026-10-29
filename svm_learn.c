/***********************************************************************/
/*                                                                     */
/*   svm_learn.c                                                       */
/*                                                                     */
/*   Learning module of Support Vector Machine.                        */
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

/* When doing simultaneous querying, try to choose exmaples that are
   mutally distant. Comment this out to use dumb sim querying (i.e. just
   choosing the k closest instances to the hyperplane) */

// #define SMART_SIM_QUERIES

#include "svm_common.h"

char docfile[200];           /* file with training examples */
char modelfile[200];         /* file for resulting classifier */
char trans_modelfile[200];         /* file for resulting transductive
				      classifier if -z 2 is set */
char every_label_file[200];  /* file for active lerning labels */

/* interface to QP-solver */
double *optimize_qp(QP *, double *, long, double *, LEARN_PARM *);
int nx = 400;                /* Maximum size of QP-subproblems */
void active_learn(DOC *,long *,long,long,LEARN_PARM *,KERNEL_PARM *,KERNEL_CACHE *,
		  MODEL *,long, long *);
void svm_simple_learn(DOC *,long *,long,long,LEARN_PARM *,KERNEL_PARM *,KERNEL_CACHE *,
		      MODEL *, double *);
void   svm_learn(DOC *, long *, long, long, LEARN_PARM *, KERNEL_PARM *, 
		 KERNEL_CACHE *, MODEL *, int);
double compute_objective_function(double *, double *, long *, long *);
void   clear_index(long *);
void   add_to_index(long *, long);
long   compute_index(long *,long, long *);
void   optimize_svm(DOC *, long *, long *, long *, long *, MODEL *, long, 
		    long *, long, double *, double *, LEARN_PARM *, CFLOAT *, 
		    KERNEL_PARM *, QP *, double *);
void   compute_matrices_for_optimization(DOC *, long *, long *, long *, 
					 long *, long *, MODEL *, double *, 
					 double *, long, long, LEARN_PARM *, 
					 CFLOAT *, KERNEL_PARM *, QP *);
long   calculate_svm_model(DOC *, long *, long *, double *, double *, 
			   double *, LEARN_PARM *, long *, MODEL *);
long   check_optimality(MODEL *, long *, long *, double *, double *,long, 
			LEARN_PARM *,double *, double, long *, long *, long *,
			long *, long, KERNEL_PARM *);
long   identify_inconsistent(double *, long *, long *, long *, LEARN_PARM *, 
			     long *, long *);
long   incorporate_unlabeled_examples(MODEL *, long *,long *, long *,
				      double *, double *, long, double *,
				      long *, long *, long, KERNEL_PARM *,
				      LEARN_PARM *);
void   update_linear_component(DOC *, long *, long *, double *, double *, 
			       long *, long, long, KERNEL_PARM *, 
			       KERNEL_CACHE *, double *,
			       CFLOAT *, double *);
long   select_next_qp_subproblem_grad(long *, long *, double *, double *, long,
				      long, LEARN_PARM *, long *, long *, 
				      long *, double *, long *, KERNEL_CACHE *,
				      long *, long *);
long   select_next_qp_subproblem_grad_cache(long *, long *, double *, double *,
					    long, long, LEARN_PARM *, long *, 
					    long *, long *, double *, long *,
					    KERNEL_CACHE *, long *, long *);
void   select_top_n(double *, long, long *, long);
long   shrink_problem(LEARN_PARM *, long *, long *,long *, long *, long *, 
		      long, long, long, double *, double **, long *);
void   reactivate_inactive_examples(long *, long *, double *, double **, 
				    double *, long, long, long, LEARN_PARM *, 
				    long *, long *, long *, long, DOC *, 
				    KERNEL_PARM *,KERNEL_CACHE *, MODEL *, 
				    CFLOAT *, double *, double *);

/* cache kernel evalutations to improve speed */
void   get_kernel_row(KERNEL_CACHE *,DOC *, long, long, long *, CFLOAT *, 
		      KERNEL_PARM *);
void   get_kernel_row_offset(KERNEL_CACHE *,DOC *, long, long, long *, 
		      CFLOAT *, KERNEL_PARM *,long,long);
void   cache_kernel_row(KERNEL_CACHE *,DOC *, long, KERNEL_PARM *);
void   cache_multiple_kernel_rows(KERNEL_CACHE *,DOC *, long *, long, 
				  KERNEL_PARM *);
void   kernel_cache_shrink(KERNEL_CACHE *,long, long, long *);
void   kernel_cache_init(KERNEL_CACHE *,long, long);
void   kernel_cache_quick_init(KERNEL_CACHE *,long, long);
void   kernel_cache_cleanup(KERNEL_CACHE *);
long   kernel_cache_malloc(KERNEL_CACHE *);
void   kernel_cache_free(KERNEL_CACHE *,long);
long   kernel_cache_free_lru(KERNEL_CACHE *);
CFLOAT *kernel_cache_clean_and_malloc(KERNEL_CACHE *,long);
long   kernel_cache_touch(KERNEL_CACHE *,long);
long   kernel_cache_check(KERNEL_CACHE *,long);

double estimate_loo_bound(MODEL *, long *, long *, long);
double estimate_margin_vcdim(MODEL *, double, KERNEL_PARM *);
double estimate_sphere(MODEL *, KERNEL_PARM *);
void   write_model(char *, MODEL *, KERNEL_PARM *);
void   write_queries(char*, long*, int, int);
void   write_prediction(char *, MODEL *, double *, double *, long *, long *,
			long, LEARN_PARM *);
void   write_alphas(char *, double *, long *, long);
int    inverseLongComp (const void*, const void*);
void   read_input_parameters(int, char **, char *, char *,long *, long *, 
			     LEARN_PARM *, KERNEL_PARM *, long *, int *,
			     char *, char *, int *, long *, int *, int *);
void   print_help();

typedef struct cache_parm_s {
  KERNEL_CACHE *kernel_cache;
  CFLOAT *cache;
  DOC *docs; 
  long m;
  KERNEL_PARM *kernel_parm;
  long offset,stepsize;
} cache_parm_t;

/*---------------------------------------------------------------------------*/

main (argc, argv)
int argc;
char *argv[];
{  
  int i,j,k,l;
  int trans, active_type, active_margin_ratio, temp=0;
  DOC *temp_docs;
  long *every_doc_label, num_query_batches;
  DOC *docs;  /* training examples */
  long *label,max_docs,max_words_doc;
  long totwords,totdoc,ll;
  long max_label_docs,temp_max_word_docs;
  long temp_totwords,temp_totdoc,temp_ll;

  long kernel_cache_size;
  KERNEL_CACHE kernel_cache;
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  MODEL model;

  /* for active learning */
  long query, candidate_doc, candidate_label, *unlabeled2dnum, num_unlabeled;
  long num_labeled;
  long best_candidate_so_far, tot_iteration;
  long hybrid_threshold; /* number of queries to perform ratio/margin method on first
			    before resorting to simple method */
  long* queries;
  double best_margin_ratio_so_far;
  double current_ratio, candidate_model_length[2];
  double current_distance, min_distance_so_far, max_distance_so_far;
  LEARN_PARM learn_parm_org;
  KERNEL_PARM kernel_parm_org;

  char temp_modelfile[216];         /* file for resulting classifiers when
                                       performing active learning */
  char file_suffix[16];        /* string to append to temp_modelfile string */
  char queryfile[216]; /* the queries we made */
  char temp_queryfile[216]; /* the queries we made */

  /* For multiple querying of docs */

  /* Idea: we create a "candidate set" of query documents comprising of a fixed number of "good" documents.
     From this set we select a smaller set of documents (the "query set") that we will simultaneously query. */

  double* row_sums=0; /* r_i in the paper */
  double  candidate_row_sum; /* r_i in the paper */
  double* distances_to_candidate_doc=0; /* distances from a candidate doc to the docs in our current query set */
  long* candidate_queries_indices=0; /* indices of all of the candidate docs */
  PAIR* candidate_queries_pairs=0; /* (score, index) pairs for the candidate queries */
  long* simultaneous_queries_indices=0; /* our query set */
  double max_distance = 0.0;
  int max_distance_index = 0;
  double spair; /* Total sum of paired distances between all of the query set docs. */ 
  double original_sum_of_paired_distances; /* Sum of paired distances between the current simulataneous
					      queries that we're going to ask */
  int num_simultaneous_queries; /* Number of simultaneous queries to ask */
  int num_simultaneous_queries_iterations; /* number of attempts to get a better query set */
  int num_candidate_queries; /* Number of good docs to consider choosing the simulateous queries from */ 
  int skip_this;

  long runtime_startclock=0, runtime_queries=0, runtime_finalmdl=0;
  

  /* seed drand48() */
  /*  srand48(gethrtime() % (1 << 24)); */
  srand48((long)time(NULL));
  drand48();

  read_input_parameters(argc,argv,docfile,modelfile,&verbosity,
			&kernel_cache_size,&learn_parm,&kernel_parm, 
			&num_query_batches, &trans, every_label_file, 
			trans_modelfile, &active_type, &hybrid_threshold,
			&num_simultaneous_queries, &num_candidate_queries);

  strcpy(temp_modelfile,modelfile);
  strcpy(temp_queryfile,queryfile);

  if(active_type == 0) {
    strcpy(queryfile,"queries.maxratio.log");
  } else if(active_type == 1) {
    strcpy(queryfile,"queries.simple.log");
  } else if(active_type == 2) {
    strcpy(queryfile,"queries.maxmargin.log");
  } else {
    strcpy(queryfile,"queries.random.log");
  }

  if(verbosity>=1) {
    printf("Scanning examples..."); fflush(stdout);
  }
  nol_ll(docfile,&max_docs,&max_words_doc,&ll); /* scan size of input file */
  max_words_doc+=2;
  ll+=2;
  max_docs+=2;
  if(verbosity>=1) {
    printf("done\n"); fflush(stdout);
  }
  docs = (DOC *)my_malloc(sizeof(DOC)*max_docs);
  label = (long *)my_malloc(sizeof(long)*max_docs);
  read_documents(docfile,docs,label,max_words_doc,ll,&totwords,&totdoc);

  queries = NULL;

  if(num_query_batches>0) { /* if we are doing active learning then read in labels file */
    queries = (long *)my_malloc(sizeof(long)*(num_query_batches*num_simultaneous_queries+2));
    if(verbosity>=1) {
      printf("Scanning label file..."); fflush(stdout);
    }
    nol_ll(every_label_file,&max_label_docs,&temp_max_word_docs,&temp_ll); /* scan size of input file */
    temp_max_word_docs+=2;
    temp_ll+=2;
    max_label_docs+=2;
    if(verbosity>=1) {
      printf("done\n"); fflush(stdout);
    }

    temp_docs = (DOC *)my_malloc(sizeof(DOC)*max_label_docs);
    every_doc_label = (long *)my_malloc(sizeof(long)*max_label_docs);
    
    read_documents(every_label_file,temp_docs,every_doc_label,
		   temp_max_word_docs,temp_ll,&temp_totwords,&temp_totdoc);
    free(temp_docs);  

    if (totdoc != temp_totdoc) {
      printf("Label file and training file contain different number of documents!\n");
      exit(0);
    }
    
    unlabeled2dnum = (long *)my_malloc(sizeof(long)*temp_totdoc);
    
    /* index unlabeled docs */
    num_unlabeled=0;
    for(i=0;i<temp_totdoc;i++) {
      if(label[i]==0) {
	unlabeled2dnum[num_unlabeled]=i;
	num_unlabeled++;
      }
    }
    num_labeled = totdoc - num_unlabeled;

    if (num_query_batches*num_simultaneous_queries > num_unlabeled) {
      printf("Requested to ask more queries than there are unlabeled documents!\n");
      exit(0);
    }
  }

  if(num_query_batches>0) {
    row_sums = (double *)my_malloc(sizeof(double)*num_simultaneous_queries);
    distances_to_candidate_doc = (double *)my_malloc(sizeof(double)*num_simultaneous_queries);
    candidate_queries_indices = (long *)my_malloc(sizeof(long)*num_candidate_queries);
    candidate_queries_pairs= (PAIR *)my_malloc(sizeof(PAIR)*num_candidate_queries);
    simultaneous_queries_indices= (long *)my_malloc(sizeof(long)*num_simultaneous_queries);
  }

  num_simultaneous_queries_iterations = 0;
#ifdef SMART_SIM_QUERIES
  num_simultaneous_queries_iterations = num_candidate_queries*10;
#endif

  /* store original settings */

  learn_parm_org.biased_hyperplane = learn_parm.biased_hyperplane;
  learn_parm_org.remove_inconsistent = learn_parm.remove_inconsistent;
  learn_parm_org.skip_final_opt_check = learn_parm.skip_final_opt_check;
  learn_parm_org.svm_maxqpsize = learn_parm.svm_maxqpsize;
  learn_parm_org.svm_iter_to_shrink = learn_parm.svm_iter_to_shrink;
  learn_parm_org.svm_c = learn_parm.svm_c;
  learn_parm_org.transduction_posratio = learn_parm.transduction_posratio;
  learn_parm_org.svm_costratio = learn_parm.svm_costratio;
  learn_parm_org.svm_costratio_unlab = learn_parm.svm_costratio_unlab;
  learn_parm_org.svm_unlabbound = learn_parm.svm_unlabbound;
  learn_parm_org.epsilon_crit = learn_parm.epsilon_crit;
  learn_parm_org.epsilon_a = learn_parm.epsilon_a;
  kernel_parm_org.kernel_type = kernel_parm.kernel_type;
  kernel_parm_org.poly_degree = kernel_parm.poly_degree;
  kernel_parm_org.rbf_gamma = kernel_parm.rbf_gamma;
  kernel_parm_org.coef_lin = kernel_parm.coef_lin;
  kernel_parm_org.coef_const = kernel_parm.coef_const;

  for (query=0;query<num_query_batches && num_unlabeled > 0;query++) {

    if(active_type == 3) { /* random sampling */
      if(kernel_parm.kernel_type == LINEAR) { /* don't need the cache */
	svm_simple_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
			 NULL,&model,&candidate_model_length[0]);
      }
      else {
	kernel_cache_init(&kernel_cache,totdoc,kernel_cache_size);	    
	svm_simple_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
			 &kernel_cache,&model,&candidate_model_length[0]);
	kernel_cache_cleanup(&kernel_cache);
      }      

      /* write this modelfile */
      snprintf(file_suffix,16,".%d",query*num_simultaneous_queries+num_labeled);
      strcat(temp_modelfile, file_suffix);
      write_model(temp_modelfile,&model,&kernel_parm);
      strcpy(temp_modelfile, modelfile);

      /* write query file */
      strcat(temp_queryfile, file_suffix);
      write_queries(temp_queryfile,queries,query,num_simultaneous_queries);
      strcpy(temp_queryfile, queryfile);

      /* randomly select an unlabeled example */
      best_candidate_so_far = drand48() * num_unlabeled;
      queries[query] = unlabeled2dnum[best_candidate_so_far];

      /* restore original settings */	  
      free(model.supvec);
      free(model.alpha);
      free(model.index);
      
      learn_parm.biased_hyperplane = learn_parm_org.biased_hyperplane;
      learn_parm.remove_inconsistent = learn_parm_org.remove_inconsistent;
      learn_parm.skip_final_opt_check = learn_parm_org.skip_final_opt_check;
      learn_parm.svm_maxqpsize = learn_parm_org.svm_maxqpsize;
      learn_parm.svm_iter_to_shrink = learn_parm_org.svm_iter_to_shrink;
      learn_parm.svm_c = learn_parm_org.svm_c;
      learn_parm.transduction_posratio = learn_parm_org.transduction_posratio;
      learn_parm.svm_costratio = learn_parm_org.svm_costratio;
      learn_parm.svm_costratio_unlab = learn_parm_org.svm_costratio_unlab;
      learn_parm.svm_unlabbound = learn_parm_org.svm_unlabbound;
      learn_parm.epsilon_crit = learn_parm_org.epsilon_crit;
      learn_parm.epsilon_a = learn_parm_org.epsilon_a;
      kernel_parm.kernel_type = kernel_parm_org.kernel_type;
      kernel_parm.poly_degree = kernel_parm_org.poly_degree;
      kernel_parm.rbf_gamma = kernel_parm_org.rbf_gamma;
      kernel_parm.coef_lin = kernel_parm_org.coef_lin;
      kernel_parm.coef_const = kernel_parm_org.coef_const;

      if(verbosity>=1) {
	printf("Querying document number: %ld and retraining\n", unlabeled2dnum[best_candidate_so_far]);
	fflush(stdout);
      }

      /* set the queried document's label to the correct label */
      label[unlabeled2dnum[best_candidate_so_far]] = every_doc_label[unlabeled2dnum[best_candidate_so_far]];
      
      /* remove best_candidate_so_far from the unlabeled list */
      for(i=best_candidate_so_far;i<num_unlabeled-1;i++) {
	unlabeled2dnum[i] = unlabeled2dnum[i+1];
      }
      num_unlabeled--;

      /* ratio or margin active learning */
    } else if((active_type == 0 || active_type == 2) && query < hybrid_threshold) { 
      

      /* learn a classifier on the current labeled data and write it to a file */

      if(kernel_parm.kernel_type == LINEAR) { /* don't need the cache */
	svm_simple_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
			 NULL,&model,&candidate_model_length[0]);
      }
      else {
	kernel_cache_init(&kernel_cache,totdoc,kernel_cache_size);	    
	svm_simple_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
			 &kernel_cache,&model,&candidate_model_length[0]);
	kernel_cache_cleanup(&kernel_cache);
      }      

      /* write this modelfile */
      snprintf(file_suffix,16,".%d",query*num_simultaneous_queries+num_labeled);
      strcat(temp_modelfile, file_suffix);
      printf("Writing to file: %s\n", temp_modelfile);
      write_model(temp_modelfile,&model,&kernel_parm);
      strcpy(temp_modelfile, modelfile);

      /* write query file */
      strcat(temp_queryfile, file_suffix);
      write_queries(temp_queryfile,queries,query,num_simultaneous_queries);
      strcpy(temp_queryfile, queryfile);
      

      /* ---------------------------------------- */

      kernel_cache_init(&kernel_cache,totdoc,kernel_cache_size);
      best_margin_ratio_so_far=INFINITY;
      max_distance_so_far=0;      

      for (candidate_doc=0;candidate_doc<num_unlabeled;candidate_doc++) {
	
	for(candidate_label=-1;candidate_label<2;candidate_label=candidate_label+2) {
	  
	  label[unlabeled2dnum[candidate_doc]]= candidate_label;	
	  
	  if(verbosity>=2) {
	    printf("labeling doc %ld: %ld\n",candidate_doc,candidate_label);
	  }
	  
	  if(kernel_parm.kernel_type == LINEAR) { /* don't need the cache */
	    svm_simple_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
			     NULL,&model,&candidate_model_length[(candidate_label+1)/2]);
	  }
	  else {
	    
	    /* Should really flush the cache completely before restarting; but it doesn't seem to 
	       make a signifcant difference in correctness (the best doc still seems to win) */
	    /*
	      kernel_cache_quick_init(&kernel_cache,totdoc,kernel_cache_size);
	    */
	    
	    svm_simple_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
			     &kernel_cache,&model,&candidate_model_length[(candidate_label+1)/2]);
	    
	    /* This is a simple approximation to flushing the entire cache: just flush the altered doc
	       kernel_cache_free(&kernel_cache,unlabeled2dnum[candidate_doc]);
	       kernel_cache.index[kernel_cache.invindex[unlabeled2dnum[candidate_doc]]]=-1;
	       kernel_cache.invindex[unlabeled2dnum[candidate_doc]]=-1;
	    */
	  }
	  
	  free(model.supvec);
	  free(model.alpha);
	  free(model.index);
	  
	  /* restore original settings */
	  learn_parm.biased_hyperplane = learn_parm_org.biased_hyperplane;
	  learn_parm.remove_inconsistent = learn_parm_org.remove_inconsistent;
	  learn_parm.skip_final_opt_check = learn_parm_org.skip_final_opt_check;
	  learn_parm.svm_maxqpsize = learn_parm_org.svm_maxqpsize;
	  learn_parm.svm_iter_to_shrink = learn_parm_org.svm_iter_to_shrink;
	  learn_parm.svm_c = learn_parm_org.svm_c;
	  learn_parm.transduction_posratio = learn_parm_org.transduction_posratio;
	  learn_parm.svm_costratio = learn_parm_org.svm_costratio;
	  learn_parm.svm_costratio_unlab = learn_parm_org.svm_costratio_unlab;
	  learn_parm.svm_unlabbound = learn_parm_org.svm_unlabbound;
	  learn_parm.epsilon_crit = learn_parm_org.epsilon_crit;
	  learn_parm.epsilon_a = learn_parm_org.epsilon_a;
	  kernel_parm.kernel_type = kernel_parm_org.kernel_type;
	  kernel_parm.poly_degree = kernel_parm_org.poly_degree;
	  kernel_parm.rbf_gamma = kernel_parm_org.rbf_gamma;
	  kernel_parm.coef_lin = kernel_parm_org.coef_lin;
	  kernel_parm.coef_const = kernel_parm_org.coef_const;
	  
	}
	
	/* reset this label */
	label[unlabeled2dnum[candidate_doc]]= 0;	
	
	if(active_type==0) {
	  /* compare ratio of model lengths (largest/smallest) to see if 
	     it is closer to 1 than our  current best ratio */
	  if (candidate_model_length[0]>candidate_model_length[1]) {
	    current_ratio = candidate_model_length[0]/candidate_model_length[1];
	  } else {
	    current_ratio = candidate_model_length[1]/candidate_model_length[0];
	  }
	  if (current_ratio < best_margin_ratio_so_far) {
	    best_margin_ratio_so_far = current_ratio;
	    best_candidate_so_far = candidate_doc;
	  }
	  if(verbosity>=1) {
	    if(verbosity==1) {
	      printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
	    }
	    printf("Current doc:%ld\tBest doc so far:%ld", 
		   candidate_doc, 
		   best_candidate_so_far);
	    fflush(stdout);
	    if(verbosity>=2) {
	      printf("Model lengths for candidate %ld: %g and %g\n", 
		     candidate_doc, candidate_model_length[0], 
		     candidate_model_length[1] );
	      printf("Current Ratio: %g\nBest ratio so far: %g\n", 
		     current_ratio, 
		     best_margin_ratio_so_far);
	    }
	  }
	} else {
	  /* compare min of margins to see if 
	     it is larger than our current min margin */
	  if (candidate_model_length[0]>candidate_model_length[1]) {
	    current_distance = 1.0/candidate_model_length[0];
	  } else {
	    current_distance = 1.0/candidate_model_length[1];
	  }
	  if (current_distance > max_distance_so_far) {
	    max_distance_so_far = current_distance;
	    best_candidate_so_far = candidate_doc;
	  }
	  if(verbosity>=1) {
	    if(verbosity==1) {
	      printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
	    }
	    printf("Current doc:%ld\tBest doc so far:%ld", 
		   candidate_doc, 
		   best_candidate_so_far);
	    fflush(stdout);
	    if(verbosity>=2) {
	      printf("Model lengths for candidate %ld: %g and %g\n", 
		     candidate_doc, 
		     candidate_model_length[0], 
		     candidate_model_length[1] );
	      printf("Current min margin: %g\nBest margin so far: %g\n", 
		     current_distance, 
		     max_distance_so_far);
	    }
	  }
	}
      }
      
      if(verbosity>=1) {
	printf("\nQuerying document number: %ld and retraining\n", 
	       unlabeled2dnum[best_candidate_so_far]);
	fflush(stdout);
      }

      /* record the index of the queried document */
      queries[query] = unlabeled2dnum[best_candidate_so_far];

      /* set the queried document's label to the correct label */
      label[unlabeled2dnum[best_candidate_so_far]] = 
	every_doc_label[unlabeled2dnum[best_candidate_so_far]];
      
      /* remove best_candidate_so_far from the unlabeled list */
      for(i=best_candidate_so_far;i<num_unlabeled-1;i++) {
	unlabeled2dnum[i] = unlabeled2dnum[i+1];
      }
      num_unlabeled--;
      kernel_cache_cleanup(&kernel_cache);

    } else { /* --------- simple active learning ------------ */

      runtime_startclock = get_runtime();

      if(kernel_parm.kernel_type == LINEAR) { /* don't need the cache */
	svm_simple_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
			 NULL,&model,&candidate_model_length[0]);
      }
      else {
	kernel_cache_init(&kernel_cache,totdoc,kernel_cache_size);	    
	svm_simple_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
			 &kernel_cache,&model,&candidate_model_length[0]);
	kernel_cache_cleanup(&kernel_cache);
      }      

      runtime_queries = runtime_queries + get_runtime() - runtime_startclock;

      /* write this modelfile */
      snprintf(file_suffix,16,".%d",query*num_simultaneous_queries+num_labeled);
      strcat(temp_modelfile, file_suffix);
      strcat(temp_queryfile, file_suffix);
      write_model(temp_modelfile,&model,&kernel_parm);
      strcpy(temp_modelfile, modelfile);
      strcpy(temp_queryfile, queryfile);

      /* write query file */
      strcat(temp_queryfile, file_suffix);
      write_queries(temp_queryfile,queries,query,num_simultaneous_queries);
      strcpy(temp_queryfile, queryfile);

      runtime_startclock = get_runtime();

      if(num_simultaneous_queries == 1) {
	best_candidate_so_far=0;
	min_distance_so_far=INFINITY;
	/* compare distances each unlabeled example is from hyperplane */
	for (candidate_doc=0;candidate_doc<num_unlabeled;candidate_doc++) {
	  current_distance = 
	    fabs(classify_example(&model,&docs[unlabeled2dnum[candidate_doc]],&kernel_parm));
	  
	  if(verbosity>=2) {
	    printf("Distance for candidate doc %ld: %g\n", candidate_doc, current_distance );
	    printf("Best distance so far: doc %ld: %g\n", best_candidate_so_far, min_distance_so_far);
	  }
	  if(current_distance<=min_distance_so_far) {
	    min_distance_so_far = current_distance;
	    best_candidate_so_far=candidate_doc;
	  }
	}
	simultaneous_queries_indices[0]=best_candidate_so_far;
      } else {

	/* ------ figure out the best num_simultaneous_queries to ask ---- */

	/* compute the indices of the best num_candidate_queries documents */
	for(i=0; i< num_candidate_queries; i++) {
	  candidate_queries_pairs[i].label = 0;
	  candidate_queries_pairs[i].value = INFINITY;
	}
	max_distance = INFINITY;
	max_distance_index = 0;

	for(candidate_doc=0; candidate_doc<num_unlabeled;candidate_doc++) {
	  current_distance = 
	    fabs(classify_example(&model,&docs[unlabeled2dnum[candidate_doc]],&kernel_parm));

	  if(current_distance <= candidate_queries_pairs[num_candidate_queries-1].value) {
	    candidate_queries_pairs[num_candidate_queries-1].label = candidate_doc;
	    candidate_queries_pairs[num_candidate_queries-1].value = current_distance;
	    qsort (candidate_queries_pairs, num_candidate_queries, sizeof(PAIR), pairComp);
	  }
	}

	if(verbosity>=3) {
	  for(i=0; i< num_candidate_queries; i++) {
	    printf("Candidate query index %d: %ld\n",i,unlabeled2dnum[candidate_queries_pairs[i].label]);
	  }
	}

	/* Okay. candidate_queries_indices now contains our
	   candidate set of possible queries.
	   Now we need to figure out which num_simultaneous_queries 
	   of these we're going to ask */

	/* candidate_queries_indices stored in acending order of score */
	for(i=0; i<num_candidate_queries; i++) {
	  candidate_queries_indices[i] = candidate_queries_pairs[i].label;
	}

	/* start off with the first num_simultaneous_queries in 
	   the candidate_queries array as being our query set. */ 
	for(i=0; i<num_simultaneous_queries; i++) {
	  simultaneous_queries_indices[i] = candidate_queries_indices[i];
	}

	/* Compute the total pairwise distances of this query set */
	spair =0;
	for(i=0; i<num_simultaneous_queries-1; i++) {
	  for(j=i+1; j<num_simultaneous_queries; j++) {
	    spair = spair + kernel(&kernel_parm,
				   &docs[unlabeled2dnum[simultaneous_queries_indices[i]]],
				   &docs[unlabeled2dnum[simultaneous_queries_indices[i]]])
	                  + kernel(&kernel_parm,
				   &docs[unlabeled2dnum[simultaneous_queries_indices[j]]],
				   &docs[unlabeled2dnum[simultaneous_queries_indices[j]]])
    	                  - 2*kernel(&kernel_parm,
				     &docs[unlabeled2dnum[simultaneous_queries_indices[i]]],
				     &docs[unlabeled2dnum[simultaneous_queries_indices[j]]]);
	  }
	}

	  printf("Sum of pairwise distances: %f\n", spair);

	if(verbosity>=3) {
	  printf("Queries: ");
	  for(i=0; i<num_simultaneous_queries; i++) {
	    printf("%ld ", unlabeled2dnum[simultaneous_queries_indices[i]]);
	  }
	  printf("\n");
	  printf("Sum of pairwise distances: %f\n", spair);
	}

	/* Compute the row sums */
	for(i=0; i<num_simultaneous_queries; i++) {
	  row_sums[i] = 0.0;
	  for(j=0; j<num_simultaneous_queries; j++) {
	    if(i==j) continue;
	    row_sums[i] = row_sums[i] + kernel(&kernel_parm,
					       &docs[unlabeled2dnum[simultaneous_queries_indices[i]]],
					       &docs[unlabeled2dnum[simultaneous_queries_indices[i]]])
	                              + kernel(&kernel_parm,
					       &docs[unlabeled2dnum[simultaneous_queries_indices[j]]],
					       &docs[unlabeled2dnum[simultaneous_queries_indices[j]]])
                	              - 2*kernel(&kernel_parm,
						 &docs[unlabeled2dnum[simultaneous_queries_indices[i]]],
						 &docs[unlabeled2dnum[simultaneous_queries_indices[j]]]);
	  }
	}

	for(k=0; k<num_simultaneous_queries_iterations; k++) {
	  
	  /* see if we can find a better candidate query in our candidate set */
	  for(i=0; i<num_candidate_queries; i++) {
	    
	    /* Don't bother with candidate queries already in the query set */
	    skip_this = 0;
	    for(j=0; j<num_simultaneous_queries; j++) {
	      if(simultaneous_queries_indices[j] == candidate_queries_indices[i]) {
		skip_this = 1;
	      }
	    }
	    if (skip_this==1) continue;

	    if(verbosity>=3) {
	      printf("Attempt %d at using candidate %ld\n", k, 
		     unlabeled2dnum[candidate_queries_indices[i]]);
	    }
	    	    
	    /* Create the distances_to_candidate_doc array */
	    candidate_row_sum = 0.0;
	    for(j=0; j<num_simultaneous_queries; j++) {
	      distances_to_candidate_doc[j] =  kernel(&kernel_parm,
						      &docs[unlabeled2dnum[candidate_queries_indices[i]]],
						      &docs[unlabeled2dnum[candidate_queries_indices[i]]])
	                                     + kernel(&kernel_parm,
						      &docs[unlabeled2dnum[simultaneous_queries_indices[j]]],
						      &docs[unlabeled2dnum[simultaneous_queries_indices[j]]])
	                                     - 2* kernel(&kernel_parm,
							 &docs[unlabeled2dnum[candidate_queries_indices[i]]],
							 &docs[unlabeled2dnum[simultaneous_queries_indices[j]]]);
	      candidate_row_sum = candidate_row_sum + distances_to_candidate_doc[j];
	    }
	    
	    /* Consider replacing each query in the query set with our candidate doc */
	    
	    best_candidate_so_far = -1;
	    max_distance = spair;
	    for(j=0; j< num_simultaneous_queries; j++) {
	      current_distance = spair - row_sums[j]
	                         + candidate_row_sum
	                         - distances_to_candidate_doc[j];
	      if (current_distance >= max_distance) {
		max_distance = current_distance;
		best_candidate_so_far = j;
	      }
	    }

	    /* If the candidate doc improves the pairwise distance then use it. */
	    if(best_candidate_so_far != -1) {
	      
	      /* update row sums */
	      for(j=0; j<num_simultaneous_queries; j++) {
		if(j == best_candidate_so_far) {
		  row_sums[j] = candidate_row_sum - distances_to_candidate_doc[j];
		} else {
		  row_sums[j] = row_sums[j] - kernel(&kernel_parm,
						      &docs[unlabeled2dnum[simultaneous_queries_indices[j]]],
						      &docs[unlabeled2dnum[simultaneous_queries_indices[j]]])
	                                    - kernel(&kernel_parm,
					 	     &docs[unlabeled2dnum[simultaneous_queries_indices[best_candidate_so_far]]],
					 	     &docs[unlabeled2dnum[simultaneous_queries_indices[best_candidate_so_far]]])
	                                    + 2* kernel(&kernel_parm,
							&docs[unlabeled2dnum[simultaneous_queries_indices[best_candidate_so_far]]],
							&docs[unlabeled2dnum[simultaneous_queries_indices[j]]])
		                            + distances_to_candidate_doc[j];
		}
	      }
	      
	      simultaneous_queries_indices[best_candidate_so_far] = candidate_queries_indices[i];
	    }

	    if(verbosity>=3) {
	      printf("Queries: ");
	      for(j=0; j<num_simultaneous_queries; j++) {
		printf("%ld ", unlabeled2dnum[simultaneous_queries_indices[j]]);
	      }
	      printf("\n");
	      printf("Sum of pairwise distances: %f\n", spair);
	    } 
	  }
	}
      }


      /* restore original settings */	  
      free(model.supvec);
      free(model.alpha);
      free(model.index);
      
      learn_parm.biased_hyperplane = learn_parm_org.biased_hyperplane;
      learn_parm.remove_inconsistent = learn_parm_org.remove_inconsistent;
      learn_parm.skip_final_opt_check = learn_parm_org.skip_final_opt_check;
      learn_parm.svm_maxqpsize = learn_parm_org.svm_maxqpsize;
      learn_parm.svm_iter_to_shrink = learn_parm_org.svm_iter_to_shrink;
      learn_parm.svm_c = learn_parm_org.svm_c;
      learn_parm.transduction_posratio = learn_parm_org.transduction_posratio;
      learn_parm.svm_costratio = learn_parm_org.svm_costratio;
      learn_parm.svm_costratio_unlab = learn_parm_org.svm_costratio_unlab;
      learn_parm.svm_unlabbound = learn_parm_org.svm_unlabbound;
      learn_parm.epsilon_crit = learn_parm_org.epsilon_crit;
      learn_parm.epsilon_a = learn_parm_org.epsilon_a;
      kernel_parm.kernel_type = kernel_parm_org.kernel_type;
      kernel_parm.poly_degree = kernel_parm_org.poly_degree;
      kernel_parm.rbf_gamma = kernel_parm_org.rbf_gamma;
      kernel_parm.coef_lin = kernel_parm_org.coef_lin;
      kernel_parm.coef_const = kernel_parm_org.coef_const;

      if(verbosity>=1) {
	for(i=0;i<num_simultaneous_queries;i++) {
	  printf("Querying document number: %ld\n", unlabeled2dnum[simultaneous_queries_indices[i]]);
	  fflush(stdout);
	}
	printf("Retraining\n");
	fflush(stdout);
      }

      /* Need to sort them in decreasing order since otherwise 
	 we will update unlabeled2dnum array incorrectly. */
      qsort(simultaneous_queries_indices, 
	    num_simultaneous_queries, 
	    sizeof(long), inverseLongComp);

      runtime_queries = runtime_queries + get_runtime() - runtime_startclock;

      for(i=0;i<num_simultaneous_queries;i++) {
	queries[query*num_simultaneous_queries+i] = unlabeled2dnum[simultaneous_queries_indices[i]];

	/* set the queried document's label to the correct label */
	label[unlabeled2dnum[simultaneous_queries_indices[i]]] = 
	  every_doc_label[unlabeled2dnum[simultaneous_queries_indices[i]]];
	
	/* remove best_candidate_so_far from the unlabeled list */
	for(j=simultaneous_queries_indices[i];j<num_unlabeled-1;j++) {
	  unlabeled2dnum[j] = unlabeled2dnum[j+1];
	}

	num_unlabeled--;

      }
    }
  }

  runtime_startclock = get_runtime();

  /* Now finally find the max margin hyperplane */

  temp=trans;
  if(trans == 2) {
    trans = 0;
  }
  if(kernel_parm.kernel_type == LINEAR) { /* don't need the cache */
    svm_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
	      NULL,&model, trans);
  }
  else {
    kernel_cache_init(&kernel_cache,totdoc,kernel_cache_size);
    svm_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
	      &kernel_cache,&model, trans);
    kernel_cache_cleanup(&kernel_cache);
  }

  runtime_finalmdl = runtime_finalmdl + get_runtime() - runtime_startclock;

  /* write this modelfile and query file */
  if(num_query_batches>0) {
    snprintf(file_suffix,16,".%d",num_query_batches*num_simultaneous_queries+num_labeled);
    strcat(temp_modelfile, file_suffix);
    write_model(temp_modelfile,&model,&kernel_parm);
    strcpy(temp_modelfile, modelfile);

    /* write query file */
    strcat(temp_queryfile, file_suffix);
    write_queries(temp_queryfile,queries,num_query_batches,num_simultaneous_queries);
    strcpy(temp_queryfile, queryfile);
  } 
  write_model(modelfile,&model,&kernel_parm);

  /* if -z 2 is set then now do transduction */
  if(temp==2) {
    trans = 1;
    /* restore original settings */
    learn_parm.biased_hyperplane = learn_parm_org.biased_hyperplane;
    learn_parm.remove_inconsistent = learn_parm_org.remove_inconsistent;
    learn_parm.skip_final_opt_check = learn_parm_org.skip_final_opt_check;
    learn_parm.svm_maxqpsize = learn_parm_org.svm_maxqpsize;
    learn_parm.svm_iter_to_shrink = learn_parm_org.svm_iter_to_shrink;
    learn_parm.svm_c = learn_parm_org.svm_c;
    learn_parm.transduction_posratio = learn_parm_org.transduction_posratio;
    learn_parm.svm_costratio = learn_parm_org.svm_costratio;
    learn_parm.svm_costratio_unlab = learn_parm_org.svm_costratio_unlab;
    learn_parm.svm_unlabbound = learn_parm_org.svm_unlabbound;
    learn_parm.epsilon_crit = learn_parm_org.epsilon_crit;
    learn_parm.epsilon_a = learn_parm_org.epsilon_a;
    kernel_parm.kernel_type = kernel_parm_org.kernel_type;
    kernel_parm.poly_degree = kernel_parm_org.poly_degree;
    kernel_parm.rbf_gamma = kernel_parm_org.rbf_gamma;
    kernel_parm.coef_lin = kernel_parm_org.coef_lin;
    kernel_parm.coef_const = kernel_parm_org.coef_const;
    free(model.supvec);
    free(model.alpha);
    free(model.index);
    
    if(kernel_parm.kernel_type == LINEAR) { /* don't need the cache */
      svm_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
		NULL,&model, trans);
    }
    else {
      kernel_cache_init(&kernel_cache,totdoc,kernel_cache_size);
      svm_learn(docs,label,totdoc,totwords,&learn_parm,&kernel_parm,
		&kernel_cache,&model, trans);
    kernel_cache_cleanup(&kernel_cache);
    }
    write_model(trans_modelfile,&model,&kernel_parm);
  }
  

  if(num_query_batches>0) {
    free(unlabeled2dnum);
    free(queries);
    free(row_sums);
    free(distances_to_candidate_doc);
    free(candidate_queries_indices);
    free(candidate_queries_pairs);
    free(simultaneous_queries_indices);
  }
  free(docs);
  free(label);
  free(model.supvec);
  free(model.alpha);

  printf("Total runtime for generating queries: %f\n", (float)(runtime_queries/100.0));
  printf("Total runtime for generating final_model: %f\n", (float)(runtime_finalmdl/100.0));

}

/*---------------------------------------------------------------------------*/

void svm_learn(docs,label,totdoc,totwords,learn_parm,kernel_parm,kernel_cache,
	       model, trans)
DOC *docs;
long *label,totdoc,totwords;
LEARN_PARM *learn_parm;
KERNEL_PARM *kernel_parm;
KERNEL_CACHE *kernel_cache;
MODEL *model;
int trans;
{
  long *inconsistent,*chosen,*key,i,ii,j,jj,*last_suboptimal_at;
  long inconsistentnum,choosenum,retrain,iteration,inactivenum;
  long misclassified,supvecnum=0,upsupvecnum,*active2dnum;
  long at_upper_bound,*active,*working2dnum,*selexam;
  long activenum,*inactive_since,deactnum;
  double loss,model_length,criterion,eq;
  double maxdiff,*a_old,*lin,*a;
  double **a_history;
  long t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0; /* timing */
  long time_kernel,time_opti,time_shrink,time_update;
  long runtime_start,runtime_end;
  long time_model,time_check,time_select;
  long transductcycle;
  long *unlabeled,transduction;
  double epsilon_crit_org; 
  double *selcrit;  /* buffer for sorting */        
  CFLOAT *aicache;  /* buffer to keep one row of hessian */
  double *weights;  /* buffer for weight vector in linear case */
  QP qp;            /* buffer for one quadratic program */

  runtime_start=get_runtime();
  time_kernel=0;
  time_opti=0;
  time_shrink=0;
  time_update=0;
  time_model=0;
  time_check=0;
  time_select=0;
  kernel_cache_statistic=0;

  epsilon_crit_org=learn_parm->epsilon_crit; /* save org */
  if(kernel_parm->kernel_type == LINEAR) {
    learn_parm->epsilon_crit=2.0;
  } 
  learn_parm->epsilon_shrink=2;
  maxdiff=1;

  inconsistent = (long *)my_malloc(sizeof(long)*totdoc);
  chosen = (long *)my_malloc(sizeof(long)*totdoc);
  unlabeled = (long *)my_malloc(sizeof(long)*totdoc);
  active = (long *)my_malloc(sizeof(long)*totdoc);
  inactive_since = (long *)my_malloc(sizeof(long)*totdoc);
  last_suboptimal_at = (long *)my_malloc(sizeof(long)*totdoc);
  key = (long *)my_malloc(sizeof(long)*(totdoc+11)); 
  selcrit = (double *)my_malloc(sizeof(double)*totdoc);
  selexam = (long *)my_malloc(sizeof(long)*totdoc);
  a = (double *)my_malloc(sizeof(double)*totdoc);
  a_old = (double *)my_malloc(sizeof(double)*totdoc);
  a_history = (double **)my_malloc(sizeof(double *)*10000);
  lin = (double *)my_malloc(sizeof(double)*totdoc);
  learn_parm->svm_cost = (double *)my_malloc(sizeof(double)*totdoc);
  aicache = (CFLOAT *)my_malloc(sizeof(CFLOAT)*totdoc);
  working2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
  active2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*(totdoc+2));
  model->alpha = (double *)my_malloc(sizeof(double)*(totdoc+2));
  model->index = (long *)my_malloc(sizeof(long)*(totdoc+2));
  qp.opt_ce = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_ce0 = (double *)my_malloc(sizeof(double));
  qp.opt_g = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize
				 *learn_parm->svm_maxqpsize);
  qp.opt_g0 = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_xinit = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_low=(double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_up=(double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  weights=(double *)my_malloc(sizeof(double)*(totwords+1));

  model->at_upper_bound=0;
  model->b=0;	       
  model->supvec[0]=0;  /* element 0 reserved and empty for now */
  model->alpha[0]=0;
  model->sv_num=1;
  activenum=totdoc;
  choosenum=0;
  inactivenum=0;
  deactnum=0;
  inconsistentnum=0;
  transductcycle=0;
  transduction=0;
  retrain=1;

  for(i=0;i<totdoc;i++) {    /* various inits */
    inconsistent[i]=0;
    chosen[i]=0;
    active[i]=1;
    a[i]=0;
    a_old[i]=0;
    lin[i]=0;
    last_suboptimal_at[i]=1;
    unlabeled[i]=0;
    if(label[i] == 0) {
      unlabeled[i]=1;
      transduction=1;
    }
    if(label[i] > 0) {
      learn_parm->svm_cost[i]=learn_parm->svm_c*learn_parm->svm_costratio*
	fabs((double)label[i]);
      label[i]=1;
    }
    else if(label[i] < 0) {
      learn_parm->svm_cost[i]=learn_parm->svm_c*fabs((double)label[i]);
      label[i]=-1;
    }
  }
  activenum=compute_index(active,totdoc,active2dnum);
  clear_index(working2dnum);

  if(verbosity==1) {
    printf("Optimizing"); fflush(stdout);
  }

                            /* repeat this loop until we have convergence */
  for(iteration=1;retrain;iteration++) {

    if(kernel_parm->kernel_type != LINEAR)
      kernel_cache->time=iteration;  /* for lru cache */
    if(verbosity>=2) {
      printf(
	"Iteration %ld: ",iteration); fflush(stdout);
    }
    else if(verbosity==1) {
      printf("."); fflush(stdout);
    }

    if(verbosity>=2) t0=get_runtime();
    if(verbosity>=3) {
      printf("\nSelecting working set... "); fflush(stdout); 
    }
    choosenum=0;
    for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* clear working set */
      chosen[j]=0; 
    }
    clear_index(working2dnum);
    if(retrain == 2) {
      for(i=0;i<totdoc;i++) { /* set inconsistent examples to zero (-i 1) */
	if(inconsistent[i] && (a[i] != 0.0)) {
	  chosen[i]=1;
	  choosenum++;
	  a[i]=0;
	}
      }
      if(learn_parm->biased_hyperplane) {
	eq=0;
	for(i=0;i<totdoc;i++) { /* make sure we fulfill equality constraint */
	  eq+=a[i]*label[i];
	}
	for(i=0;(i<totdoc) && (fabs(eq) > learn_parm->epsilon_a);i++) {
	  if(eq*label[i] > 0) {
	    chosen[i]=1;
	    choosenum++;
	    if((eq*label[i]) > a[i]) {
	      eq-=(a[i]*label[i]);
	      a[i]=0;
	    }
	    else {
	      a[i]-=(eq*label[i]);
	      eq=0;
	    }
	  }
	}
      }
      compute_index(chosen,totdoc,working2dnum);
    }
    else {      /* select working set according to steepest gradient */ 
      if((learn_parm->svm_maxqpsize>=4) 
	 && (kernel_parm->kernel_type != LINEAR)) {
	/* select part of the working set from cache */
	choosenum+=select_next_qp_subproblem_grad_cache(
                               label,unlabeled,a,lin,totdoc,
			       (long)((learn_parm->svm_maxqpsize-choosenum)/2),
			       learn_parm,inconsistent,active2dnum,
			       working2dnum,selcrit,selexam,kernel_cache,
			       key,chosen);
      }
      choosenum+=select_next_qp_subproblem_grad(label,unlabeled,a,lin,totdoc,
                               (long)(learn_parm->svm_maxqpsize-choosenum),
                               learn_parm,inconsistent,active2dnum,
			       working2dnum,selcrit,selexam,kernel_cache,key,
			       chosen);
    }

    if(verbosity>=2) {
      printf(" %ld vectors chosen\n",choosenum); fflush(stdout); 
    }

    if(verbosity>=2) t1=get_runtime();

    if(kernel_parm->kernel_type != LINEAR) 
      cache_multiple_kernel_rows(kernel_cache,docs,working2dnum,
				 choosenum,kernel_parm); 
    
    if(verbosity>=2) t2=get_runtime();
    if(retrain != 2) {
      optimize_svm(docs,label,unlabeled,chosen,active2dnum,model,totdoc,
		   working2dnum,choosenum,a,lin,learn_parm,aicache,
		   kernel_parm,&qp,&epsilon_crit_org);
    }

    if(verbosity>=2) t3=get_runtime();
    update_linear_component(docs,label,active2dnum,a,a_old,working2dnum,totdoc,
			    totwords,kernel_parm,kernel_cache,lin,aicache,
			    weights);

    if(verbosity>=2) t4=get_runtime();
    supvecnum=calculate_svm_model(docs,label,unlabeled,lin,a,a_old,learn_parm,
				  working2dnum,model);

    if(verbosity>=2) t5=get_runtime();

    /* The following computation of the objective function works only */
    /* relative to the active variables */
    if(verbosity>=3) {
      criterion=compute_objective_function(a,lin,label,active2dnum);
      printf("Objective function (over active variables): %.16f\n",criterion);
      fflush(stdout); 
    }

    for(jj=0;(i=working2dnum[jj])>=0;jj++) {
      a_old[i]=a[i];
    }

    if(retrain == 2) {  /* reset inconsistent unlabeled examples */
      for(i=0;(i<totdoc);i++) {
	if(inconsistent[i] && unlabeled[i]) {
	  inconsistent[i]=0;
	  label[i]=0;
	}
      }
    }

    retrain=check_optimality(model,label,unlabeled,a,lin,totdoc,learn_parm,
			     &maxdiff,epsilon_crit_org,&misclassified,
			     inconsistent,active2dnum,last_suboptimal_at,
			     iteration,kernel_parm);
    if(verbosity>=2) {
      t6=get_runtime();
      time_select+=t1-t0;
      time_kernel+=t2-t1;
      time_opti+=t3-t2;
      time_update+=t4-t3;
      time_model+=t5-t4;
      time_check+=t6-t5;
    }

    if((!retrain) && (inactivenum>0) 
       && ((!learn_parm->skip_final_opt_check) 
	   || (kernel_parm->kernel_type == LINEAR))) { 
      if(((verbosity>=1) && (kernel_parm->kernel_type != LINEAR)) 
	 || (verbosity>=2)) {
	if(verbosity==1) {
	  printf("\n");
	}
	printf(" Checking optimality of inactive variables..."); 
	fflush(stdout);
      }
      t1=get_runtime();
      reactivate_inactive_examples(label,unlabeled,a,a_history,lin,totdoc,
				   totwords,iteration,learn_parm,inconsistent,
				   active,inactive_since,deactnum,docs,
				   kernel_parm,kernel_cache,model,aicache,
				   weights,&maxdiff);
      /* Update to new active variables. */
      activenum=compute_index(active,totdoc,active2dnum);
      inactivenum=totdoc-activenum;
      /* termination criterion */
      retrain=0;
      if(maxdiff > learn_parm->epsilon_crit) 
	retrain=1;
      time_shrink+=get_runtime()-t1;
      if(((verbosity>=1) && (kernel_parm->kernel_type != LINEAR)) 
	 || (verbosity>=2)) {
	printf("done.\n");  fflush(stdout);
        printf(" Number of inactive variables = %ld\n",inactivenum);
      }		  
    }

    if((!retrain) && (learn_parm->epsilon_crit>maxdiff)) 
      learn_parm->epsilon_crit=maxdiff;
    if((!retrain) && (learn_parm->epsilon_crit>epsilon_crit_org)) {
      learn_parm->epsilon_crit/=2.0;
      retrain=1;
    }
    if(learn_parm->epsilon_crit<epsilon_crit_org) 
      learn_parm->epsilon_crit=epsilon_crit_org;
    
    if(verbosity>=2) {
      printf(" => (%ld SV (incl. %ld SV at u-bound), max violation=%.5f)\n",
	     supvecnum,model->at_upper_bound,maxdiff); 
      fflush(stdout);
    }
    if(verbosity>=3) {
      printf("\n");
    }

    if((!retrain) & learn_parm->remove_inconsistent) {
      if(verbosity>=1) {
	printf(" Moving BSVs to inconsistent examples...");
	fflush(stdout);
      }
      retrain=identify_inconsistent(a,label,unlabeled,working2dnum,learn_parm,
				    &inconsistentnum,inconsistent);
      if(verbosity>=1) {
	printf("done.\n");
	if(retrain) {
	  printf(" Now %ld inconsistent examples.\n",inconsistentnum);
	}
      }
    }
    
    if((!retrain) && (transduction) && trans==1) {
      for(i=0;(i<totdoc);i++) {
	active[i]=1;
      }
      activenum=compute_index(active,totdoc,active2dnum);
      inactivenum=0;
      if(verbosity==1) printf("done\n");
      retrain=incorporate_unlabeled_examples(model,label,inconsistent,
					     unlabeled,a,lin,totdoc,
					     selcrit,selexam,key,
					     transductcycle,kernel_parm,
					     learn_parm);
      epsilon_crit_org=learn_parm->epsilon_crit;
      if(kernel_parm->kernel_type == LINEAR)
	learn_parm->epsilon_crit=1; 
      transductcycle++;
    } 
    else if((iteration % 10) == 0) {
      activenum=shrink_problem(learn_parm,last_suboptimal_at,active,
			       inactive_since,active2dnum,&deactnum,
			       iteration,totdoc,max((long)(activenum/10),100),
			       a,a_history,inconsistent);
      inactivenum=totdoc-activenum;
      if((kernel_parm->kernel_type != LINEAR)
	 && (supvecnum>kernel_cache->max_elems)
	 && ((kernel_cache->activenum-activenum)>max((long)(activenum/10),500))) {
	kernel_cache_shrink(kernel_cache,totdoc,max((long)(activenum/10),500),
			    active); 
      }
    }
  } /* end of loop */

  for(i=0;(i<totdoc);i++) { /* make all active for final statistic */
    active[i]=1;
  }
  activenum=compute_index(active,totdoc,active2dnum);
  inactivenum=0;
  retrain=check_optimality(model,label,unlabeled,a,lin,totdoc,learn_parm,
			   &maxdiff,epsilon_crit_org,&misclassified,
			   inconsistent,active2dnum,last_suboptimal_at,
			   iteration,kernel_parm);

  if(verbosity>=1) {
    if(verbosity==1) printf("done. (%ld iterations)\n",iteration);
    printf("Optimization finished (%ld misclassified, maxdiff=%.5f).\n",
	   misclassified,maxdiff); 
    runtime_end=get_runtime();
    if(verbosity>=2) {
      printf("Runtime in cpu-seconds: %.2f (%.2f%% for kernel/%.2f%% for optimizer/%.2f%% for final/%.2f%% for update/%.2f%% for model/%.2f%% for check/%.2f%% for select)\n",
	     ((float)runtime_end-(float)runtime_start)/100.0,
	     (100.0*time_kernel)/(float)(runtime_end-runtime_start),
	     (100.0*time_opti)/(float)(runtime_end-runtime_start),
	     (100.0*time_shrink)/(float)(runtime_end-runtime_start),
	     (100.0*time_update)/(float)(runtime_end-runtime_start),
	     (100.0*time_model)/(float)(runtime_end-runtime_start),
	     (100.0*time_check)/(float)(runtime_end-runtime_start),
	     (100.0*time_select)/(float)(runtime_end-runtime_start));
    }
    else {
      printf("Runtime in cpu-seconds: %.2f\n",
	     (runtime_end-runtime_start)/100.0);
    }
    if(learn_parm->remove_inconsistent) {
      printf("Number of SV: %ld (plus %ld inconsistent examples)\n",
	     supvecnum,inconsistentnum);
    }
    else {
      upsupvecnum=0;
      for(i=1;i<model->sv_num;i++) {
	if(fabs(model->alpha[i]) >= (learn_parm->svm_cost[(model->supvec[i])->docnum]-learn_parm->epsilon_a)) upsupvecnum++;
      }
      printf("Number of SV: %ld (including %ld at upper bound)\n",
	     supvecnum,upsupvecnum);
    }

    if((verbosity>=1) && (!learn_parm->skip_final_opt_check)) {
      loss=0;
      model_length=0; 
      for(i=0;i<totdoc;i++) {
	if((lin[i]-model->b)*(double)label[i] < 1.0-epsilon_crit_org)
	  loss+=1.0-(lin[i]-model->b)*(double)label[i];
	model_length+=a[i]*label[i]*lin[i];
      }
      model_length=sqrt(model_length);
      fprintf(stdout,"L1 loss: loss=%.5f\n",loss);
      fprintf(stdout,"Norm of weight vector: |w|=%.5f\n",model_length);
      /* fprintf(stdout,"Estimated VCdim of classifier: VCdim<=%.5f\n",
	      estimate_margin_vcdim(model,model_length,kernel_parm)); */
      /* fprintf(stdout,"Fraction of alphas>=1.0: %l.1f%%\n",
	      estimate_loo_bound(model,label,unlabeled,totdoc)*100.0); */
    }
    if(verbosity>=1) {
      printf("Number of kernel evaluations: %ld\n",kernel_cache_statistic);
    }
  }

  if(learn_parm->alphafile[0])
    write_alphas(learn_parm->alphafile,a,label,totdoc);

  free(inconsistent);
  free(chosen);
  free(unlabeled);
  free(active);
  free(inactive_since);
  free(last_suboptimal_at);
  free(key);
  free(selcrit);
  free(selexam);
  free(a);
  free(a_old);
  free(a_history);
  free(lin);
  free(learn_parm->svm_cost);
  free(aicache);
  free(working2dnum);
  free(active2dnum);
  free(qp.opt_ce);
  free(qp.opt_ce0);
  free(qp.opt_g);
  free(qp.opt_g0);
  free(qp.opt_xinit);
  free(qp.opt_low);
  free(qp.opt_up);
  free(weights);
}

/*---------------------------------------------------------------------------*/

void svm_simple_learn(docs,label,totdoc,totwords,learn_parm,kernel_parm,kernel_cache,
		      model,model_length_ptr)
DOC *docs;
long *label,totdoc,totwords;
LEARN_PARM *learn_parm;
KERNEL_PARM *kernel_parm;
KERNEL_CACHE *kernel_cache;
MODEL *model;
double *model_length_ptr;
{
  long *inconsistent,*chosen,*key,i,ii,j,jj,*last_suboptimal_at;
  long inconsistentnum,choosenum,retrain,iteration,inactivenum;
  long misclassified,supvecnum=0,upsupvecnum,*active2dnum;
  long at_upper_bound,*active,*working2dnum,*selexam;
  long activenum,*inactive_since,deactnum;
  double loss,model_length,criterion,eq;
  double maxdiff,*a_old,*lin,*a;
  double **a_history;
  long t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0; /* timing */
  long time_kernel,time_opti,time_shrink,time_update;
  long runtime_start,runtime_end;
  long time_model,time_check,time_select;
  long transductcycle;
  long *unlabeled,transduction;
  double epsilon_crit_org; 
  double *selcrit;  /* buffer for sorting */        
  CFLOAT *aicache;  /* buffer to keep one row of hessian */
  double *weights;  /* buffer for weight vector in linear case */
  QP qp;            /* buffer for one quadratic program */

  runtime_start=get_runtime();
  time_kernel=0;
  time_opti=0;
  time_shrink=0;
  time_update=0;
  time_model=0;
  time_check=0;
  time_select=0;
  kernel_cache_statistic=0;

  epsilon_crit_org=learn_parm->epsilon_crit; /* save org */
  if(kernel_parm->kernel_type == LINEAR) {
    learn_parm->epsilon_crit=2.0;
  } 
  learn_parm->epsilon_shrink=2;
  maxdiff=1;

  inconsistent = (long *)my_malloc(sizeof(long)*totdoc);
  chosen = (long *)my_malloc(sizeof(long)*totdoc);
  unlabeled = (long *)my_malloc(sizeof(long)*totdoc);
  active = (long *)my_malloc(sizeof(long)*totdoc);
  inactive_since = (long *)my_malloc(sizeof(long)*totdoc);
  last_suboptimal_at = (long *)my_malloc(sizeof(long)*totdoc);
  key = (long *)my_malloc(sizeof(long)*(totdoc+11)); 
  selcrit = (double *)my_malloc(sizeof(double)*totdoc);
  selexam = (long *)my_malloc(sizeof(long)*totdoc);
  a = (double *)my_malloc(sizeof(double)*totdoc);
  a_old = (double *)my_malloc(sizeof(double)*totdoc);
  a_history = (double **)my_malloc(sizeof(double *)*10000);
  lin = (double *)my_malloc(sizeof(double)*totdoc);
  learn_parm->svm_cost = (double *)my_malloc(sizeof(double)*totdoc);
  aicache = (CFLOAT *)my_malloc(sizeof(CFLOAT)*totdoc);
  working2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
  active2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*(totdoc+2));
  model->alpha = (double *)my_malloc(sizeof(double)*(totdoc+2));
  model->index = (long *)my_malloc(sizeof(long)*(totdoc+2));
  qp.opt_ce = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_ce0 = (double *)my_malloc(sizeof(double));
  qp.opt_g = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize
				 *learn_parm->svm_maxqpsize);
  qp.opt_g0 = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_xinit = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_low=(double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_up=(double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  weights=(double *)my_malloc(sizeof(double)*(totwords+1));

  model->at_upper_bound=0;
  model->b=0;	       
  model->supvec[0]=0;  /* element 0 reserved and empty for now */
  model->alpha[0]=0;
  model->sv_num=1;
  activenum=totdoc;
  choosenum=0;
  inactivenum=0;
  deactnum=0;
  inconsistentnum=0;
  transductcycle=0;
  transduction=0;
  retrain=1;

  for(i=0;i<totdoc;i++) {    /* various inits */
    inconsistent[i]=0;
    chosen[i]=0;
    active[i]=1;
    a[i]=0;
    a_old[i]=0;
    lin[i]=0;
    last_suboptimal_at[i]=1;
    unlabeled[i]=0;
    if(label[i] == 0) {
      unlabeled[i]=1;
      transduction=1;
    }
    if(label[i] > 0) {
      learn_parm->svm_cost[i]=learn_parm->svm_c*learn_parm->svm_costratio*
	fabs((double)label[i]);
      label[i]=1;
    }
    else if(label[i] < 0) {
      learn_parm->svm_cost[i]=learn_parm->svm_c*fabs((double)label[i]);
      label[i]=-1;
    }
  }
  activenum=compute_index(active,totdoc,active2dnum);
  clear_index(working2dnum);

  /*
  if(verbosity==1) {
    printf("Optimizing"); fflush(stdout);
  }
  */
                            /* repeat this loop until we have convergence */
  for(iteration=1;retrain;iteration++) {

    if(kernel_parm->kernel_type != LINEAR)
      kernel_cache->time=iteration;  /* for lru cache */
    if(verbosity>=2) {
      printf(
	"Iteration %ld: ",iteration); fflush(stdout);
    }

    /*
    else if(verbosity==1) {
      printf("."); fflush(stdout);
    }
    */

    if(verbosity>=2) t0=get_runtime();
    if(verbosity>=3) {
      printf("\nSelecting working set... "); fflush(stdout); 
    }
    choosenum=0;
    for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* clear working set */
      chosen[j]=0; 
    }
    clear_index(working2dnum);
    if(retrain == 2) {
      for(i=0;i<totdoc;i++) { /* set inconsistent examples to zero (-i 1) */
	if(inconsistent[i] && (a[i] != 0.0)) {
	  chosen[i]=1;
	  choosenum++;
	  a[i]=0;
	}
      }
      if(learn_parm->biased_hyperplane) {
	eq=0;
	for(i=0;i<totdoc;i++) { /* make sure we fulfill equality constraint */
	  eq+=a[i]*label[i];
	}
	for(i=0;(i<totdoc) && (fabs(eq) > learn_parm->epsilon_a);i++) {
	  if(eq*label[i] > 0) {
	    chosen[i]=1;
	    choosenum++;
	    if((eq*label[i]) > a[i]) {
	      eq-=(a[i]*label[i]);
	      a[i]=0;
	    }
	    else {
	      a[i]-=(eq*label[i]);
	      eq=0;
	    }
	  }
	}
      }
      compute_index(chosen,totdoc,working2dnum);
    }
    else {      /* select working set according to steepest gradient */ 
      if((learn_parm->svm_maxqpsize>=4) 
	 && (kernel_parm->kernel_type != LINEAR)) {
	/* select part of the working set from cache */
	choosenum+=select_next_qp_subproblem_grad_cache(
                               label,unlabeled,a,lin,totdoc,
			       (long)((learn_parm->svm_maxqpsize-choosenum)/2),
			       learn_parm,inconsistent,active2dnum,
			       working2dnum,selcrit,selexam,kernel_cache,
			       key,chosen);
      }
      choosenum+=select_next_qp_subproblem_grad(label,unlabeled,a,lin,totdoc,
                               (long)(learn_parm->svm_maxqpsize-choosenum),
                               learn_parm,inconsistent,active2dnum,
			       working2dnum,selcrit,selexam,kernel_cache,key,
			       chosen);
    }

    if(verbosity>=2) {
      printf(" %ld vectors chosen\n",choosenum); fflush(stdout); 
    }

    if(verbosity>=2) t1=get_runtime();

    if(kernel_parm->kernel_type != LINEAR) 
      cache_multiple_kernel_rows(kernel_cache,docs,working2dnum,
				 choosenum,kernel_parm); 
    
    if(verbosity>=2) t2=get_runtime();
    if(retrain != 2) {
      optimize_svm(docs,label,unlabeled,chosen,active2dnum,model,totdoc,
		   working2dnum,choosenum,a,lin,learn_parm,aicache,
		   kernel_parm,&qp,&epsilon_crit_org);
    }

    if(verbosity>=2) t3=get_runtime();
    update_linear_component(docs,label,active2dnum,a,a_old,working2dnum,totdoc,
			    totwords,kernel_parm,kernel_cache,lin,aicache,
			    weights);

    if(verbosity>=2) t4=get_runtime();
    supvecnum=calculate_svm_model(docs,label,unlabeled,lin,a,a_old,learn_parm,
				  working2dnum,model);

    if(verbosity>=2) t5=get_runtime();

    /* The following computation of the objective function works only */
    /* relative to the active variables */
    if(verbosity>=3) {
      criterion=compute_objective_function(a,lin,label,active2dnum);
      printf("Objective function (over active variables): %.16f\n",criterion);
      fflush(stdout); 
    }

    for(jj=0;(i=working2dnum[jj])>=0;jj++) {
      a_old[i]=a[i];
    }

    if(retrain == 2) {  /* reset inconsistent unlabeled examples */
      for(i=0;(i<totdoc);i++) {
	if(inconsistent[i] && unlabeled[i]) {
	  inconsistent[i]=0;
	  label[i]=0;
	}
      }
    }

    retrain=check_optimality(model,label,unlabeled,a,lin,totdoc,learn_parm,
			     &maxdiff,epsilon_crit_org,&misclassified,
			     inconsistent,active2dnum,last_suboptimal_at,
			     iteration,kernel_parm);
    if(verbosity>=2) {
      t6=get_runtime();
      time_select+=t1-t0;
      time_kernel+=t2-t1;
      time_opti+=t3-t2;
      time_update+=t4-t3;
      time_model+=t5-t4;
      time_check+=t6-t5;
    }

    if((!retrain) && (inactivenum>0) 
       && ((!learn_parm->skip_final_opt_check) 
	   || (kernel_parm->kernel_type == LINEAR))) { 
      if(((verbosity>=1) && (kernel_parm->kernel_type != LINEAR)) 
	 || (verbosity>=2)) {
	if(verbosity==1) {
	  printf("\n");
	}
	printf(" Checking optimality of inactive variables..."); 
	fflush(stdout);
      }
      t1=get_runtime();
      reactivate_inactive_examples(label,unlabeled,a,a_history,lin,totdoc,
				   totwords,iteration,learn_parm,inconsistent,
				   active,inactive_since,deactnum,docs,
				   kernel_parm,kernel_cache,model,aicache,
				   weights,&maxdiff);
      /* Update to new active variables. */
      activenum=compute_index(active,totdoc,active2dnum);
      inactivenum=totdoc-activenum;
      /* termination criterion */
      retrain=0;
      if(maxdiff > learn_parm->epsilon_crit) 
	retrain=1;
      time_shrink+=get_runtime()-t1;
      if(((verbosity>=1) && (kernel_parm->kernel_type != LINEAR)) 
	 || (verbosity>=2)) {
	printf("done.\n");  fflush(stdout);
        printf(" Number of inactive variables = %ld\n",inactivenum);
      }		  
    }

    if((!retrain) && (learn_parm->epsilon_crit>maxdiff)) 
      learn_parm->epsilon_crit=maxdiff;
    if((!retrain) && (learn_parm->epsilon_crit>epsilon_crit_org)) {
      learn_parm->epsilon_crit/=2.0;
      retrain=1;
    }
    if(learn_parm->epsilon_crit<epsilon_crit_org) 
      learn_parm->epsilon_crit=epsilon_crit_org;
    
    if(verbosity>=2) {
      printf(" => (%ld SV (incl. %ld SV at u-bound), max violation=%.5f)\n",
	     supvecnum,model->at_upper_bound,maxdiff); 
      fflush(stdout);
    }
    if(verbosity>=3) {
      printf("\n");
    }

    if((!retrain) & learn_parm->remove_inconsistent) {
      if(verbosity>=1) {
	printf(" Moving BSVs to inconsistent examples...");
	fflush(stdout);
      }
      retrain=identify_inconsistent(a,label,unlabeled,working2dnum,learn_parm,
				    &inconsistentnum,inconsistent);
      if(verbosity>=1) {
	printf("done.\n");
	if(retrain) {
	  printf(" Now %ld inconsistent examples.\n",inconsistentnum);
	}
      }
    }

    if((iteration % 10) == 0) {
      activenum=shrink_problem(learn_parm,last_suboptimal_at,active,
			       inactive_since,active2dnum,&deactnum,
			       iteration,totdoc,max((long)(activenum/10),100),
			       a,a_history,inconsistent);
      inactivenum=totdoc-activenum;
      if((kernel_parm->kernel_type != LINEAR)
	 && (supvecnum>kernel_cache->max_elems)
	 && ((kernel_cache->activenum-activenum)>max((long)(activenum/10),500))) {
	kernel_cache_shrink(kernel_cache,totdoc,max((long)(activenum/10),500),
			    active); 
      }
    }
  } /* end of loop */

  for(i=0;(i<totdoc);i++) { /* make all active for final statistic */
    active[i]=1;
  }
  activenum=compute_index(active,totdoc,active2dnum);
  inactivenum=0;
  retrain=check_optimality(model,label,unlabeled,a,lin,totdoc,learn_parm,
			   &maxdiff,epsilon_crit_org,&misclassified,
			   inconsistent,active2dnum,last_suboptimal_at,
			   iteration,kernel_parm);

  /*
    if(verbosity==1) printf("done. (%ld iterations\n",iteration);
  */

  if(verbosity>=2) {
    runtime_end=get_runtime();
    printf("Optimization finished (%ld misclassified, maxdiff=%.5f).\n",
	   misclassified,maxdiff);     
    printf("Runtime in cpu-seconds: %.2f (%.2f%% for kernel/%.2f%% for optimizer/%.2f%% for final/%.2f%% for update/%.2f%% for model/%.2f%% for check/%.2f%% for select)\n",
	   ((float)runtime_end-(float)runtime_start)/100.0,
	   (100.0*time_kernel)/(float)(runtime_end-runtime_start),
	   (100.0*time_opti)/(float)(runtime_end-runtime_start),
	   (100.0*time_shrink)/(float)(runtime_end-runtime_start),
	   (100.0*time_update)/(float)(runtime_end-runtime_start),
	   (100.0*time_model)/(float)(runtime_end-runtime_start),
	   (100.0*time_check)/(float)(runtime_end-runtime_start),
	   (100.0*time_select)/(float)(runtime_end-runtime_start));

    if(learn_parm->remove_inconsistent) {
      printf("Number of SV: %ld (plus %ld inconsistent examples)\n",
	     supvecnum,inconsistentnum);
    }
    else {
      upsupvecnum=0;
      for(i=1;i<model->sv_num;i++) {
	if(fabs(model->alpha[i]) >= (learn_parm->svm_cost[(model->supvec[i])->docnum]-learn_parm->epsilon_a)) upsupvecnum++;
      }
      printf("Number of SV: %ld (including %ld at upper bound)\n",
	     supvecnum,upsupvecnum);
    }
  }

  loss=0;
  model_length=0; 
  for(i=0;i<totdoc;i++) {
    if((lin[i]-model->b)*(double)label[i] < 1.0-epsilon_crit_org)
      loss+=1.0-(lin[i]-model->b)*(double)label[i];
    model_length+=a[i]*label[i]*lin[i];
  }
  
  model_length=sqrt(model_length);
  *model_length_ptr=model_length;
  
  if((verbosity>=2) && (!learn_parm->skip_final_opt_check)) {
    
    fprintf(stdout,"L1 loss: loss=%.5f\n",loss);
    fprintf(stdout,"Norm of weight vector: |w|=%.5f\n",model_length);
    printf("Number of kernel evaluations: %ld\n",kernel_cache_statistic);
  }

  free(inconsistent);
  free(chosen);
  free(unlabeled);
  free(active);
  free(inactive_since);
  free(last_suboptimal_at);
  free(key);
  free(selcrit);
  free(selexam);
  free(a);
  free(a_old);
  free(a_history);
  free(lin);
  free(learn_parm->svm_cost);
  free(aicache);
  free(working2dnum);
  free(active2dnum);
  free(qp.opt_ce);
  free(qp.opt_ce0);
  free(qp.opt_g);
  free(qp.opt_g0);
  free(qp.opt_xinit);
  free(qp.opt_low);
  free(qp.opt_up);
  free(weights);
}

/*---------------------------------------------------------------------------*/

void active_learn(docs,label,totdoc,totwords,learn_parm,kernel_parm,kernel_cache,
		  model,num_query_batches, every_doc_label)
DOC *docs;
long *label,*every_doc_label,num_query_batches,totdoc,totwords;
LEARN_PARM *learn_parm;
KERNEL_PARM *kernel_parm;
KERNEL_CACHE *kernel_cache;
MODEL *model;
{
  long query, candidate_doc, candidate_label, *unlabeled2dnum, num_unlabeled;
  long best_candidate_so_far, tot_iteration, i;
  double best_margin_ratio_so_far=INFINITY;
  double current_ratio, candidate_model_length[2];
  LEARN_PARM learn_parm_org;
  KERNEL_PARM kernel_parm_org;

  unlabeled2dnum = (long *)my_malloc(sizeof(long)*totdoc);

  /* index unlabeled docs */
  num_unlabeled=0;
  for(i=0;i<totdoc;i++) {
    if(label[i]==0) {
      unlabeled2dnum[num_unlabeled]=i;
      num_unlabeled++;
    }
  }

  /* store original settings */

  learn_parm_org.biased_hyperplane = learn_parm->biased_hyperplane;
  learn_parm_org.remove_inconsistent = learn_parm->remove_inconsistent;
  learn_parm_org.skip_final_opt_check = learn_parm->skip_final_opt_check;
  learn_parm_org.svm_maxqpsize = learn_parm->svm_maxqpsize;
  learn_parm_org.svm_iter_to_shrink = learn_parm->svm_iter_to_shrink;
  learn_parm_org.svm_c = learn_parm->svm_c;
  learn_parm_org.transduction_posratio = learn_parm->transduction_posratio;
  learn_parm_org.svm_costratio = learn_parm->svm_costratio;
  learn_parm_org.svm_costratio_unlab = learn_parm->svm_costratio_unlab;
  learn_parm_org.svm_unlabbound = learn_parm->svm_unlabbound;
  learn_parm_org.epsilon_crit = learn_parm->epsilon_crit;
  learn_parm_org.epsilon_a = learn_parm->epsilon_a;
  kernel_parm_org.kernel_type = kernel_parm->kernel_type;
  kernel_parm_org.poly_degree = kernel_parm->poly_degree;
  kernel_parm_org.rbf_gamma = kernel_parm->rbf_gamma;
  kernel_parm_org.coef_lin = kernel_parm->coef_lin;
  kernel_parm_org.coef_const = kernel_parm->coef_const;

  for (query=0;query<num_query_batches;query++) {
    for (candidate_doc=0;candidate_doc<num_unlabeled;candidate_doc++) {

      for(candidate_label=-1;candidate_label<2;candidate_label=candidate_label+2) {

	label[unlabeled2dnum[candidate_doc]]= candidate_label;	
	svm_simple_learn(docs,label,totdoc,totwords,learn_parm,kernel_parm,kernel_cache,
			 model,&candidate_model_length[(candidate_label+1)/2]);       

	
	/* restore original settings */
	learn_parm->biased_hyperplane = learn_parm_org.biased_hyperplane;
	learn_parm->remove_inconsistent = learn_parm_org.remove_inconsistent;
	learn_parm->skip_final_opt_check = learn_parm_org.skip_final_opt_check;
	learn_parm->svm_maxqpsize = learn_parm_org.svm_maxqpsize;
	learn_parm->svm_iter_to_shrink = learn_parm_org.svm_iter_to_shrink;
	learn_parm->svm_c = learn_parm_org.svm_c;
	learn_parm->transduction_posratio = learn_parm_org.transduction_posratio;
	learn_parm->svm_costratio = learn_parm_org.svm_costratio;
	learn_parm->svm_costratio_unlab = learn_parm_org.svm_costratio_unlab;
	learn_parm->svm_unlabbound = learn_parm_org.svm_unlabbound;
	learn_parm->epsilon_crit = learn_parm_org.epsilon_crit;
	learn_parm->epsilon_a = learn_parm_org.epsilon_a;
	kernel_parm->kernel_type = kernel_parm_org.kernel_type;
	kernel_parm->poly_degree = kernel_parm_org.poly_degree;
	kernel_parm->rbf_gamma = kernel_parm_org.rbf_gamma;
	kernel_parm->coef_lin = kernel_parm_org.coef_lin;
	kernel_parm->coef_const = kernel_parm_org.coef_const;
      }
   
      /* reset this label */
      label[unlabeled2dnum[candidate_doc]]= 0;	

      /* compare ratio of model lengths (largest/smallest) to see if 
	 it is closer to 1 than our  current best ratio */
      if (candidate_model_length[0]>candidate_model_length[1]) {
	current_ratio = candidate_model_length[0]/candidate_model_length[1];
      } else {
	current_ratio = candidate_model_length[1]/candidate_model_length[0];
      }
      printf("Model lengths for candidate %ld: %g and %g\n", candidate_doc, candidate_model_length[0], candidate_model_length[1] );
      printf("Current Ratio: %g\nBest ratio so far: %g\n", current_ratio, best_margin_ratio_so_far);
      if (current_ratio < best_margin_ratio_so_far) {
	best_margin_ratio_so_far = current_ratio;
	best_candidate_so_far = candidate_doc;
      }
    }

    if(verbosity>=1) {
      printf("Querying document number: %ld and retraining", unlabeled2dnum[best_candidate_so_far]);
      fflush(stdout);
    }

    /* set the queried document's label to the correct label */
    label[unlabeled2dnum[best_candidate_so_far]] = every_doc_label[unlabeled2dnum[best_candidate_so_far]];

    /* remove best_candidate_so_far from the unlabeled list */
    for(i=best_candidate_so_far;i<num_unlabeled-1;i++) {
      unlabeled2dnum[i] = unlabeled2dnum[i+1];
    }
    num_unlabeled--;
  }

  free(unlabeled2dnum);
}

/* -------------------------------------------------------------------- */



double compute_objective_function(a,lin,label,active2dnum)
double *a,*lin;             /* Return value of objective function. */
long *label,*active2dnum;   /* Works only relative to the active variables! */
{
  long i,ii;
  double criterion;
  /* calculate value of objective function */
  criterion=0;
  for(ii=0;active2dnum[ii]>=0;ii++) {
    i=active2dnum[ii];
    criterion=criterion-a[i]+0.5*a[i]*label[i]*lin[i];
  }
  return(criterion);
}

void clear_index(index)
long *index;  /* initializes and empties index */
{
  index[0]=-1;
} 

void add_to_index(index,elem)
long *index,elem;  /* initializes and empties index */
{
  register long i;
  for(i=0;index[i] != -1;i++);
  index[i]=elem;
  index[i+1]=-1;
}

long compute_index(binfeature,range,index)
long *binfeature,range,*index;  /* create an inverted index of binfeature */
{               
  register long i,ii;

  ii=0;
  for(i=0;i<range;i++) {
    if(binfeature[i]) {
      index[ii]=i;
      ii++;
    }
  }
  for(i=0;i<4;i++) {
    index[ii+i]=-1;
  }
  return(ii);
}

void optimize_svm(docs,label,unlabeled,chosen,active2dnum,model,totdoc,
		  working2dnum,varnum,a,lin,learn_parm,aicache,kernel_parm,
		  qp,epsilon_crit_target)
DOC *docs;               /* Do optimization on the working set. */
long *label,*unlabeled,*chosen,*active2dnum;
MODEL *model;
long totdoc,*working2dnum,varnum;
double *a,*lin;
LEARN_PARM *learn_parm;
CFLOAT *aicache;
KERNEL_PARM *kernel_parm;
QP *qp;
double *epsilon_crit_target;
{
    long i;
    double *a_v;

    compute_matrices_for_optimization(docs,label,unlabeled,chosen,active2dnum,
				      working2dnum,model,a,lin,varnum,
				      totdoc,learn_parm,aicache,kernel_parm,
				      qp);

    if(verbosity>=3) {
      printf("Running optimizer..."); fflush(stdout);
    }
    /* call the qp-subsolver */
    a_v=optimize_qp(qp,epsilon_crit_target,
		    learn_parm->svm_maxqpsize,
		    &(model->b),   /* in case the optimizer gives us */
                                   /* the threshold for free. otherwise */
                                   /* b is calculated in calculate_model. */
		    learn_parm);
    if(verbosity>=3) {         
      printf("done\n");
    }

    for(i=0;i<varnum;i++) {
      a[working2dnum[i]]=a_v[i];
      /*
      if(a_v[i]<=(0+learn_parm->epsilon_a)) {
	a[working2dnum[i]]=0;
      }
      else if(a_v[i]>=(learn_parm->svm_cost[working2dnum[i]]-learn_parm->epsilon_a)) {
	a[working2dnum[i]]=learn_parm->svm_cost[working2dnum[i]];
      }
      */
    }
}

void compute_matrices_for_optimization(docs,label,unlabeled,chosen,active2dnum,
				       key,model,a,lin,varnum,totdoc,
				       learn_parm,aicache,kernel_parm,qp)
DOC *docs;
long *label,*unlabeled,*chosen,*active2dnum,*key;
MODEL *model;
double *a,*lin;
long varnum,totdoc;
LEARN_PARM *learn_parm;
CFLOAT *aicache;
KERNEL_PARM *kernel_parm;
QP *qp;
{
  register long ki,kj,i,j,dnum;
  register double yn,kernel_temp;

  if(verbosity>=3) {
    fprintf(stdout,"Computing qp-matrices (type %ld kernel [degree %ld, rbf_gamma %f, coef_lin %f, coef_const %f])...",kernel_parm->kernel_type,kernel_parm->poly_degree,kernel_parm->rbf_gamma,kernel_parm->coef_lin,kernel_parm->coef_const); 
    fflush(stdout);
  }

  qp->opt_n=varnum;
  qp->opt_ce0[0]=0; /* compute the constant for quality constraint */
  for(j=1;j<model->sv_num;j++) { /* start at 1 */
    if(!chosen[(model->supvec[j])->docnum]) {
      qp->opt_ce0[0]+=model->alpha[j];
    }
  } 
  if(learn_parm->biased_hyperplane) 
    qp->opt_m=1;
  else 
    qp->opt_m=0;  /* eq-constraint will be ignored */

  /* init linear part of objective function */
  for(i=0;i<varnum;i++) {
    qp->opt_g0[i]=lin[key[i]];
  }

  for(i=0;i<varnum;i++) {
    ki=key[i];

    /* Compute the matrix for equality constraints */
    qp->opt_ce[i]=label[ki];
    qp->opt_low[i]=0;
    qp->opt_up[i]=learn_parm->svm_cost[ki];

    kernel_temp=(double)kernel(kernel_parm,&(docs[ki]),&(docs[ki])); 
    /* compute linear part of objective function */
    qp->opt_g0[i]-=(kernel_temp*a[ki]*(double)label[ki]); 
    /* compute quadratic part of objective function */
    qp->opt_g[varnum*i+i]=kernel_temp;
    for(j=i+1;j<varnum;j++) {
      kj=key[j];
      kernel_temp=(double)kernel(kernel_parm,&(docs[ki]),&(docs[kj]));
      /* compute linear part of objective function */
      qp->opt_g0[i]-=(kernel_temp*a[kj]*(double)label[kj]);
      qp->opt_g0[j]-=(kernel_temp*a[ki]*(double)label[ki]); 
      /* compute quadratic part of objective function */
      qp->opt_g[varnum*i+j]=(double)label[ki]*(double)label[kj]*kernel_temp;
      qp->opt_g[varnum*j+i]=(double)label[ki]*(double)label[kj]*kernel_temp;
    }

    if(verbosity>=3) {
      if(i % 20 == 0) {
	fprintf(stdout,"%ld..",i); fflush(stdout);
      }
    }
  }

  for(i=0;i<varnum;i++) {
    /* assure starting at feasible point */
    qp->opt_xinit[i]=a[key[i]];
    /* set linear part of objective function */
    qp->opt_g0[i]=-1.0+qp->opt_g0[i]*(double)label[key[i]];    
  }

  if(verbosity>=3) {
    fprintf(stdout,"done\n");
  }
}

long calculate_svm_model(docs,label,unlabeled,lin,a,a_old,learn_parm,
			 working2dnum,model)
DOC *docs;              /* Compute decision function based on current values */
long *label,*unlabeled; /* of alpha. */
double *lin,*a,*a_old;
LEARN_PARM *learn_parm;
long *working2dnum;
MODEL *model;
{
  long i,ii,pos,b_calculated=0;
  double ex_c;

  if(verbosity>=3) {
    printf("Calculating model..."); fflush(stdout);
  }

  if(!learn_parm->biased_hyperplane) {
    model->b=0;
    b_calculated=1;
  }

  for(ii=0;(i=working2dnum[ii])>=0;ii++) {
    if((a_old[i]>0) && (a[i]==0)) { /* remove from model */
      pos=model->index[i]; 
      model->index[i]=-1;
      (model->sv_num)--;
      model->supvec[pos]=model->supvec[model->sv_num];
      model->alpha[pos]=model->alpha[model->sv_num];
      model->index[(model->supvec[pos])->docnum]=pos;
    }
    else if((a_old[i]==0) && (a[i]>0)) { /* add to model */
      model->supvec[model->sv_num]=&(docs[i]);
      model->alpha[model->sv_num]=a[i]*(double)label[i];
      model->index[i]=model->sv_num;
      (model->sv_num)++;
    }
    else if(a_old[i]==a[i]) { /* nothing to do */
    }
    else {  /* just update alpha */
      model->alpha[model->index[i]]=a[i]*(double)label[i];
    }
      
    ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
    if((a_old[i]>=ex_c) && (a[i]<ex_c)) { 
      (model->at_upper_bound)--;
    }
    else if((a_old[i]<ex_c) && (a[i]>=ex_c)) { 
      (model->at_upper_bound)++;
    }

    if((!b_calculated) 
       && (a[i]>learn_parm->epsilon_a) && (a[i]<ex_c)) {   /* calculate b */
	model->b=(-(double)label[i]+lin[i]);
	b_calculated=1;
    }
  }      

  if(verbosity>=3) {
    printf("done\n"); fflush(stdout);
  }

  /* If there is no alpha in the working set not at bounds, then
     just use the model->b from the last iteration */

  return(model->sv_num-1); /* have to substract one, since element 0 is empty*/
}

long check_optimality(model,label,unlabeled,a,lin,totdoc,learn_parm,maxdiff,
		      epsilon_crit_org,misclassified,inconsistent,
		      active2dnum,last_suboptimal_at,iteration,kernel_parm)
MODEL *model;            /* Check KT-conditions */
long *label,*unlabeled;
double *a,*lin;
long totdoc;
LEARN_PARM *learn_parm;
double *maxdiff,epsilon_crit_org;
long *misclassified;
long *inconsistent,*active2dnum,*last_suboptimal_at,iteration;
KERNEL_PARM *kernel_parm;
{
  long i,ii,retrain;
  double dist,diff,ex_c;

  if(kernel_parm->kernel_type == LINEAR) {  /* be optimistic */
    learn_parm->epsilon_shrink=-learn_parm->epsilon_crit+epsilon_crit_org;  
  }
  else {  /* be conservative */
    learn_parm->epsilon_shrink=learn_parm->epsilon_shrink*0.7+(*maxdiff)*0.3; 
  }
  retrain=0;
  (*maxdiff)=0;
  (*misclassified)=0;
  for(ii=0;(i=active2dnum[ii])>=0;ii++) {
    if((!inconsistent[i]) && label[i]) {
      dist=(lin[i]-model->b)*(double)label[i];/* 'distance' from hyperplane*/
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if(dist <= 0) {       
	(*misclassified)++;  /* does not work due to deactivation of var */
      }
      if((a[i]>learn_parm->epsilon_a) && (dist > 1)) {
	if((dist-1.0)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-1.0;
      }
      else if((a[i]<ex_c) && (dist < 1)) {
	if((1.0-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=1.0-dist;
      }
      /* Count how long a variable was at lower/upper bound (and optimal).*/
      /* Variables, which were at the bound and optimal for a long */
      /* time are unlikely to become support vectors. In case our */
      /* cache is filled up, those variables are excluded to save */
      /* kernel evaluations. (See chapter 'Shrinking').*/ 
      if((a[i]>(learn_parm->epsilon_a)) 
	 && (a[i]<ex_c)) { 
	last_suboptimal_at[i]=iteration;         /* not at bound */
      }
      else if((a[i]<=(learn_parm->epsilon_a)) 
	      && (dist < (1.0+learn_parm->epsilon_shrink))) {
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
      else if((a[i]>=ex_c)
	      && (dist > (1.0-learn_parm->epsilon_shrink)))  { 
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
    }   
  }
  /* termination criterion */
  if((!retrain) && ((*maxdiff) > learn_parm->epsilon_crit)) {  
    retrain=1;
  }
  return(retrain);
}

long identify_inconsistent(a,label,unlabeled,working2dnum,learn_parm,
			   inconsistentnum,inconsistent)
double *a;
long *label,*unlabeled,*working2dnum;
LEARN_PARM *learn_parm;
long *inconsistentnum,*inconsistent;
{
  long i,ii,retrain;

  /* Throw out examples with multipliers at upper bound. This */
  /* corresponds to the -i 1 option. */
  /* ATTENTION: this is just a heuristic for finding a close */
  /*            to minimum number of examples to exclude to */
  /*            make the problem separable with desired margin */
  retrain=0;
  for(ii=0;(i=working2dnum[ii])>=0;ii++) {
    if((!inconsistent[i]) && (!unlabeled[i]) 
       && (a[i]>=(learn_parm->svm_cost[i]-learn_parm->epsilon_a))) { 
	(*inconsistentnum)++;
	inconsistent[i]=1;  /* never choose again */
	retrain=2;          /* start over */
	if(verbosity>=3) {
	  printf("inconsistent(%ld)..",i); fflush(stdout);
	}
    }
  }
  return(retrain);
}

void update_linear_component(docs,label,active2dnum,a,a_old,working2dnum,
			     totdoc,totwords,kernel_parm,kernel_cache,lin,
			     aicache,weights)
DOC *docs;
long *label; 
long *active2dnum;                  /* keep track of the linear component */
double *a,*a_old;                   /* lin of the gradient etc. by updating */
long *working2dnum,totdoc,totwords; /* based on the change of the variables */
KERNEL_PARM *kernel_parm;           /* in the current working set */
KERNEL_CACHE *kernel_cache;
double *lin;
CFLOAT *aicache;
double *weights;
{
  register long i,ii,j,jj;
  register double tec;

  if(kernel_parm->kernel_type==0) { /* special linear case */
    clear_vector_n(weights,totwords);
    for(ii=0;(i=working2dnum[ii])>=0;ii++) {
      if(a[i] != a_old[i]) {
	add_vector_ns(weights,docs[i].words,((a[i]-a_old[i])*(double)label[i]));
      }
    }
    for(jj=0;(j=active2dnum[jj])>=0;jj++) {
      lin[j]+=sprod_ns(weights,docs[j].words);
    }
  }
  else {                            /* general case */
    for(jj=0;(i=working2dnum[jj])>=0;jj++) {
      if(a[i] != a_old[i]) {
	get_kernel_row(kernel_cache,docs,i,totdoc,active2dnum,aicache,
		       kernel_parm);
	for(ii=0;(j=active2dnum[ii])>=0;ii++) {
	  tec=aicache[j];
	  lin[j]+=(((a[i]*tec)-(a_old[i]*tec))*(double)label[i]);
	}
      }
    }
  }
}


long incorporate_unlabeled_examples(model,label,inconsistent,unlabeled,a,lin,totdoc,selcrit,
				    select,key,transductcycle,kernel_parm,learn_parm)
MODEL *model;
long *label;
long *inconsistent,*unlabeled;
double *a,*lin;
long totdoc;
double *selcrit;
long *select,*key,transductcycle;
KERNEL_PARM *kernel_parm;
LEARN_PARM *learn_parm;
{
  long i,j,k,j1,j2,j3,j4,unsupaddnum=20,unsupaddnum1=0,unsupaddnum2=0;
  long pos,neg,upos,uneg,orgpos,orgneg,nolabel,newpos,newneg,allunlab,didit;
  double dist,model_length,posratio,negratio,a_max,a_avg,a_avg_pos;
  long check_every=2,a_avg_count,a_avg_pos_count,outerpos,outerneg;
  long maxindex;
  double maxval,loss;
  static double switchsens=0.0,switchsensorg=0.0;
  double umin,umax,sumalpha;
  long imin,imax;
  static long switchnum=0;

  switchsens/=1.2;

  /* assumes that lin[] is up to date -> no inactive vars */

  orgpos=0;
  orgneg=0;
  newpos=0;
  newneg=0;
  nolabel=0;
  allunlab=0;
  for(i=0;i<totdoc;i++) {
    if(!unlabeled[i]) {
      if(label[i] > 0) {
	orgpos++;
      }
      else {
	orgneg++;
      }
    }
    else {
      allunlab++;
      if(unlabeled[i]) {
	if(label[i] > 0) {
	  newpos++;
	}
	else if(label[i] < 0) {
	  newneg++;
	}
      }
    }
    if(label[i]==0) {
      nolabel++;
    }
  }

  if(learn_parm->transduction_posratio >= 0) {
    posratio=learn_parm->transduction_posratio;
  }
  else {
    posratio=(double)orgpos/(double)(orgpos+orgneg); /* use ratio of pos/neg */
  }                                                  /* in training data */
  negratio=1.0-posratio;

  learn_parm->svm_costratio=1.0;                     /* global */
  if(posratio>0) {
    learn_parm->svm_costratio_unlab=negratio/posratio;
  }
  else {
    learn_parm->svm_costratio_unlab=1.0;
  }
  
  pos=0;
  neg=0;
  upos=0;
  uneg=0;
  for(i=0;i<totdoc;i++) {
    dist=(lin[i]-model->b);  /* 'distance' from hyperplane*/
    if(dist>0) {
      pos++;
    }
    else {
      neg++;
    }
    if(unlabeled[i]) {
      if(dist>0) {
	upos++;
      }
      else {
	uneg++;
      }
    }
    if((!unlabeled[i]) && (a[i]>(learn_parm->svm_cost[i]-learn_parm->epsilon_a))) {
      /*      printf("Ubounded %ld (class %ld, unlabeled %ld)\n",i,label[i],unlabeled[i]); */
    }
  }
  if(verbosity>=2) {
    printf("POS=%ld, ORGPOS=%ld, ORGNEG=%ld\n",pos,orgpos,orgneg);
    printf("POS=%ld, NEWPOS=%ld, NEWNEG=%ld\n",pos,newpos,newneg);
    printf("pos ratio = %f (%f).\n",(double)(upos)/(double)(allunlab),posratio);
    fflush(stdout);
  }

  if(transductcycle == 0) {
    j1=0; 
    j2=0;
    j4=0;
    for(i=0;i<totdoc;i++) {
      dist=(lin[i]-model->b);  /* 'distance' from hyperplane*/
      if((label[i]==0) && (unlabeled[i])) {
	selcrit[j4]=dist;
	key[j4]=i;
	j4++;
      }
    }
    unsupaddnum1=0;	
    unsupaddnum2=0;	
    select_top_n(selcrit,j4,select,(long)(allunlab*posratio+0.5));
    for(k=0;(k<(long)(allunlab*posratio+0.5));k++) {
      i=key[select[k]];
      label[i]=1;
      unsupaddnum1++;	
      j1++;
    }
    for(i=0;i<totdoc;i++) {
      if((label[i]==0) && (unlabeled[i])) {
	label[i]=-1;
	j2++;
	unsupaddnum2++;
      }
    }
    for(i=0;i<totdoc;i++) {  /* set upper bounds on vars */
      if(unlabeled[i]) {
	if(label[i] == 1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_costratio_unlab*learn_parm->svm_unlabbound;
	}
	else if(label[i] == -1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_unlabbound;
	}
      }
    }
    if(verbosity>=1) {
      /* printf("costratio %lf, costratio_unlab %lf, unlabbound %lf\n",
	 learn_parm->svm_costratio,learn_parm->svm_costratio_unlab,
	 learn_parm->svm_unlabbound); */
      printf("Classifying unlabeled data as %ld POS / %ld NEG.\n",
	     unsupaddnum1,unsupaddnum2); 
      fflush(stdout);
    }
    if(verbosity >= 1) 
      printf("Retraining.");
    if(verbosity >= 2) printf("\n");
    return((long)3);
  }
  if((transductcycle % check_every) == 0) {
    if(verbosity >= 1) 
      printf("Retraining.");
    if(verbosity >= 2) printf("\n");
    j1=0;
    j2=0;
    unsupaddnum1=0;
    unsupaddnum2=0;
    for(i=0;i<totdoc;i++) {
      if((unlabeled[i] == 2)) {
	unlabeled[i]=1;
	label[i]=1;
	j1++;
	unsupaddnum1++;
      }
      else if((unlabeled[i] == 3)) {
	unlabeled[i]=1;
	label[i]=-1;
	j2++;
	unsupaddnum2++;
      }
    }
    for(i=0;i<totdoc;i++) {  /* set upper bounds on vars */
      if(unlabeled[i]) {
	if(label[i] == 1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_costratio_unlab*learn_parm->svm_unlabbound;
	}
	else if(label[i] == -1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_unlabbound;
	}
      }
    }

    if(verbosity>=2) {
      /* printf("costratio %lf, costratio_unlab %lf, unlabbound %lf\n",
	     learn_parm->svm_costratio,learn_parm->svm_costratio_unlab,
	     learn_parm->svm_unlabbound); */
      printf("%d positive -> Added %ld POS / %ld NEG unlabeled examples.\n",
	     upos,unsupaddnum1,unsupaddnum2); 
      fflush(stdout);
    }

    if(learn_parm->svm_unlabbound == 1) {
      learn_parm->epsilon_crit=0.001; /* do the last run right */
    }
    else {
      learn_parm->epsilon_crit=0.01; /* otherwise, no need to be so picky */
    }

    return((long)3);
  }
  else if(((transductcycle % check_every) < check_every)) { 
    model_length=0;
    sumalpha=0;
    loss=0;
    for(i=0;i<totdoc;i++) {
      model_length+=a[i]*label[i]*lin[i];
      sumalpha+=a[i];
      dist=(lin[i]-model->b);  /* 'distance' from hyperplane*/
      if((label[i]*dist)<(1.0-learn_parm->epsilon_crit)) {
	loss+=(1.0-(label[i]*dist))*learn_parm->svm_cost[i]; 
      }
    }
    model_length=sqrt(model_length); 
    if(verbosity>=2) {
      printf("Model-length = %lf (%lf), loss = %lf, objective = %lf\n",
	     model_length,sumalpha,loss,loss+0.5*model_length*model_length);
      fflush(stdout);
    }
    j1=0;
    j2=0;
    j3=0;
    j4=0;
    unsupaddnum1=0;	
    unsupaddnum2=0;	
    umin=99999;
    umax=-99999;
    j4=1;
    while(j4) {
      umin=99999;
      umax=-99999;
      for(i=0;(i<totdoc);i++) { 
	dist=(lin[i]-model->b);  
	if((label[i]>0) && (unlabeled[i]) && (!inconsistent[i]) 
	   && (dist<umin)) {
	  umin=dist;
	  imin=i;
	}
	if((label[i]<0) && (unlabeled[i])  && (!inconsistent[i]) 
	   && (dist>umax)) {
	  umax=dist;
	  imax=i;
	}
      }
      if((umin < (umax+switchsens-1E-4))) {
	j1++;
	j2++;
	unsupaddnum1++;	
	unlabeled[imin]=3;
	inconsistent[imin]=1;
	unsupaddnum2++;	
	unlabeled[imax]=2;
	inconsistent[imax]=1;
      }
      else
	j4=0;
      j4=0;
    }
    for(j=0;(j<totdoc);j++) {
      if(unlabeled[j] && (!inconsistent[j])) {
	if(label[j]>0) {
	  unlabeled[j]=2;
	}
	else if(label[j]<0) {
	  unlabeled[j]=3;
	}
	/* inconsistent[j]=1; */
	j3++;
      }
    }
    switchnum+=unsupaddnum1+unsupaddnum2;
    if((!unsupaddnum1) && (!unsupaddnum2)) {
      if((learn_parm->svm_unlabbound>=1) && ((newpos+newneg) == allunlab)) {
	for(j=0;(j<totdoc);j++) {
	  inconsistent[j]=0;
	  if(unlabeled[j]) unlabeled[j]=1;
	}
	write_prediction(learn_parm->predfile,model,lin,a,unlabeled,label,
			 totdoc,learn_parm);  
	if(verbosity>=1)
	  printf("Number of switches: %ld\n",switchnum);
	return((long)0);
      }
      switchsens=switchsensorg;
      learn_parm->svm_unlabbound*=1.5;
      if(learn_parm->svm_unlabbound>1) {
	learn_parm->svm_unlabbound=1;
      }
      model->at_upper_bound=0; /* since upper bound increased */
      if(verbosity>=1) 
	printf("Increasing influence of unlabeled examples to %lf%% .",
	       learn_parm->svm_unlabbound*100.0);
    }
    else if(verbosity>=1) {
      printf("%d positive -> Switching labels of %ld POS / %ld NEG unlabeled examples.",
	     upos,unsupaddnum1,unsupaddnum2); 
      fflush(stdout);
    }

    if(verbosity >= 2) printf("\n");
    
    learn_parm->epsilon_crit=0.5; /* don't need to be so picky */

    for(i=0;i<totdoc;i++) {  /* set upper bounds on vars */
      if(unlabeled[i]) {
	if(label[i] == 1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_costratio_unlab*learn_parm->svm_unlabbound;
	}
	else if(label[i] == -1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_unlabbound;
	}
      }
    }

    return((long)2);
  }

  return((long)0); 
}

/*************************** Working set selection ***************************/

long select_next_qp_subproblem_grad(label,unlabeled,a,lin,totdoc,qp_size,
				    learn_parm,inconsistent,active2dnum,
				    working2dnum,selcrit,select,kernel_cache,
				    key,chosen)
long *label,*unlabeled;
double *a,*lin;      /* Use the feasible direction approach to select the */
long totdoc,qp_size; /* next qp-subproblem  (see section 'Selecting a good */
LEARN_PARM *learn_parm; /* working set') */
long *inconsistent,*active2dnum,*working2dnum;
double *selcrit;
long *select;
KERNEL_CACHE *kernel_cache;
long *key,*chosen;
{
  long choosenum,i,j,k,activedoc,inum;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=-label[j];
    if((!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0)))
       && (!inconsistent[j]) 
       && (label[j])
       && (!chosen[j])) {
      selcrit[activedoc]=lin[j]-(double)label[j];
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(kernel_cache,i); /* make sure it does not get kicked */
                                        /* out of cache */
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=label[j];
    if((!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0))) 
       && (!inconsistent[j]) 
       && (label[j])
       && (!chosen[j])) {
      selcrit[activedoc]=(double)(label[j])-lin[j];
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(kernel_cache,i); /* make sure it does not get kicked */
                                        /* out of cache */
  } 
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}

long select_next_qp_subproblem_grad_cache(label,unlabeled,a,lin,totdoc,qp_size,
					  learn_parm,inconsistent,active2dnum,
					  working2dnum,selcrit,select,
					  kernel_cache,key,chosen)
long *label,*unlabeled;
double *a,*lin;         /* Use the feasible direction approach to select the */
long totdoc,qp_size;    /* next qp-subproblem  (see chapter 'Selecting a  */
LEARN_PARM *learn_parm; /* good working set') among the variable with */
long *inconsistent,*active2dnum,*working2dnum; /* cached kernel */
double *selcrit;
long *select;
KERNEL_CACHE *kernel_cache;
long *key,*chosen;
{
  long choosenum,i,j,k,activedoc,inum;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=-label[j];
    if((kernel_cache->index[j]>=0)
       && (!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0)))
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j]))
      {
      selcrit[activedoc]=(double)label[j]*(-1.0+(double)label[j]*lin[j]);
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(kernel_cache,i); /* make sure it does not get kicked */
                                        /* out of cache */
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=label[j];
    if((kernel_cache->index[j]>=0)
       && (!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0))) 
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j])) 
      {
      selcrit[activedoc]=-(double)(label[j]*(-1.0+(double)label[j]*lin[j]));
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(kernel_cache,i); /* make sure it does not get kicked */
                                        /* out of cache */
  } 
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}

void select_top_n(selcrit,range,select,n)
double *selcrit;
long range,*select,n;
{
  register long i,j;

  for(i=0;(i<n) && (i<range);i++) { /* Initialize with the first n elements */
    for(j=i;j>=0;j--) {
      if((j>0) && (selcrit[select[j-1]]<selcrit[i])){
	select[j]=select[j-1];
      }
      else {
	select[j]=i;
	j=-1;
      }
    }
  }
  for(i=n;i<range;i++) {  
    if(selcrit[i]>selcrit[select[n-1]]) {
      for(j=n-1;j>=0;j--) {
	if((j>0) && (selcrit[select[j-1]]<selcrit[i])) {
	  select[j]=select[j-1];
	}
	else {
	  select[j]=i;
	  j=-1;
	}
      }
    }
  }
}      
      

/******************************** Shrinking  *********************************/

long shrink_problem(learn_parm,last_suboptimal_at,active,
		    inactive_since,active2dnum,deactnum,iteration,totdoc,
		    minshrink,a,a_history,inconsistent)
     /* shrink some variables away */
     /* do the shrinking only if at least minshrink variables can be removed */
LEARN_PARM *learn_parm;
long *last_suboptimal_at,*active,*inactive_since;
long *active2dnum,*deactnum,iteration;
long totdoc,minshrink;
double *a,**a_history;
long *inconsistent;
{
  long i,ii,change,activenum;
  double *a_old;
  
  activenum=0;
  change=0;
  for(ii=0;active2dnum[ii]>=0;ii++) {
    i=active2dnum[ii];
    activenum++;
    if(((iteration-last_suboptimal_at[i])>learn_parm->svm_iter_to_shrink) 
       || (inconsistent[i])) {
      change++;
    }
  }
  if(change>=minshrink) { /* shrink only if sufficiently many candidates */
    /* Shrink problem by removing those variables which are */
    /* optimal at a bound for a minimum number of iterations */
    if(verbosity>=2) {
      printf(" Shrinking..."); fflush(stdout);
    }
    a_old=(double *)my_malloc(sizeof(double)*totdoc);
    a_history[(*deactnum)]=a_old;
    for(i=0;i<totdoc;i++) {
      a_old[i]=a[i];
    }
    change=0;
    for(ii=0;active2dnum[ii]>=0;ii++) {
      i=active2dnum[ii];
      if((((iteration-last_suboptimal_at[i])>learn_parm->svm_iter_to_shrink) 
	  || (inconsistent[i]))) {
	active[i]=0;
	inactive_since[i]=(*deactnum);
	change++;
      }
    }
    activenum=compute_index(active,totdoc,active2dnum);
    (*deactnum)++;
    if(verbosity>=2) {
      printf("done.\n"); fflush(stdout);
      printf(" Number of inactive variables = %ld\n",totdoc-activenum);
    }
  }
  return(activenum);
} 


void reactivate_inactive_examples(label,unlabeled,a,a_history,lin,totdoc,
				  totwords,iteration,learn_parm,inconsistent,
				  active,inactive_since,deactnum,docs,
				  kernel_parm,kernel_cache,model,aicache,
				  weights,maxdiff)
long *label,*unlabeled;              /* Make all variables active again */
double *a,**a_history,*lin;          /* which had been removed by shrinking. */
long totdoc,totwords,iteration;      /* Computes lin for those */
LEARN_PARM *learn_parm;              /* variables from scratch. */
long *inconsistent,*active,*inactive_since,deactnum;
DOC *docs;
KERNEL_PARM *kernel_parm;
KERNEL_CACHE *kernel_cache;
MODEL *model;
CFLOAT *aicache;
double *weights,*maxdiff;
{
  register long i,j,ii,jj,t,*changed2dnum,*inactive2dnum;
  long *changed,*inactive;
  register double kernel_val,*a_old,dist;
  double ex_c;

  changed=(long *)my_malloc(sizeof(long)*totdoc);
  changed2dnum=(long *)my_malloc(sizeof(long)*(totdoc+11));
  inactive=(long *)my_malloc(sizeof(long)*totdoc);
  inactive2dnum=(long *)my_malloc(sizeof(long)*(totdoc+11));
  for(t=deactnum-1;(t>=0) && a_history[t];t--) {
    if(verbosity>=2) {
      printf("%ld..",t); fflush(stdout);
    }
    a_old=a_history[t];    
    for(i=0;i<totdoc;i++) {
      inactive[i]=((!active[i]) && (inactive_since[i] == t));
      changed[i]= (a[i] != a_old[i]);
    }
    compute_index(inactive,totdoc,inactive2dnum);
    compute_index(changed,totdoc,changed2dnum);

    if(kernel_parm->kernel_type == LINEAR) { /* special linear case */
      clear_vector_n(weights,totwords);
      for(ii=0;changed2dnum[ii]>=0;ii++) {
	i=changed2dnum[ii];
	add_vector_ns(weights,docs[i].words,
		      ((a[i]-a_old[i])*(double)label[i]));
      }
      for(jj=0;(j=inactive2dnum[jj])>=0;jj++) {
	lin[j]+=sprod_ns(weights,docs[j].words);
      }
    }
    else {
      for(ii=0;(i=changed2dnum[ii])>=0;ii++) {
	get_kernel_row(kernel_cache,docs,i,totdoc,inactive2dnum,aicache,
		       kernel_parm);
	for(jj=0;(j=inactive2dnum[jj])>=0;jj++) {
	  kernel_val=aicache[j];
	  lin[j]+=(((a[i]*kernel_val)-(a_old[i]*kernel_val))*(double)label[i]);
	}
      }
    }
  }
  (*maxdiff)=0;
  for(i=0;i<totdoc;i++) {
    inactive_since[i]=deactnum-1;
    if(!inconsistent[i]) {
      dist=(lin[i]-model->b)*(double)label[i];
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if((a[i]>learn_parm->epsilon_a) && (dist > 1)) {
	if((dist-1.0)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-1.0;
      }
      else if((a[i]<ex_c) && (dist < 1)) {
	if((1.0-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=1.0-dist;
      }
      if((a[i]>(0+learn_parm->epsilon_a)) 
	 && (a[i]<ex_c)) { 
	active[i]=1;                            /* not at bound */
      }
      else if((a[i]<=(0+learn_parm->epsilon_a)) && (dist < (1+learn_parm->epsilon_shrink))) {
	active[i]=1;
      }
      else if((a[i]>=ex_c)
	      && (dist > (1-learn_parm->epsilon_shrink))) {
	active[i]=1;
      }
    }
  }
  for(i=0;i<totdoc;i++) {
    (a_history[deactnum-1])[i]=a[i];
  }
  for(t=deactnum-2;(t>=0) && a_history[t];t--) {
      free(a_history[t]);
      a_history[t]=0;
  }
  free(changed);
  free(changed2dnum);
  free(inactive);
  free(inactive2dnum);
}

/****************************** Cache handling *******************************/

void get_kernel_row(kernel_cache,docs,docnum,totdoc,active2dnum,buffer,
		    kernel_parm)
KERNEL_CACHE *kernel_cache;
DOC *docs;          /* Get's a row of the matrix of kernel values */
long docnum,totdoc; /* This matrix has the same form as the Hessian, */ 
long *active2dnum;  /* just that the elements are not multiplied by */
CFLOAT *buffer;     /* y_i * y_j * a_i * a_j */
KERNEL_PARM *kernel_parm; /* Takes the values from the cache if available. */
{
  register long i,j,start;
  DOC *ex;

  ex=&(docs[docnum]);
  if(kernel_cache->index[docnum] != -1) { /* is cached? */
    kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; /* lru */
    start=kernel_cache->activenum*kernel_cache->index[docnum];
    for(i=0;(j=active2dnum[i])>=0;i++) {
      if(kernel_cache->totdoc2active[j] >= 0) {
	buffer[j]=kernel_cache->buffer[start+kernel_cache->totdoc2active[j]];
      }
      else {
	buffer[j]=(CFLOAT)kernel(kernel_parm,ex,&(docs[j]));
      }
    }
  }
  else {
    for(i=0;(j=active2dnum[i])>=0;i++) {
      buffer[j]=(CFLOAT)kernel(kernel_parm,ex,&(docs[j]));
    }
  }
}

void get_kernel_row_offset(kernel_cache,docs,docnum,totdoc,active2dnum,buffer,
		    kernel_parm,offset,stepsize)
KERNEL_CACHE *kernel_cache;
DOC *docs;          /* Get's a row of the matrix of kernel values */
long docnum,totdoc; /* This matrix has the same form as the Hessian, */ 
long *active2dnum;  /* just that the elements are not multiplied by */
CFLOAT *buffer;     /* y_i * y_j * a_i * a_j */
KERNEL_PARM *kernel_parm; /* Takes the values from the cache if available. */
long offset,stepsize;
{
  register long i,j,start;
  DOC *ex;

  ex=&(docs[docnum]);
  if(kernel_cache->index[docnum] != -1) { /* is cached? */
    /* kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; /* lru */
    start=kernel_cache->activenum*kernel_cache->index[docnum];
    for(i=offset;(j=active2dnum[i])>=0;i+=stepsize) {
      if(kernel_cache->totdoc2active[j] >= 0) {
	buffer[j]=kernel_cache->buffer[start+kernel_cache->totdoc2active[j]];
      }
      else {
	buffer[j]=(CFLOAT)kernel(kernel_parm,ex,&(docs[j]));
      }
    }
  }
  else {
    for(i=offset;(j=active2dnum[i])>=0;i+=stepsize) {
      buffer[j]=(CFLOAT)kernel(kernel_parm,ex,&(docs[j]));
    }
  }
}

void cache_kernel_row(kernel_cache,docs,m,kernel_parm)
KERNEL_CACHE *kernel_cache;
DOC *docs;  /* Fills cache for the row m */
long m;
KERNEL_PARM *kernel_parm;
{
  register DOC *ex;
  register long i,j,k,l;
  register CFLOAT *cache;

  if(!kernel_cache_check(kernel_cache,m)) {  /* not cached yet*/
    cache = kernel_cache_clean_and_malloc(kernel_cache,m);
    if(cache) {
      l=kernel_cache->totdoc2active[m];
      ex=&(docs[m]);
      for(j=0;j<kernel_cache->activenum;j++) {  /* fill cache */
	k=kernel_cache->active2totdoc[j];
	if((kernel_cache->index[k] != -1) && (l != -1) && (k != m)) {
	  cache[j]=kernel_cache->buffer[kernel_cache->activenum
				       *kernel_cache->index[k]+l];
	}
	else {
	  cache[j]=kernel(kernel_parm,ex,&(docs[k]));
	} 
      }
    }
    else {
      perror("Error: Kernel cache full! => increase cache size");
    }
  }
}

 
void cache_multiple_kernel_rows(kernel_cache,docs,key,varnum,kernel_parm)
KERNEL_CACHE *kernel_cache;
DOC *docs;  /* Fills cache for the rows in key */
long *key;
long varnum;
KERNEL_PARM *kernel_parm;
{
  register DOC *ex;
  register long i,j,k,l,m;
  register CFLOAT *cache;

  for(i=0;i<varnum;i++) {  /* fill up kernel cache */
    cache_kernel_row(kernel_cache,docs,key[i],kernel_parm);
  }
}

 
void kernel_cache_shrink(kernel_cache,totdoc,numshrink,after)
KERNEL_CACHE *kernel_cache;
long totdoc,numshrink,*after; /* remove numshrink columns in the cache */
{                           /* which correspond to examples marked  */
  register long i,j,jj,from=0,to=0,scount;     /* 0 in after. */
  long *keep;

  if(verbosity>=2) {
    printf(" Reorganizing cache..."); fflush(stdout);
  }

  keep=(long *)my_malloc(sizeof(long)*totdoc);
  for(j=0;j<totdoc;j++) {
    keep[j]=1;
  }
  scount=0;
  for(jj=0;(jj<kernel_cache->activenum) && (scount<numshrink);jj++) {
    j=kernel_cache->active2totdoc[jj];
    if(!after[j]) {
      scount++;
      keep[j]=0;
    }
  }

  for(i=0;i<kernel_cache->max_elems;i++) {
    for(jj=0;jj<kernel_cache->activenum;jj++) {
      j=kernel_cache->active2totdoc[jj];
      if(!keep[j]) {
	from++;
      }
      else {
	kernel_cache->buffer[to]=kernel_cache->buffer[from];
	to++;
	from++;
      }
    }
  }

  kernel_cache->activenum=0;
  for(j=0;j<totdoc;j++) {
    if((keep[j]) && (kernel_cache->totdoc2active[j] != -1)) {
      kernel_cache->active2totdoc[kernel_cache->activenum]=j;
      kernel_cache->totdoc2active[j]=kernel_cache->activenum;
      kernel_cache->activenum++;
    }
    else {
      kernel_cache->totdoc2active[j]=-1;
    }
  }

  kernel_cache->max_elems=(long)(kernel_cache->buffsize/kernel_cache->activenum);
  if(kernel_cache->max_elems>totdoc) {
    kernel_cache->max_elems=totdoc;
  }

  free(keep);

  if(verbosity>=2) {
    printf("done.\n"); fflush(stdout);
    printf(" Cache-size in rows = %ld\n",kernel_cache->max_elems);
  }
}

void kernel_cache_init(kernel_cache,totdoc,buffsize)
KERNEL_CACHE *kernel_cache;
long totdoc,buffsize;
{
  long i;

  kernel_cache->index = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->occu = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->lru = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->invindex = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->active2totdoc = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->totdoc2active = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->buffer = (CFLOAT *)my_malloc(buffsize*1024*1024);

  kernel_cache->buffsize=(long)(buffsize*1024*1024/sizeof(CFLOAT));

  kernel_cache->max_elems=(long)(kernel_cache->buffsize/totdoc);
  if(kernel_cache->max_elems>totdoc) {
    kernel_cache->max_elems=totdoc;
  }

  if(verbosity>=2) {
    printf(" Cache-size in rows = %ld\n",kernel_cache->max_elems);
    printf(" Kernel evals so far: %ld\n",kernel_cache_statistic);    
  }

  kernel_cache->elems=0;   /* initialize cache */
  for(i=0;i<totdoc;i++) {
    kernel_cache->index[i]=-1;
  }
  for(i=0;i<kernel_cache->max_elems;i++) {
    kernel_cache->occu[i]=0;
    kernel_cache->invindex[i]=-1;
  }

  kernel_cache->activenum=totdoc;;
  for(i=0;i<totdoc;i++) {
      kernel_cache->active2totdoc[i]=i;
      kernel_cache->totdoc2active[i]=i;
  }

  kernel_cache->time=0;  
} 

/* same as kernel_cache_init but no malloc'ing */
void kernel_cache_quick_init(kernel_cache,totdoc,buffsize)
KERNEL_CACHE *kernel_cache;
long totdoc,buffsize;
{
  long i;

  kernel_cache->buffsize=(long)(buffsize*1024*1024/sizeof(CFLOAT));

  kernel_cache->max_elems=(long)(kernel_cache->buffsize/totdoc);
  if(kernel_cache->max_elems>totdoc) {
    kernel_cache->max_elems=totdoc;
  }

  if(verbosity>=2) {
    printf(" Cache-size in rows = %ld\n",kernel_cache->max_elems);
    printf(" Kernel evals so far: %ld\n",kernel_cache_statistic);    
  }

  /* initialize cache */

  kernel_cache->elems=0;  
  for(i=0;i<totdoc;i++) {
    kernel_cache->index[i]=-1;
  }
  for(i=0;i<kernel_cache->max_elems;i++) {
    kernel_cache->occu[i]=0;
    kernel_cache->invindex[i]=-1;
  }

  kernel_cache->activenum=totdoc;;
  for(i=0;i<totdoc;i++) {
      kernel_cache->active2totdoc[i]=i;
      kernel_cache->totdoc2active[i]=i;
  }
  kernel_cache->time=0;  
} 

void kernel_cache_cleanup(kernel_cache)
KERNEL_CACHE *kernel_cache;
{
  free(kernel_cache->index);
  free(kernel_cache->occu);
  free(kernel_cache->lru);
  free(kernel_cache->invindex);
  free(kernel_cache->active2totdoc);
  free(kernel_cache->totdoc2active);
  free(kernel_cache->buffer);
}

long kernel_cache_malloc(kernel_cache) 
KERNEL_CACHE *kernel_cache;
{
  long i;

  if(kernel_cache->elems < kernel_cache->max_elems) {
    for(i=0;i<kernel_cache->max_elems;i++) {
      if(!kernel_cache->occu[i]) {
	kernel_cache->occu[i]=1;
	kernel_cache->elems++;
	return(i);
      }
    }
  }
  return(-1);
}

void kernel_cache_free(kernel_cache,i)
KERNEL_CACHE *kernel_cache;
long i;
{
  kernel_cache->occu[i]=0;
  kernel_cache->elems--;
}

long kernel_cache_free_lru(kernel_cache) /* remove least recently used cache */
KERNEL_CACHE *kernel_cache;              /* element */
{                                     
  register long k,least_elem=-1,least_time;

  least_time=kernel_cache->time+1;
  for(k=0;k<kernel_cache->max_elems;k++) {
    if(kernel_cache->invindex[k] != -1) {
      if(kernel_cache->lru[k]<least_time) {
	least_time=kernel_cache->lru[k];
	least_elem=k;
      }
    }
  }
  if(least_elem != -1) {
    kernel_cache_free(kernel_cache,least_elem);
    kernel_cache->index[kernel_cache->invindex[least_elem]]=-1;
    kernel_cache->invindex[least_elem]=-1;
    return(1);
  }
  return(0);
}


CFLOAT *kernel_cache_clean_and_malloc(kernel_cache,docnum)
KERNEL_CACHE *kernel_cache; 
long docnum;  /* Get a free cache entry. In case cache is full, the lru */
{             /* element is removed. */
  long result;
  if((result = kernel_cache_malloc(kernel_cache)) == -1) {
    if(kernel_cache_free_lru(kernel_cache)) {
      result = kernel_cache_malloc(kernel_cache);
    }
  }
  kernel_cache->index[docnum]=result;
  if(result == -1) {
    return(0);
  }
  kernel_cache->invindex[result]=docnum;
  kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; /* lru */
  return((CFLOAT *)((long)kernel_cache->buffer
		    +(kernel_cache->activenum*sizeof(CFLOAT)*
		      kernel_cache->index[docnum])));
}

long kernel_cache_touch(kernel_cache,docnum)
KERNEL_CACHE *kernel_cache; 
long docnum; /* Update lru time to avoid removal from cache. */
{
  if(kernel_cache && kernel_cache->index[docnum] != -1) {
    kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; /* lru */
    return(1);
  }
  return(0);
}
  
long kernel_cache_check(kernel_cache,docnum)
KERNEL_CACHE *kernel_cache; 
long docnum; /* Is that row cached? */
{
  return(kernel_cache->index[docnum] != -1);
}
  
/************************** Compute bounds ******************************/

double estimate_loo_bound(model,label,unlabeled,totdoc) /* loo-bound based on observation */
MODEL *model;                           /* that loo-errors must have an   */
long *label,*unlabeled,totdoc;                            /* alpha greater equal 1.0        */
{
  long j,l=0,ulab=0,lab=0,labpos=0,labneg=0,totulab=0;
  double totlab=0,totlabpos=0,totlabneg=0,labsum=0,ulabsum=0,lsum=0;;

  for(j=0;j<totdoc;j++) {
    if(unlabeled[j]) {
      totulab++;
    }
    else {
      totlab++;
      if(label[j] > 0) 
	totlabpos++;
      else 
	totlabneg++;
    }
  }
  for(j=1;j<model->sv_num;j++) {
   if(fabs(model->alpha[j]) >= 1) {  /* only if examples have unit length */
      l++;
      lsum+=fabs(model->alpha[j]);
      if(unlabeled[model->supvec[j]->docnum]) {
	ulab++;
	ulabsum+=fabs(model->alpha[j]);
      }
      else {
	lab++;
	if(model->alpha[j] > 0) 
	  labpos++;
	else 
	  labneg++;
	labsum+=fabs(model->alpha[j]);
      }
    }
  }
  printf("Alphas>=1: labeledpos=%.1f labeledneg=%.1f default=%.1f\n",(double)labpos/(double)totlab*100.0,(double)labneg/(double)totlab*100.0,(double)totlabpos/(double)(totlab)*100.0);
  printf("Alphas>=1: labeled=%.1f unlabled=%.1f all=%.1f\n",(double)lab/(double)totlab*100.0,(double)ulab/(double)totulab*100.0,(double)l/(double)(totdoc)*100.0);
  printf("Alphassum: labeled=%.1f unlabled=%.1f all=%.1f\n",(double)labsum/(double)totlab*100.0,(double)ulabsum/(double)totulab*100.0,(double)lsum/(double)(totdoc)*100.0);
  return((double)l/(double)(totdoc-1));
}

double estimate_margin_vcdim(model,w,kernel_parm) 
MODEL *model;
double w; /* optional: length of model vector in feature space */
KERNEL_PARM *kernel_parm;
{
  double h,R;

  /* follows chapter 5.6.4 in [Vapnik/95] */

  if(w<0) {
    w=model_length_s(model,kernel_parm);
  }
  R=estimate_sphere(model,kernel_parm); 
  h = w*w * R*R +1; 
  return(h);
}

double estimate_sphere(model,kernel_parm) 
MODEL *model;             /* Approximates the radius of the ball containing */
KERNEL_PARM *kernel_parm; /* the support vectors by bounding it with the */
{                         /* length of the longest support vector. This is */
  register long j;        /* pretty good for text categorization, since all */
  double xlen,maxxlen=0;  /* documents have feature vectors of length 1. It */
  DOC nulldoc;            /* assumes that the center of the ball is at the */
  WORD nullword;          /* origin of the space. */

  if(verbosity>=1) {
    fprintf(stdout,"Estimating radius R of sphere..."); fflush(stdout);
  }
  nullword.wnum=0;
  nulldoc.words=&nullword;
  nulldoc.twonorm_sq=0;
  nulldoc.docnum=-1;

  for(j=1;j<model->sv_num;j++) {
    xlen=sqrt(kernel(kernel_parm,model->supvec[j],model->supvec[j])
	      -2*kernel(kernel_parm,model->supvec[j],&nulldoc)
	      +kernel(kernel_parm,&nulldoc,&nulldoc));
    if(xlen>maxxlen) {
      maxxlen=xlen;
    }
  }
  if(verbosity>=1) {
    fprintf(stdout,"done (R <= longest support vector = %.5f)\n",maxxlen);
  }
  
  return(maxxlen);
}

/****************************** IO-handling **********************************/

void write_model(modelfile,model,kernel_parm)
char *modelfile;
MODEL *model;
KERNEL_PARM *kernel_parm;
{
  FILE *modelfl;
  long j,i;

  if(verbosity>=1) {
    printf("Writing model file..."); fflush(stdout);
  }
  if ((modelfl = fopen (modelfile, "w")) == NULL)
  { perror (modelfile); exit (1); }
  fprintf(modelfl,"SVM-light Version %s\n",VERSION);
  fprintf(modelfl,"%ld # kernel type\n",kernel_parm->kernel_type);
  fprintf(modelfl,"%ld # kernel parameter -d \n",kernel_parm->poly_degree);
  fprintf(modelfl,"%.8g # kernel parameter -g \n",kernel_parm->rbf_gamma);
  fprintf(modelfl,"%.8g # kernel parameter -s \n",kernel_parm->coef_lin);
  fprintf(modelfl,"%.8g # kernel parameter -r \n",kernel_parm->coef_const);
  fprintf(modelfl,"%s # kernel parameter -u \n",kernel_parm->custom);
 
  fprintf(modelfl,"%ld # number of support vectors \n",model->sv_num);
  fprintf(modelfl,"%.8g # threshold b \n",model->b);
  for(i=1;i<model->sv_num;i++) {
    fprintf(modelfl,"%.32g ",model->alpha[i]);
    for (j=0; ((model->supvec[i])->words[j]).wnum; j++) {
      fprintf(modelfl,"%ld:%.8g ",
	      (long)((model->supvec[i])->words[j]).wnum,
	      (double)(((model->supvec[i])->words[j]).weight));
    }
    fprintf(modelfl,"\n");
  }
  fclose(modelfl);
  if(verbosity>=1) {
    printf("done\n");
  }
}

void write_queries(queryfile,queries,num_query_batches,num_simultaneous_queries)
char *queryfile;
long *queries;
int num_query_batches;
int num_simultaneous_queries;
{
  FILE *queryfl;
  long j,i;

  if(verbosity>=1) {
    printf("Writing query file..."); fflush(stdout);
  }
  if ((queryfl = fopen (queryfile, "w")) == NULL)
  { perror (queryfile); exit (1); }
  for(i=0; i<num_query_batches*num_simultaneous_queries; i++) {
    fprintf(queryfl,"%ld\n",queries[i]);
  }
  fclose(queryfl);
  if(verbosity>=1) {
    printf("done\n");
  }
}


void write_prediction(predfile,model,lin,a,unlabeled,label,totdoc,learn_parm)
char *predfile;
MODEL *model;
double *lin,*a;
long *unlabeled;
long *label;
long totdoc;
LEARN_PARM *learn_parm;
{
  FILE *predfl;
  long j,i;
  double dist,a_max;

  if(verbosity>=1) {
    printf("Writing prediction file..."); fflush(stdout);
  }
  if ((predfl = fopen (predfile, "w")) == NULL)
  { perror (predfile); exit (1); }
  a_max=learn_parm->epsilon_a;
  for(i=0;i<totdoc;i++) {
    if((unlabeled[i]) && (a[i]>a_max)) {
      a_max=a[i];
    }
  }
  for(i=0;i<totdoc;i++) {
    if(unlabeled[i]) {
      if((a[i]>(learn_parm->epsilon_a))) {
	dist=(double)label[i]*(1.0-learn_parm->epsilon_crit-a[i]/(a_max*2.0));
      }
      else {
	dist=(lin[i]-model->b);
      }
      if(dist>0) {
	fprintf(predfl,"%.8g:+1 %.8g:-1\n",dist,-dist);
      }
      else {
	fprintf(predfl,"%.8g:-1 %.8g:+1\n",-dist,dist);
      }
    }
  }
  fclose(predfl);
  if(verbosity>=1) {
    printf("done\n");
  }
}

void write_alphas(alphafile,a,label,totdoc)
char *alphafile;
double *a;
long *label;
long totdoc;
{
  FILE *alphafl;
  long i;

  if(verbosity>=1) {
    printf("Writing alpha file..."); fflush(stdout);
  }
  if ((alphafl = fopen (alphafile, "w")) == NULL)
  { perror (alphafile); exit (1); }
  for(i=0;i<totdoc;i++) {
    fprintf(alphafl,"%.8g\n",a[i]*(double)label[i]);
  }
  fclose(alphafl);
  if(verbosity>=1) {
    printf("done\n");
  }
}

/* Causes sorting in decreasing order */
int inverseLongComp (const void* e1, const void* e2)
{
  long *p1, *p2;
  p1 = (long*)e1;
  p2 = (long*)e2;

  if (*p1 > *p2) {
    return (-1);
  } else if (*p1 < *p2) {
    return (1);
  } else {
    return (0);
  }
}

void read_input_parameters(argc,argv,docfile,modelfile,verbosity,
			   kernel_cache_size,learn_parm,kernel_parm,
			   num_query_batches_ptr, trans_ptr, every_label_file,
			   trans_modelfile, active_type_ptr, hybrid_threshold_ptr,
			   num_simultaneous_queries_ptr, num_candidate_queries_ptr)
int argc, *trans_ptr, *active_type_ptr, *num_simultaneous_queries_ptr, *num_candidate_queries_ptr;
char *argv[],*docfile,*modelfile, *every_label_file, *trans_modelfile;
long *verbosity,*kernel_cache_size, *num_query_batches_ptr, *hybrid_threshold_ptr;
LEARN_PARM *learn_parm;
KERNEL_PARM *kernel_parm;
{
  long i;
  
  /* set default */
  strcpy (modelfile, "svm_model");
  strcpy (trans_modelfile, "svm_trans_model");
  strcpy (every_label_file, "labels");
  strcpy (learn_parm->predfile, "trans_predictions");
  strcpy (learn_parm->alphafile, "");
  (*verbosity)=1;
  (*num_query_batches_ptr)=0;
  (*trans_ptr)=1;
  (*active_type_ptr)=0;
  (*hybrid_threshold_ptr)=100000000;
  learn_parm->biased_hyperplane=1;
  learn_parm->remove_inconsistent=0;
  learn_parm->skip_final_opt_check=0;
  learn_parm->svm_maxqpsize=10;
  learn_parm->svm_iter_to_shrink=-9999;
  (*kernel_cache_size)=40;
  learn_parm->svm_c=1000.0;
  learn_parm->transduction_posratio=-1.0;
  learn_parm->svm_costratio=1.0;
  learn_parm->svm_costratio_unlab=1.0;
  learn_parm->svm_unlabbound=1E-5;
  learn_parm->epsilon_crit=0.001;
  learn_parm->epsilon_a=1E-12;
  kernel_parm->kernel_type=0;
  kernel_parm->poly_degree=3;
  kernel_parm->rbf_gamma=1.0;
  kernel_parm->coef_lin=1;
  kernel_parm->coef_const=1;
  strcpy(kernel_parm->custom,"empty");
  (*num_simultaneous_queries_ptr) = 1;
  (*num_candidate_queries_ptr) = 1;

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) 
      { 
      case '?': print_help(); exit(0);
      case 'v': i++; (*verbosity)=atol(argv[i]); break;
      case 'b': i++; learn_parm->biased_hyperplane=atol(argv[i]); break;
      case 'i': i++; learn_parm->remove_inconsistent=atol(argv[i]); break;
      case 'f': i++; learn_parm->skip_final_opt_check=!atol(argv[i]); break;
      case 'q': i++; learn_parm->svm_maxqpsize=atol(argv[i]); break;
      case 'h': i++; learn_parm->svm_iter_to_shrink=atol(argv[i]); break;
      case 'm': i++; (*kernel_cache_size)=atol(argv[i]); break;
      case 'c': i++; learn_parm->svm_c=atof(argv[i]); break;
      case 'p': i++; learn_parm->transduction_posratio=atof(argv[i]); break;
      case 'j': i++; learn_parm->svm_costratio=atof(argv[i]); break;
      case 'e': i++; learn_parm->epsilon_crit=atof(argv[i]); break;
      case 't': i++; kernel_parm->kernel_type=atol(argv[i]); break;
      case 'd': i++; kernel_parm->poly_degree=atol(argv[i]); break;
      case 'g': i++; kernel_parm->rbf_gamma=atof(argv[i]); break;
      case 's': i++; kernel_parm->coef_lin=atof(argv[i]); break;
      case 'r': i++; kernel_parm->coef_const=atof(argv[i]); break;
      case 'u': i++; strcpy(kernel_parm->custom,argv[i]); break;
      case 'l': i++; strcpy(learn_parm->predfile,argv[i]); break;
      case 'a': i++; strcpy(learn_parm->alphafile,argv[i]); break;
      case 'k': i++; strcpy(every_label_file,argv[i]); break;
      case 'n': i++; (*num_query_batches_ptr)=atol(argv[i]); break;
      case 'o': i++; (*active_type_ptr)=atoi(argv[i]); break;
      case 'w': i++; (*num_simultaneous_queries_ptr)=atoi(argv[i]); break;
      case 'x': i++; (*hybrid_threshold_ptr)=atol(argv[i]); break;
      case 'y': i++; strcpy(trans_modelfile,argv[i]); break;
      case 'z': i++; (*trans_ptr)=atoi(argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n",argv[i]);
	       print_help();
	       exit(0);
      }
  }
  if(i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    print_help();
    exit(0);
  }
  strcpy (docfile, argv[i]);
  if((i+1)<argc) {
    strcpy (modelfile, argv[i+1]);
  }
  if(learn_parm->svm_iter_to_shrink == -9999) {
    if(kernel_parm->kernel_type == LINEAR) 
      learn_parm->svm_iter_to_shrink=2;
    else
      learn_parm->svm_iter_to_shrink=100;
  }
  if((learn_parm->skip_final_opt_check) 
     && (kernel_parm->kernel_type == LINEAR)) {
    printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n");
  }    
  if((learn_parm->svm_maxqpsize<2) || (learn_parm->svm_maxqpsize>nx)) {
    printf("\nMaximum size of QP-subproblems not in valid range: %ld [2,%ld]\n",learn_parm->svm_maxqpsize,(long)nx); 
    print_help();
    exit(0);
  }
  if(learn_parm->svm_iter_to_shrink<1) {
    printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
    print_help();
    exit(0);
  }
  if(learn_parm->svm_c<=0) {
    printf("\nThe C parameter must be greater than zero!\n\n");
    print_help();
    exit(0);
  }
  if(learn_parm->transduction_posratio>1) {
    printf("\nThe fraction of unlabeled examples to classify as positives must\n");
    printf("be less than 1.0 !!!\n\n");
    print_help();
    exit(0);
  }
  if(learn_parm->svm_costratio<=0) {
    printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
    print_help();
    exit(0);
  }
  if(learn_parm->epsilon_crit<=0) {
    printf("\nThe epsilon parameter must be greater than zero!\n\n");
    print_help();
    exit(0);
  }

  *num_candidate_queries_ptr = *num_simultaneous_queries_ptr;
#ifdef SMART_SIM_QUERIES
  *num_candidate_queries_ptr = *num_simultaneous_queries_ptr * 2;
#endif


}

void print_help()
{
  printf("\nSVM-light %s: Support Vector Machine, learning module     %s\n",VERSION,VERSION_DATE);
  copyright_notice();
  printf("   usage: svm_learn [options] example_file model_file\n\n");
  printf("Arguments:\n");
  printf("         example_file-> file with training data\n");
  printf("         model_file  -> file to store learned decision rule in\n");

  printf("General options:\n");
  printf("         -?          -> this help\n");
  printf("         -v [0..3]   -> verbosity level (default 1)\n");
  printf("Learning options:\n");
  printf("         -c float    -> C: trade-off between training error\n");
  printf("                        and margin (default 1000)\n");
  printf("         -j float    -> Cost: cost-factor, by which training errors on\n");
  printf("                        positive examples outweight errors on negative\n");
  printf("                        examples (default 1)\n");
  printf("         -b [0,1]    -> use biased hyperplane (i.e. x*w+b>0) instead\n");
  printf("                        of unbiased hyperplane (i.e. x*w>0) (default 1)\n");
  printf("         -i [0,1]    -> remove inconsistent training examples\n");
  printf("                        and retrain (default 0)\n");
  printf("Active Learning options:\n");
  printf("         -k char     -> file that contains the labels for all of the\n");
  printf("                        documents in the training file (including the\n");
  printf("                        unlabeled documents in the training file)\n");
  printf("         -n int      -> number of batches of queries permitted (default 0)\n");
  printf("         -o [0,1,2,3]-> use ratio, simple, margin active or random (default 0)\n");
  printf("         -w int      -> number of simultaneous queries to perform\n");
  printf("         -x int      -> threshold to stop using ratio/margin and start using simple\n");
  printf("Transduction options:\n");
  printf("         -p [0..1]   -> fraction of unlabeled examples to be classified\n");
  printf("                        into the positive class (default is the ratio of\n");
  printf("                        positive and negative examples in the training data)\n");
  printf("         -y char     -> file to write transduction model to if -z 2 is set\n");
  printf("         -z [0,1,2]  -> 0=disable transduction, 1=enable, 2=both (default 1)\n");
  printf("Kernel options:\n");
  printf("         -t int      -> type of kernel function:\n");
  printf("                        0: linear (default)\n");
  printf("                        1: polynomial (s a*b+c)^d\n");
  printf("                        2: radial basis function exp(-gamma ||a-b||^2)\n");
  printf("                        3: sigmoid tanh(s a*b + c)\n");
  printf("                        4: user defined kernel from kernel.h\n");
  printf("         -d int      -> parameter d in polynomial kernel\n");
  printf("         -g float    -> parameter gamma in rbf kernel\n");
  printf("         -s float    -> parameter s in sigmoid/poly kernel\n");
  printf("         -r float    -> parameter c in sigmoid/poly kernel\n");
  printf("         -u string   -> parameter of user defined kernel\n");
  printf("Optimization options:\n");
  printf("         -q [2..%d] -> maximum size of QP-subproblems (default 10)\n",nx);
  printf("         -m [5..]    -> size of cache for kernel evaluations in MB (default 40)\n");
  printf("                        The larger the faster...\n");
  printf("         -e float    -> eps: Allow that error for termination criterion\n");
  printf("                        [y [w*x+b] - 1] >= eps (default 0.001)\n");
  printf("         -h [5..]    -> number of iterations a variable needs to be\n"); 
  printf("                        optimal before considered for shrinking (default 100)\n");
  printf("         -f [0,1]    -> do final optimality check for variables removed\n");
  printf("                        by shrinking. Although this test is usually \n");
  printf("                        positive, there is no guarantee that the optimum\n");
  printf("                        was found if the test is omitted. (default 1)\n");
  printf("Output options:\n");
  printf("         -l char     -> file to write predicted labels of unlabeled\n");
  printf("                        examples into after transductive learning\n");
  printf("         -a char     -> write all alphas to this file after learning\n");
  printf("                        (in the same order as in the training set)\n\n");
}



