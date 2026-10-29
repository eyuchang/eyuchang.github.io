//
// Simulator for the RIME paper to SIGMOD'99
//
//   Sept. 1998
//
//   Chen Li
//   Computer Science Department
//   Stanford University, CA 94309
//   chenli@cs.stanford.edu
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "knn.h"
#include "point.h"
#include "directory.h"
#include "cluster.h"
#include "public.h"
#include "clustering.h"
#include "sockserver.h"

#include <sys/types.h>
#include <unistd.h>
#include <strings.h>

void SockInit();

struct StdTable stdTable[32] = {
  {4,  -1.751},
  {8,  -1.415},
  {11, -1.227},
  {14, -1.08},
  {17, -0.954},
  {20, -0.842},
  {23, -0.741},
  {26, -0.643},
  {29, -0.553},
  {32, -0.487},
  {35, -0.385},
  {38, -0.315},
  {41, -0.226},
  {44, -0.151},
  {47, -0.075},
  {50, 0},
  {53, 0.075},
  {56, 0.151},
  {59, 0.226},
  {62, 0.315},
  {65, 0.385},
  {68, 0.478},
  {71, 0.553},
  {74, 0.643},
  {77, 0.741},
  {77, 0.741},
  {80, 0.842},
  {83, 0.954},
  {86, 1.08},
  {89, 1.227},
  {92, 1.415},
  {96, 1.751}
};

#define DATA_FILENAME "/dfs/sole/2/chenli/Papers/1999/rime-VLDB99/sim/convfile/v1.2/MF.feature.48"
//#define DATA_FILENAME "MF.feature.48.sample"
#define C123STATS_FILENAME "/dfs/sole/2/chenli/Papers/1999/rime-VLDB99/sim/convfile/v1.2/C123_stats.txt"

double deviations[DIMENSION]; 
double mean[DIMENSION]; 

double adjcentRatio;
int    K;

#define NUMBITS_PERDIM_FINE   ( K    )  // number of bits for fine granularity

//#define MIN_VALUE             ( -32768 )
//#define MIN_VALUE      ( 625 )
#define MIN_VALUE      ( 5000 )
//#define MAX_VALUE             ( 32767  )
#define MAX_VALUE      ( 16270 )
#define MAX_POINT_NUM  ( 30000 ) //  >= 29974, Lion images are 29974-29984

//#define MAX_POINT_NUM         ( 40 )
#define DISK_BLOCK_SIZE       ( 4096 )

//int         horizon = 1;
int         horizon;

unsigned char bigIntMask;

void  Init();
void  ReadInData();   // read in the data
void  CleanUp();
void  GetRecord(const char *line, Point *point);
void  PrintStat();
void  ReadParameter(int argc, char *argv[]);

int         randomSeed;

// points
Point       *pointArray;
int         pointNum;

// directory, clusters, ocean
Directory   *directory;
Cluster     *clusters;
int         lastClusterNum;     // current cluster number
int         maxClusterNum;  // maximal cluster number

Cluster     *oceanCluster;

int                  sockfd, newsockfd, clilen;
struct sockaddr_in   cli_addr, serv_addr;

main(int argc, char *argv[])
{
  ReadParameter(argc, argv);
  Init();
  ReadInData();   // read in the data
  DoClustering(); // cluster the data
  PrintStat();    // print statistics
  DoKNN();        // do k-nearest neighbor search
  CleanUp();
}

void
SockInit()
{
  if ( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0 )
    err_dump("Server: can't open stream socket");
  
  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sin_family        = AF_INET;
  serv_addr.sin_addr.s_addr   = htonl(INADDR_ANY);
  serv_addr.sin_port          = htons(SERV_TCP_PORT);
  
  if ( bind(sockfd, (struct sockaddr *) &serv_addr, 
	    sizeof(serv_addr)) < 0 )
    err_dump("server: can't bind local address");
  
  listen(sockfd, 5);
}

//------------------------------------------------------
//  Function   Init()
//
//  Initialize.
//------------------------------------------------------
void
ReadParameter(int argc, char *argv[])
{
  if ( argc != 4 )
  {
    printf("Usage: MF [K] [ratio] [theta] \n");
    exit(1);
  }

  K = atoi(argv[1]);
  adjcentRatio = atof(argv[2]);
  horizon = atoi(argv[3]);
  printf("K = %d, adjacent ratio = %f, theta = %d.\n", K, adjcentRatio,
	 horizon); 
  SockInit();
}

//------------------------------------------------------
//  Function   Init()
//
//  Initialize.
//------------------------------------------------------
void
Init()
{
  bigIntMask = 0;
  for ( int i = 0; i <= K - 1; i ++ ) // each dimension one substring
    bigIntMask = ( bigIntMask << 1 ) | 0x1;
  
  pointArray = NULL;
  pointArray = new Point[MAX_POINT_NUM];
  if ( pointArray == NULL )
  {
    printf("Can't allocate array for points.\n");
    exit(1);
  }
  
  randomSeed = 466;              // random seed
  srand( randomSeed );
  
  // mountain clusters and their directory
  maxClusterNum = 30000;  // maximal cluster number
  clusters = new Cluster[maxClusterNum](DIMENSION);
  if ( clusters == NULL )
  {
    printf("Init(): Can't allocate memory.\n");
    exit(1);
  }

  // directory
  directory = new Directory(DIMENSION, K, MIN_VALUE, MAX_VALUE);
  if ( directory == NULL )
  {
    printf("Init(): Can't allocate memory for directory.\n");
    exit(1);
  }

  lastClusterNum = 0;            // # of mountain clusters

  // ocean
  oceanCluster = new Cluster(DIMENSION);
  if ( oceanCluster == NULL )
  {
    printf("Init(): Can't allocate memory for oceanCluster.\n");
    exit(1);
  }

  FILE *statFp;
  if ((statFp = fopen(C123STATS_FILENAME, "r")) == NULL) 
  {
    perror(DATA_FILENAME);
    exit(1);
  }

  for ( int i = 0; i <= DIMENSION - 1; i ++ )
  {
    fscanf(statFp, "%lf", &(mean[i]));
    fscanf(statFp, "%lf", &(deviations[i]));
    //    printf(" %f %f \n", mean[i], deviations[i] );
  }
  fclose(statFp);
}

//------------------------------------------------------
//  Function   CleanUp()
//
//  Clean up.
//------------------------------------------------------
void
CleanUp()
{
  delete []pointArray;
  delete directory;
  delete []clusters;
  delete oceanCluster;
}

//------------------------------------------------------
//  Function   ReadInData()
//
//  Read in the data.
//------------------------------------------------------
void
ReadInData()
{
  FILE        *fp;
  char        buffer[BUFFER_SIZE];

  if ((fp = fopen(DATA_FILENAME, "r")) == NULL) 
  {
    perror(DATA_FILENAME);
    exit(1);
  }

  pointNum = 0;

  while(fgets(buffer, sizeof(buffer), fp) != NULL) 
  {
    if(buffer[0] == '#' || buffer[0] == '\n')
      continue;
    
    GetRecord(buffer, &pointArray[pointNum]);
    pointNum ++;
  }
  
  fclose(fp);
}

//------------------------------------------------------
//  Function   GetRecord()
//
//   Read records from file
//------------------------------------------------------
void
GetRecord(const char *line,  Point *point_return)
{
  const char *head, *tail;
  char        buffer[BUFFER_SIZE];
  int         i, length;
  Point       &point = *point_return;
  
  head = line;
  
  // fname 
  if((tail = strchr(head, ':')) == NULL) 
  {
    fprintf(stderr, "`:' is expected.\n");
    exit(1);
  }
  length = tail - head;
  head = tail + 1;
  
  // point 
  for ( i = 0; i < DIMENSION; i++) 
  {
    if((tail = strchr(head, ':')) == NULL)
    {
      fprintf(stderr, "`:' is expected.\n");
      exit(1);
    }
    length = tail - head;
    strncpy(buffer, head, length);
    buffer[length] = 0;
    
    point.SetCoord(i, atof(buffer));
    head = tail + 1;
  }
}


//------------------------------------------------------
//  Function   PrintStat()
//
//   Print statistics
//------------------------------------------------------
void
PrintStat()
{
  int         clusterNum;     // current cluster number

  clusterNum = 0;

  int maxClusterSize = -1;
  int minClusterSize = 1000000;
  int sumClusterSize = 0;
  for (int i = 0; i <= lastClusterNum - 1; i ++ )
  {
    if ( !clusters[i].valid )
      continue;
    clusterNum ++;
    int clusterSize = clusters[i].GetPointNum();
    sumClusterSize += clusterSize;
    if ( maxClusterSize < clusterSize )
      maxClusterSize = clusterSize;
    if ( minClusterSize > clusterSize )
      minClusterSize = clusterSize;
  }
  
  printf("Horizon (theta) = %d.\n", horizon);
  printf("K = %d.\n", K);
  printf("Cluster # = %d.\n", clusterNum);
  
  printf("Cell # = %d.\n", directory->GetCellNum());
  printf("MaxClusterSize = %d.\n", maxClusterSize);
  printf("AvgClusterSize = %d.\n", sumClusterSize / clusterNum);
  printf("MinClusterSize = %d.\n", minClusterSize);

  printf("Ocean Size = %d.\n", oceanCluster->GetPointNum());
  printf("Adjacent Ratio = %f.\n", adjcentRatio);
}
