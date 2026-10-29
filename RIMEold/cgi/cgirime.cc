//
// This is a cgi program for RIME demo.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>

#include <sys/types.h>
#include <unistd.h>
#include <netdb.h>
#include <strings.h>
#include <arpa/inet.h>
#include "public.h"
#include "util.h"

#define EMAIL_FILENAME "webcomments.txt"
int imageArray[TOPK];
char imageNames[TOPK][MAXLINE];
char imageDistance[TOPK][MAXLINE];
char queryImageFile[MAXLINE];

void GetImages(int sockfd);
bool CommToServer();

ArgStruct *argArray;  // argument array
int       numArgs = 0;
void      ReadInArg();
void      WriteAnswerHTML();
void      WriteSorryHTML();
int       parseFormArgs(const char *FArgs);

int                  sockfd;
struct sockaddr_in   serv_addr;
struct hostent       *hp;
int                  imageID;
int                  numReturnedImages;
float                threshold;

int
main()
{
  char *reqMethod;
  char *imageIDStr;
  char *thresholdStr;

  for ( int i = 0; i <= TOPK - 1; i ++ )
    imageArray[i] = i + 20;

  reqMethod = getenv("REQUEST_METHOD");
  if ( !strcmp(reqMethod, "GET") ) // "GET" method
  {
    imageIDStr = getenv("QUERY_STRING");
  }
  else // "post" methord
  {
    ReadInArg();  // read in arguments
    if ( !FindFormArg("imageID", imageIDStr) ) 
    {
      WriteSorryHTML();
      exit(1);
    }

    if ( !FindFormArg("threshold", thresholdStr) ) 
    {
      WriteSorryHTML();
      exit(1);
    }
    ;

  }
    
  imageID = atoi(imageIDStr);
  threshold = atof(thresholdStr);

  if ( !CommToServer() )
  {
    WriteSorryHTML();
    exit(1);
  }

  WriteAnswerHTML();
  return ( 0 );
}

// Read in the argument from the user input
void
ReadInArg()
{
  char *buf;
  int  bufLen;
  int  bytesRead;
  char *inputArgs;
  int  totalBytesRead;

  buf       = getenv("CONTENT_LENGTH");
  bufLen    = atoi(buf);  // get the length of input buffer
  inputArgs = new char[bufLen + 10]; // allocate our own buffer
  
  // read in those arguments
  totalBytesRead = 0;
  while ((bytesRead = read(0, 
			  inputArgs + totalBytesRead, 
			  bufLen - totalBytesRead))) 
    totalBytesRead += bytesRead;
  
  // double check whether we have read successfully
  if (totalBytesRead != bufLen) 
  {
    cerr << "Read " 
	 << totalBytesRead << " instead of " 
	 << bufLen << endl;
    delete []inputArgs ;
    return;
  }

  if (parseFormArgs(inputArgs) != STR_GOOD) 
  {
    cerr << "Error parsing form args" << endl;
    delete [] inputArgs;
    return;
  }
  
  delete [] inputArgs;
}
//
// Function: WriteAnwserHTML
//
//    Write the content to the HTML file (with the file pointer "fp").
//
void  
WriteAnswerHTML()
{
  char urlStr[300];
  int  i;
  int  realImageNum;

  realImageNum = 0;
  for ( i = 0; i <= numReturnedImages - 1; i ++ )
  {
    if (imageNames[i] < 0)
      break;
    realImageNum ++;
  }

  cout
    << "Content-type: text/html\n\n"
    << "<TITLE>RIME Demo</TITLE>" << endl
    << "</HEAD>" << endl
    << "<BODY TEXT=\"#000000\" BGCOLOR=\"#FFFFFF\">" << endl
    << "<META HTTP-EQUIV=\"Content-Type\" "
    << "CONTENT=\"text/html;charset=gb2312>\"" << endl
    << "<META NAME=\"GENERATOR\" CONTENT=\"Mozilla/4.04 [en] (X11; U; SunOS
5.6 sun4u) [Netscape]\">" << endl;
  cout <<  " <TITLE>RIME Demo (Version 1.0)</TITLE>" <<endl;

  cout << "<FONT SIZE=+2><B>RIME Demo (Version 1.0): Image Copy Detection</B></FONT><P>";
  cout << "Try our image-search engine on a half-million-image database!</FONT><P>";
  cout << "<font color=\"#CC0000\">&nbsp;<font size=+1>Since all the images in this
demo were crawled from the WWW, they may include some objectionable images.
If you are easily offended, please do not see this demo! </font></font><P>";


  // query image
  cout << "The input image ID is: " << imageID << endl;
  /*  sprintf(urlStr,
      "\"http://bluehead.stanford.edu:8080/images/\"", imageID);*/
  sprintf(urlStr,
      "\"http://bluehead.stanford.edu:8080/images/%s\"", queryImageFile);
  cout << "<img src=" << urlStr << " HEIGHT=150 WIDTH=100></a><br><P>" << endl;

  // returned images
  cout
    << "The top " << realImageNum
    << " returned images and their distances to the query image are:<P> "
    << endl;
  cout << "<TABLE border cellspacing=2 width=100>" << endl;

  for ( i = 0; i <= realImageNum - 1; i ++ )
  {
    if (imageNames[i] < 0)
      continue;
    // url
    sprintf(urlStr, "\"http://bluehead.stanford.edu:8080/images/%s\"", 
	    imageNames[i]);
    cout << "<TD VALIGN=top>"
	 << "<a href=\"http://bluehead.stanford.edu:8080/cgi-bin/cgirime1.0?" 
	 << imageArray[i] << "\">" << endl
	 << "<img src=" << urlStr << " HEIGHT=150 WIDTH=100></a><br>" << endl;

    // distance
    cout << imageDistance[i];

    if ( i % 5 == 4 && i != realImageNum - 1)
      cout << "<TR>" << endl;
    }

  cout << "</TABLE>" << endl;

  cout << "<FORM method=\"post\" "
       << "action=\"http://bluehead.stanford.edu:8080/cgi-bin/cgirime1.0\">" << endl
       << "<BR> Please click on one image or enter a new image ID"
       << "(0 - 447367) to start a new search:" << endl
       << "<input type=\"text\" name=\"imageID\" size=10 value=\"\">" << endl
       << "<BR>and the threshold of the comparison (0 - 10):" << endl
       << "<input type=\"text\" name=\"threshold\" size=10 value=\"" 
       << threshold << "\">"
       << endl
       << "<INPUT type=\"Submit\"" 
       << "value=\"Submit\">.</FONT></I></FORM></TD>" << endl;
  
  cout << "</BODY>" << endl
       << "</HTML>" << endl;
}


//------------------------------------
//   Parse the CGI string.
//------------------------------------
int
parseFormArgs(const char *FArgs)
{
  // first make a copy of the args
  if(!FArgs)
  {
    cerr << "Bad form argument : " << FArgs << endl;
    return STR_BAD;
  }
  
  char *FormArgs = new char[strlen(FArgs) +1];
  strcpy(FormArgs, FArgs);
  
  // first find the number of arguments (separated by "&")
  int i;
  char *CurrentPtr = FormArgs;
  
  for (i = 0; ; i++) {
    if(!CurrentPtr) {
      cerr << "Bad form input" << endl;
      return STR_BAD;
    }
    CurrentPtr = strchr(CurrentPtr, '&');
    if (CurrentPtr == NULL)
      break;
    CurrentPtr++;
  } // for
  
  numArgs = i+1;
  argArray = new struct ArgStruct[numArgs];
  // now actually read each argument
  CurrentPtr = FormArgs;
  char *Arg = strtok(CurrentPtr, "&");
  if (!Arg) { 
    cerr << "No arguments!" << endl;
   return STR_BAD;
  }
  
  if (argArray[0].AssignValue(Arg) != STR_GOOD) {
    cerr << "Error assigning value 0" << endl;
    return STR_BAD;
  }
  for (i = 1; i < numArgs; i++) {
    char *Arg = strtok(NULL, "&");
    if (argArray[i].AssignValue(Arg) != STR_GOOD) {
      cerr << "Error assigning value " << i << endl;
      return STR_BAD;
    }
  }
  delete [] FormArgs;
  return STR_GOOD;
}

void
GetImages(int sockfd)
{
  int  n;

  n = sizeof(imageID);
  if ( write(sockfd, &imageID, n) != n)
  {
    printf("client: GetImages() write error.\n");
    exit(1);
  }

  n = sizeof(threshold);
  if ( write(sockfd, &threshold, n) != n )
  {
    printf("client: GetImages() write error.\n");
    exit(1);
  }

  // get query image file name from the server
  n = read(sockfd, queryImageFile, MAXLINE);
  if ( n < 0 )
    {
      printf("client: GetImages() read error.\n");
      exit(1);
    }

  // get the number of returned images
  n = read(sockfd, &numReturnedImages, sizeof(numReturnedImages));
  if ( n < 0 )
  {
    printf("client: GetImages() read error.\n");
    exit(1);
  }
  if ( numReturnedImages > TOPK ) // make sure it's not too big
    numReturnedImages = TOPK;

  // get the returned images, including the IDs and file names
  for ( int i = 0; i < numReturnedImages; i ++ )
    {
      // read the image ID
      n = read(sockfd, &(imageArray[i]), sizeof(imageArray[i]));
      if ( n < 0 )
	{
	  printf("client: GetImages() read error.\n");
	  exit(1);
	}

      // read the image file name
      n = read(sockfd, imageNames[i], MAXLINE);
      if ( n < 0 )
	{
	  printf("client: GetImages() read error.\n");
	  exit(1);
	}

      // read the image distance to the query image
      n = read(sockfd, imageDistance[i], MAXLINE);
      if ( n < 0 )
	{
	  printf("client: GetImages() read error.\n");
	  exit(1);
	}
    }
}

bool
CommToServer()
{
  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sin_family  = AF_INET;
  serv_addr.sin_port    = htons(SERV_TCP_PORT);
  hp                    = gethostbyname( SERV_HOST_NAME );
  bcopy ( hp -> h_addr_list[0], &serv_addr.sin_addr.s_addr,
	  hp -> h_length );

  if (( sockfd = socket(AF_INET, SOCK_STREAM, 0 )) < 0 )
  {
    printf("client: socket error.\n");
    return ( false );
  }

  if (connect(sockfd, (struct sockaddr *)&serv_addr,
	      sizeof(serv_addr)) < 0 )
  {
    printf("client: connect error.\n");
    return ( false );
  }

  GetImages(sockfd);
  close(sockfd);
  return ( true );
}

void WriteSorryHTML()
{
  cout
    << "Content-type: text/html\n\n"
    << "<HEAD>" <<endl
    << "<META HTTP-EQUIV=\"Content-Type\" "
    << "CONTENT=\"text/html;charset=gb2312>\"" << endl
    << "<META NAME=\"GENERATOR\" CONTENT=\"Mozilla/4.04 [en] (X11; U; SunOS
5.6 sun4u) [Netscape]\">" << endl
    <<  " <TITLE>Sorry</TITLE>" <<endl
    << "<FONT SIZE=+2>Sorry: the server is not available or "
    << " the input is not correct.</FONT></FONT><P>" << endl;

  cout
    <<"</BODY>"<<endl
    << "</HTML>" << endl;
}
