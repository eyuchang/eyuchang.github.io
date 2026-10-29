#include <stdio.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>

#include "util.h"

int
ArgStruct::AssignValue(const char *FormArg) 
{
  if(!FormArg)
  {
    cerr << "Bad form argument : " << FormArg << endl;
    return STR_BAD;
  }
  char *ValuePtr = strchr(FormArg, '=');
  if (!ValuePtr) 
  {
    cerr << "Bad form argument : " << FormArg << endl;
    return STR_BAD;
  }
  ArgName = new char[ValuePtr - FormArg + 1];
  strncpy(ArgName, FormArg, ValuePtr - FormArg);
  ArgName[ValuePtr - FormArg] = 0;
  
  ArgValue = new char[strlen(ValuePtr+1) + 1];
  strncpy(ArgValue, ValuePtr+1, strlen(ValuePtr+1));
  ArgValue[strlen(ValuePtr+1)] = 0;
  decipher(ArgValue);
 return STR_GOOD;
}


void 
ArgStruct::decipher(char *url) 
{
  register int i,j,k,z;
  
  /** Change all plusses back to spaces **/
  for(k=0; url[k]; k++){
    if(url[k] == '+') url[k] = ' ' ;
  }
  for(i=0,j=0; url[j]; ++i,++j) {
    if((url[i] = url[j]) == '%') {
      url[i] = hex2char(&url[j+1]) ;
      j+= 2 ;
    }
  }
  /** Change all double quotes to single quotes since they symbolize an end of a string to PREDATOR**/
  for(z=0; url[z]; z++){
    if(url[z] == '\"') url[z] = '\'' ;
  }
  url[i] = '\0' ;
}



char 
ArgStruct::hex2char(char *what) 
{
  register char digit;
  
  digit = (what[0] >= 'A' ? ((what[0] & 0xdf) - 'A')+10 : (what[0] - '0'));
  digit *= 16;
  digit += (what[1] >= 'A' ? ((what[1] & 0xdf) - 'A')+10 : (what[1] - '0'));
  return(digit);
}



void
SendEmail(const char* Recipient, char* Message, char* DocumentName)
{
  char command[1000];
  sprintf(command, "/usr/ucb/mail -s \"%s\" %s < %s ", 
	  Message, Recipient, DocumentName);
  if (system(command)) 
  {
    cerr << "Error in the emailing functionality" << endl;
  }
}


bool
FindFormArg(const char *ArgName, const char *& ArgValue)
{
  for (int i = 0; i < numArgs; i++) {
    if (!strcmp(argArray[i].ArgName, ArgName)) 
    {
      ArgValue = argArray[i].ArgValue;
      return true;
    }
  }
  return false;
}
