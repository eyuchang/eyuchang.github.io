
#ifndef _UTIL_H_
#define _UTIL_H_

#define HTML_FILENAME "mybone.html"
#define STR_GOOD ( 1 )
#define STR_BAD  ( 0 )

class ArgStruct {
public:
  char *ArgName;   // name of the field
  char *ArgValue;  // value of the field
  ArgStruct() { ArgName = NULL; ArgValue = NULL;}
  ~ArgStruct() {
    if (ArgName) delete [] ArgName;
    if (ArgValue) delete [] ArgValue;
  }
  int AssignValue(const char *FormArg);
  void decipher(char *FormArg);
  char hex2char(char *FormArg);
};

bool FindFormArg(const char *ArgName,
		 const char *& ArgValue);

void SendEmail(const char* Recipient, 
	       char* Message, char* DocumentName);

extern ArgStruct *argArray; 
extern int       numArgs;

#endif


