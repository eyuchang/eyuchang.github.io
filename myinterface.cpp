#include "stdafx.h"
#include "..\MPEG2Decoder\status.h"
#include "inbuffer.h"
#include "RenderDisplayClass.h"
#include "debug.h"

#include "mpeg2ts.h"

#include "MPG2Wrap.h"
#include "dtv_mpeg.h"
#include "yuvtorgb.h"

#define PAGE_SIZE 4048
#define HEAP_SLOTS 1000		// 1,000 slots each 4,096 bytes (intel default)
#define HEAP_READ NUM_BUFFERS
#define HEAP_WRITE 200
#define HEAP_SIZE PAGE_SIZE * HEAP_SLOTS

BOOL _gHEAPValid[HEAP_SLOTS];
unsigned char *_gpHEAP[HEAP_SLOTS];
unsigned char *_gpHEAPHead;
unsigned char *_gpHEAPTail;
unsigned char *_gpHEAPNextByteWrite;
int _gCurrent;
int _gHEAPFirstValidPage;
int _gHEAPLastValidPage;
int _gHEAPNextPageToCopy;
int _gHEAPTotalValidPages;
int _gHEAPTotalEmptyPages;
int _gHEAPWrite;
int _gHEAPRead;

HANDLE _gEventWrite;
HANDLE _gEventRead;

// Critical Section!!!

CRITICAL_SECTION _gMemoryPointers;

// Define FIVE global variables

int _gDirectDrawDisplay = true;
int _gReturnDecodedFrame = false;

// dimension and data of decoded video
int _gVideoWidth;
int _gVideoHeight;
unsigned char *_gVideoData = 0;
int _gGO;
BOOL _gPause = FALSE;
BOOL _gReachedEndOfFile = FALSE;

// Data passing between threads
Mpeg2Transport *_gVideoConnection;
mpeg2tshandle videobufhandles[NUM_BUFFERS];
			
ULONG muMaxNumInputBuffers = NUM_BUFFERS;
ULONG muNextVideoBufferOffsetToWrite = 0;		

ULONG muFirstEmptyBufferOffset = 0;		
ULONG muLastEmptyBufferOffset = NUM_BUFFERS;
ULONG muNumEmptyBuffers = 0;

ULONG muFirstValidVideoBufferOffset = 0;		
ULONG muNumberValidBuffers = 0;
ULONG muBuffersFilled = 0;

RenderDisplay * mRender;
RECT g_rSrc;

typedef void (__cdecl *MyCallBack2)();
MyCallBack2 gCallBack;
void HackForNow()
{
	gCallBack();
}

__declspec( dllexport ) void Dtv_Mpeg_UpdateOverlay(int left, int top, int width, int height)
{
	RECT g_rDest;

	g_rDest.left = left;
	g_rDest.top = top;
	g_rDest.right = left + width;
	g_rDest.bottom = top + height;

	mRender->DDUpdateOverlay(g_rSrc, g_rDest, false);
}

_declspec( dllexport ) UINT Dtv_Reader(LPVOID pParam)
{

	HANDLE rFile = CreateFile("C:\\edcodec\\Ch1", GENERIC_READ | GENERIC_WRITE,
		FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

	DWORD dwBytesWritten, tmp;
	int size, size1;
	int index;

	while (TRUE) {

		WaitForSingleObject(_gEventRead, INFINITE);

		if (_gHEAPLastValidPage >= HEAP_WRITE) {
			size = HEAP_WRITE * PAGE_SIZE;
			index = _gHEAPLastValidPage - HEAP_WRITE + 1;
			ReadFile(rFile, _gpHEAP[index], size, &dwBytesWritten, NULL);
		} else {
			index = _gHEAPLastValidPage - HEAP_WRITE + HEAP_SLOTS + 1;
			size = (HEAP_SLOTS - _gHEAPLastValidPage + 1) * PAGE_SIZE;
			ReadFile(rFile, _gpHEAP[index], size1, &tmp, NULL);
			size = (HEAP_WRITE * PAGE_SIZE) - size; 
			ReadFile(rFile, _gpHEAP[0], size, &dwBytesWritten, NULL);
			dwBytesWritten += tmp;
		}

		if (dwBytesWritten == HEAP_WRITE * PAGE_SIZE) {
			EnterCriticalSection(&_gMemoryPointers);
			_gHEAPLastValidPage = index;
			_gHEAPTotalValidPages -= HEAP_WRITE;
			LeaveCriticalSection(&_gMemoryPointers);
		} else {
			break; // fatal IO error
		}
	}
	return 0;
}

_declspec( dllexport ) UINT Dtv_Writer(LPVOID pParam)
{
	HANDLE wFile = CreateFile("C:\\edcodec\\Ch1", GENERIC_READ | GENERIC_WRITE,
		FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

	DWORD dwBytesWritten, tmp;
	int size, size1;
	int index;

	while (TRUE) {

		WaitForSingleObject(_gEventWrite, INFINITE);

		if (_gHEAPLastValidPage >= HEAP_WRITE) {
			size = HEAP_WRITE * PAGE_SIZE;
			index = _gHEAPLastValidPage - HEAP_WRITE + 1;
			WriteFile(wFile, _gpHEAP[index], size, &dwBytesWritten, NULL);
		} else {
			index = _gHEAPLastValidPage - HEAP_WRITE + HEAP_SLOTS + 1;
			size = (HEAP_SLOTS - _gHEAPLastValidPage + 1) * PAGE_SIZE;
			WriteFile(wFile, _gpHEAP[index], size1, &tmp, NULL);
			size = (HEAP_WRITE * PAGE_SIZE) - size; 
			WriteFile(wFile, _gpHEAP[0], size, &dwBytesWritten, NULL);
			dwBytesWritten += tmp;
		}

		if (dwBytesWritten == HEAP_WRITE * PAGE_SIZE) {
			EnterCriticalSection(&_gMemoryPointers);
			_gHEAPLastValidPage = index;
			_gHEAPTotalValidPages -= HEAP_WRITE;
			LeaveCriticalSection(&_gMemoryPointers);
		} else {
			break; // fatal IO error
		}
	}
	return 0;


}


__declspec( dllexport ) UINT Dtv_NewReceiver(LPVOID pParam)
{

	// Read video transport stream until eof 

    while (!_gReachedEndOfFile) {
		
		EnterCriticalSection(&_gMemoryPointers);
		while (_gHEAPTotalValidPages >= HEAP_SLOTS - 2) {
			LeaveCriticalSection(&_gMemoryPointers);
			TRACE("Receive Stuck\n");
			Sleep(10);
			EnterCriticalSection(&_gMemoryPointers);
		}
//		LeaveCriticalSection(&_gMemoryPointers);

		TRACE("NewReceiver Proceeds\n");
		int myresult;
		mpeg2tsbuf buf;

		myresult = _gVideoConnection->GetData(&buf,5);

		if (myresult == 1) {
			// buffer fetch succeeded, put pointer to it in decoder's buffer array
			// store handle so we can return this buffer later

			int remainBytes = _gpHEAPTail - _gpHEAPNextByteWrite + 1;

			if (buf.datasize <= remainBytes) {
				memcpy((char *)_gpHEAPNextByteWrite, (const char *) buf.data, buf.datasize);
			}
			else {
				memcpy((char *)_gpHEAPNextByteWrite, (const char *) buf.data, remainBytes);
				unsigned char * pos = buf.data;
				pos += remainBytes;
				int left = buf.datasize - remainBytes;
				memcpy((char *)_gpHEAPHead, pos, left);
				TRACE("BOUNDARU WRITE: Remaining bytes %d 1st write = %d bytes left = %d\n", 
					buf.datasize, remainBytes, left);
			}
			_gpHEAPNextByteWrite = _gpHEAPNextByteWrite + buf.datasize;
			if (_gpHEAPNextByteWrite > _gpHEAPTail) 
				_gpHEAPNextByteWrite -= HEAP_SIZE;
			// Update pointers

			_gVideoConnection->ReturnBuf(buf.bufid);
//			EnterCriticalSection(&_gMemoryPointers);
			// Last valid page
			_gHEAPLastValidPage = (_gpHEAPNextByteWrite - _gpHEAPHead) / PAGE_SIZE;
			_gHEAPTotalValidPages = _gHEAPLastValidPage - _gHEAPFirstValidPage + 1;
			if (_gHEAPTotalValidPages < 0)
				_gHEAPTotalValidPages += HEAP_SLOTS;
			TRACE("NewReceiver First, Last, TotalValid %d %d %d %d\n", buf.datasize, 
				_gHEAPFirstValidPage, _gHEAPLastValidPage, _gHEAPTotalValidPages);
			_gHEAPTotalEmptyPages = HEAP_SLOTS - _gHEAPTotalValidPages -1; // can be off by one, but ok
			LeaveCriticalSection(&_gMemoryPointers);

			// Trigger a write if needed

			if (_gHEAPTotalEmptyPages < HEAP_WRITE) {
				SetEvent(_gEventWrite);
			}


		}
		// error on fetch, which means either 1.) end of file on file-based
		// runs of this code, or 2.) mpeg2ts class had problems parsing buffers
		// which probably means data is corrupted; or 3.) something's really
		// wrong in driver and everything above it failed too; need to expand
		// these different errors beyond all = -1; for now, we assume it's 
		// either error #1 or #3 and keep on trucking
		else if (myresult == -1) {
			DEBUGMSG("Error or end of file reached on video buffer fetch");
			//if (!g_LiveTransportStream && (g_Datarate == 0)) {
			    _gReachedEndOfFile = TRUE;
				break;
			//}
		}
		// else buffers not available yet; flag underflow and then
		// sit and wait for buffer
		else if (myresult == 0) {
			DEBUGMSG("Video buffer fetch returned empty");
			Sleep(10); // Yield to other threads before resuming

			// if headend or driver stops delivering data, testdec will
			// get stuck here until new data comes in or stop is requested at GUI
			// while(1) {
			//	myresult = _gVideoConnection->GetData(&buf, 500);
			//	if (myresult)
			//		break;
			//} 
		}
	} // end of while

	// Read in data whenever they arrive and
	// o trigger a write event if needed
	// o set the playback condition if needed
    return TRUE;
}

__declspec( dllexport ) UINT Dtv_Receiver(LPVOID pParam)
{

	// Open video transport stream connection
	// open video transport stream connection 

	_gGO = 0;
	while (TRUE) {
	EnterCriticalSection(&_gMemoryPointers);

	while (_gGO != 0) {
		LeaveCriticalSection(&_gMemoryPointers);
		Sleep(1);
		EnterCriticalSection(&_gMemoryPointers);
	}

	muBuffersFilled = 0;
	muFirstValidVideoBufferOffset = muNextVideoBufferOffsetToWrite;
	//while (TRUE) {
	while (muNumberValidBuffers < NUM_BUFFERS - 2) {
		int myresult;
		mpeg2tsbuf buf;

		myresult = _gVideoConnection->GetData(&buf,5);

		if (myresult == 1) {
			videobufhandles[muNextVideoBufferOffsetToWrite]=buf.bufid;
							
			// setup pointer and length in buffer structure
			DataInVideoArray[muNextVideoBufferOffsetToWrite].pVideoPacket = buf.data;
			DataInVideoArray[muNextVideoBufferOffsetToWrite].cPacketSz = buf.datasize;
				
			// increment buffer offset and wrap around if reached the end						
			muNextVideoBufferOffsetToWrite++;
			muNextVideoBufferOffsetToWrite = (muNextVideoBufferOffsetToWrite < NUM_BUFFERS) ? muNextVideoBufferOffsetToWrite : 0;
			muBuffersFilled++;	// number of buffers filled this time around
			muNumberValidBuffers++;	// another valid buffer
			TRACE("Thread: DTV_Receiver: Number of valid buffers = %d data read = %d bufid = %d\n", 
				muNumberValidBuffers, buf.datasize, buf.bufid);
			// just to test if this works
			// XXX it does not make a difference. why? i suspect it assigns ids 
			// in the following way
			//_gVideoConnection->ReturnBuf(buf.bufid);

		}
		// error on fetch, which means either 1.) end of file on file-based
		// runs of this code, or 2.) mpeg2ts class had problems parsing buffers
		// which probably means data is corrupted; or 3.) something's really
		// wrong in driver and everything above it failed too; need to expand
		// these different errors beyond all = -1; for now, we assume it's 
		// either error #1 or #3 and keep on trucking
		else if (myresult == -1) {
			DEBUGMSG("Error or end of file reached on video buffer fetch");
			//if (!g_LiveTransportStream && (g_Datarate == 0)) {
			    _gReachedEndOfFile = TRUE;
				break;
			//}
		}
		// else buffers not available yet; flag underflow and then
		// sit and wait for buffer
		else if (myresult == 0) {
			DEBUGMSG("Video buffer fetch returned empty");

			// if headend or driver stops delivering data, testdec will
			// get stuck here until new data comes in or stop is requested at GUI
			while(1) {
				myresult = _gVideoConnection->GetData(&buf, 500);
				if (myresult)
					break;
			} 
		}

	} // inner while
	_gGO = 1;
	LeaveCriticalSection(&_gMemoryPointers);
	} // outer while

	// Read in data whenever they arrive and
	// o trigger a write event if needed
	// o set the playback condition if needed
    return TRUE;
}
__declspec( dllexport ) UINT Dtv_Reader()
{
	// Init: open file



	// Wait for a read event

	// Read delta_read

	// Update pointers
	return TRUE;
}

__declspec( dllexport ) UINT Dtv_Writer()
{
	// Init: open file

	// Wait for a write event

	// Write delta_write

	// Update pointers
	return TRUE;
}

__declspec( dllexport ) UINT Dtv_Mpeg_Decode(LPVOID pParam)
{
	
	// Init the decoder, wait for events before start
	// Decode in a loop and update variables

	DecoderParam * param = (DecoderParam *) pParam;
	HANDLE hThreads[3];
	MPEG2CONTROLSETTINGS	IntelMPEG2Settings		= {PLAY_IPB_ONLY, FALSE};
	IntelMPEG2Settings.dwPlayBackLevel = param->PlayBackLevel;
	IntelMPEG2Settings.bFrameDropDisabled = !param->FrameDropEnable;

	_gDirectDrawDisplay = param->DirectDrawDisplay;
	_gReturnDecodedFrame = param->ReturnDecodedFrame;
	_gVideoWidth	= param->VideoWidth;
	_gVideoHeight	= param->VideoHeight;

	// HEAP alloc and test if fine
	// XXX Init Global Variables

	_gpHEAPHead = (unsigned char *) malloc (HEAP_SIZE);
	_gpHEAP[0] = _gpHEAPHead;
	for (int i = 1; i < HEAP_SLOTS; i++) {
		_gpHEAP[i] = _gpHEAP[i-1] + PAGE_SIZE;
		//TRACE("Mem address %d\n", _gpHEAP[i]);
	}
	_gpHEAPTail = _gpHEAPHead + HEAP_SIZE -1;
	_gpHEAPNextByteWrite = _gpHEAPHead;
	_gHEAPFirstValidPage = _gHEAPLastValidPage = _gHEAPTotalValidPages = 0;
	_gHEAPNextPageToCopy = 0;
	
	muNextVideoBufferOffsetToWrite = 0;
	muNumberValidBuffers = 0;

	// Init all thread synchronization objects

	InitializeCriticalSection(&_gMemoryPointers);

	if(_gReturnDecodedFrame)
		_gVideoData		= (unsigned char *) malloc(_gVideoWidth * _gVideoHeight * 3);

	MyCallBack CallBackOnFrameDone = param->callBack;

	// display decoded frame or not
	bool    displayVideo				=	true;

	// decode into system or DirectDraw memory
	bool	indirectWrite			=	false;

	// Output frame used for debug mode
	unsigned char * OutFrameDebug = 0; //(unsigned char *) malloc(_gVideoWidth * _gVideoHeight * 3);


	LONG lFrameToUse;	// frame tracking and reordering

	VIDEOQUALITY sVideoQuality;
	sVideoQuality.lTimeInms = 0;

    RESULT Result;
	HRESULT DDResult; 

    DWORD dwMPEGStreamType      = 1;
    DWORD dwHorizontalSize      = 0;
    DWORD dwVerticalSize        = 0;
	DWORD dwActualHorizontalSize= 0;
	DWORD dwActualVerticalSize	= 0;
	DWORD dwIFrames				= 0;
	DWORD dwPFrames				= 0;
	DWORD dwBFrames				= 0;
    DWORD dwFrameRate           = 0;
	FLOAT fTargetFrameRate		= 0.0f;
	FLOAT fAverageFrameRate		= 0.0f;
	DWORD dwFourCC = 0;
	BOOL bFrameLocked = FALSE;	// set when a DD surface is locked.  Use for exception handling
	RENDERSURFACEDESC rsdVideoMem;
	RENDERSURFACEDESC rsdMPEGout;
	RECT sRectStore;

	// flag to indicate we're in restart condition
	BOOL DecoderIsInRestart = FALSE;

    // XXX Create DTV_Receiver thread

	_gVideoConnection = new Mpeg2Transport(param->Videopidno);
	if (!_gVideoConnection || !_gVideoConnection->IsConnected()) {
		AfxMessageBox("Failed to open video connection");
		return FALSE;
	}

	// look for video sequence header in incoming bitstream
	if (_gVideoConnection->GetVideoSequence((int*)&dwMPEGStreamType,(int*)&dwFrameRate,
		(int*)&dwActualHorizontalSize,(int*)&dwActualVerticalSize,15000) != 1) {
		AfxMessageBox("Could not find video sequence header in stream!");
		return FALSE;
	}

	_gEventRead  = CreateEvent(NULL, FALSE, FALSE, NULL);
	_gEventWrite = CreateEvent(NULL, FALSE, FALSE, NULL);

	//hThreads[0] = AfxBeginThread(Dtv_Receiver, HackForNow, THREAD_PRIORITY_HIGHEST);
	hThreads[0] = AfxBeginThread(Dtv_NewReceiver, HackForNow, THREAD_PRIORITY_HIGHEST);
	hThreads[1] = AfxBeginThread(Dtv_Reader, HackForNow, THREAD_PRIORITY_HIGHEST);
	hThreads[2] = AfxBeginThread(Dtv_Writer, HackForNow, THREAD_PRIORITY_HIGHEST);
	
    // Create an object that will do MPEG2 video decoding.
    DVDMPEG2Decoding * MPEG2Decoder = new DVDMPEG2Decoding;

    Result = MPEG2Decoder->MPEG2Create();
    if (FAILURE == Result) {
        AfxMessageBox("Unable to init MPEG2 decoder!!!");
        return (0);
	} else if (MEMORY_NOT_ALLOCATED == Result) {
        AfxMessageBox("Unable to allocate memory in MPEG2 decoder!!!");
        return (0);
	}

	// Update Settings to Decoder!!!
	MPEG2Decoder->MPEG2SetControlInfo(&(IntelMPEG2Settings));

	mRender = new RenderDisplay;
	mRender->Initialize(false, false);

	/*
	** Select FourCC
	*/
	if(_gDirectDrawDisplay) {
		DWORD NC = 10;	// Assume max 10 FourCC's
		DWORD Code[10];
		DWORD i;

		/* Get available FourCC's */
		mRender->GetFourCC(&NC, Code);

		/* Search in order of preference */
		if (dwFourCC==0)  for (i=0; i<NC; i++)  if (Code[i] == mmioFOURCC('C','L','P','L'))  dwFourCC = Code[i];
		if (dwFourCC==0)  for (i=0; i<NC; i++)  if (Code[i] == mmioFOURCC('Y','V','1','2'))  dwFourCC = Code[i];
		if (dwFourCC==0)  for (i=0; i<NC; i++)  if (Code[i] == mmioFOURCC('Y','U','Y','2'))  dwFourCC = Code[i];
		//	if (dwFourCC==0)  for (i=0; i<NC; i++)  if (Code[i] == mmioFOURCC('U','Y','V','Y'))  dwFourCC = Code[i];
		//	if (dwFourCC==0)  for (i=0; i<NC; i++)  if (Code[i] == mmioFOURCC('Y','V','Y','U'))  dwFourCC = Code[i];
	
		if (dwFourCC==0) {
			AfxMessageBox("Error: No supported video (FourCC) types");
			return(FALSE);
		}	
	} else {
		dwFourCC = mmioFOURCC('Y','V','1','2');
	}

	dwVerticalSize = dwActualVerticalSize;	

	/*
	** JAS/RWF
	** Force dwHorizontalSize (and pitch, via DDCreate()) to *32Bytes
	** so all MBlocks in all rows get write-combined performance
	*/

	switch (dwFourCC) {
		case mmioFOURCC('C','L','P','L'):	// All planes ->32B width, by setting Y to 704 or 768
		case mmioFOURCC('Y','V','1','2'):
		case mmioFOURCC('M','C','1','2'):
			DWORD temp;
			if ((temp = dwActualHorizontalSize % 64) != 0) {
				dwHorizontalSize = dwActualHorizontalSize + 64 - temp;
			} else
				dwHorizontalSize = dwActualHorizontalSize;
			break;

		case mmioFOURCC('Y','U','Y','2'):	// Pitch is naturally 32B-wide	(and WC not as important for indirect (sequential to cached mem) writes)
			dwHorizontalSize = dwActualHorizontalSize;
		break;

	}

	/*
	** Force IndirectWrite (and convert) as necessary
	*/
	switch (dwFourCC) {
		case mmioFOURCC('C','L','P','L'):
		case mmioFOURCC('Y','V','1','2'):
		case mmioFOURCC('M','C','1','2'):
			break;

		case mmioFOURCC('Y','U','Y','2'):
			MPEG2Decoder->SetDirectYUY2Mode((BOOL)(1 - indirectWrite));		// set YUY2's direct (or not) mode SRJ 10-29-97
		break;
	}

	sRectStore.left = 0;
	sRectStore.top = 0;
	sRectStore.right = dwHorizontalSize;
	sRectStore.bottom =  dwVerticalSize;

	// Create DirectDraw complex surface object
	DDResult = mRender->DDCreate(	"MPEG2 Output",		// title
									4,					// number of surfaces
														// Alert: must be >=4 or app locks up
									dwFourCC,			// RWF pixel format
									0,					// x display position
									0,					// y display position
									dwHorizontalSize,	// window width
									dwVerticalSize);	// window height
	if (DDResult != DD_OK) {
		AfxMessageBox(mRender->Err2Str(DDResult));
		AfxMessageBox("Direct Draw failed in creation phase.\nScreen rendering is disabled");
		displayVideo = FALSE;	// Turn off rendering
		return 0;
		// Bug Alert: 
		// Changing m_bDisplayVideo from FALSE to TRUE  while playing  often locks up the app
		// (Could add extra var to disable actual changing while playing)
	}

	// Size Controls
	// g_rSrc and g_rDest are defined in testDlg.h, but the object
	// does not know the encoded picture size. An enum is passed and dimensions
	// set here and when m_bSettingsChanged
	g_rSrc.left = g_rSrc.top = 1;
	g_rSrc.right = dwActualHorizontalSize;
	g_rSrc.bottom = dwActualVerticalSize;

	// this is a temporary hack for MC12 demo: The mcData pointer within rsdMPEGout
	// is assigned within the DDGet(), and I want to be able to force software motion
	// comp to take place on the MC12 surface. This can be resolved in many ways.
	// 1. add a boolean for useMotionComp, rather than the function pointer
	// 2. add a function to dvdDisplay to return the pointer, rather than in DDGet()
	// 3. leave the test/override within the loop. It's only a demo after all
	// 4.
	//
	// for now, rsdMPEGout is created after the DDCreate(). A bogus DDGet is called to
	// get the pointer, and a paired DDLet() is called.
	if (1) {
		HRESULT ddrval;

		if ( (ddrval = mRender->DDGet(0, &rsdMPEGout)) != DD_OK) {
			__asm int 3;
			AfxMessageBox(mRender->Err2Str(DDResult));
			return (0);
			}
		if ( (ddrval = mRender->DDLet()) != DD_OK) {
			__asm int 3;
			AfxMessageBox(mRender->Err2Str(DDResult));
			return (0);
			}

		// Restore buffers
		ddrval = mRender->InitBuffers();
	}

	// defeat hardware motion comp
	rsdMPEGout.mcData->HwMotionCompFunc = NULL;


    // Variables used for controlling input and output buffers

    DWORD dwPicturesDecoded     = 0;
	DWORD dwPicturesDropped		= 0;
    DWORD dwSequencesDecoded    = 0;
    DWORD dwGOPsDecoded         = 0;
    DWORD dwSlicesDecoded       = 0;
    DWORD dwMacroblocksDecoded  = 0;
    DWORD dwAverageBitRate      = 0;
    DWORD dwCurrentBitRate      = 0;
	DWORD dwMinBitRate          = 0x7fffffff;
    DWORD dwMaxBitRate          = 0;
	FLOAT fCurrentFrameRate = 0.0f;
	FLOAT fCurrentDropRate = 0.0f;
	DWORD dwInterlacedFrame     = 0;
	DWORD dwProgressiveFrame    = 0;
	float flSecondsDecoded		= 0.0f;

	ULONG uBufferConsumedStartOffset=0;
	ULONG uNumberBuffersConsumed=0;
	ULONG uStatus=0;

	LINE21DESC Line21Data[20];

	// Initialize overlay surface descriptor
	rsdMPEGout.dwFourCCCode = dwFourCC;	// RWF pass fourCC to decoder  // JAS Note: apparently this is not used
	rsdMPEGout.lpY = NULL;
	rsdMPEGout.lpU = NULL;
	rsdMPEGout.lpV = NULL;


	// increment buffer offset and wrap around if reached the end						
	//muNextVideoBufferOffsetToWrite++;
	//muNextVideoBufferOffsetToWrite = (muNextVideoBufferOffsetToWrite < NUM_BUFFERS) ? muNextVideoBufferOffsetToWrite : 0;
	//muBuffersFilled++;	// number of buffers filled this time around
	//muNumberValidBuffers++;	// another valid buffer

	//int Result;
	_gReachedEndOfFile = FALSE;

    while (!_gReachedEndOfFile) {

		// The logic on the while should work but is doubtful that it is optimum. GD\
		// so don't blindly copy this code into Active Movie.

		// if a picture was done last time around, decoder needs a new frame buffer
		if (uStatus == 0) {
			// Default MPEG output surface to system memory
			// (If direct-write, DDGet() overrides to video memory)
			rsdMPEGout.lpY	  = OutFrameDebug;
			rsdMPEGout.lpV	  = rsdMPEGout.lpY + dwHorizontalSize	* dwVerticalSize;	//choose offsets
			rsdMPEGout.lpU	  = rsdMPEGout.lpV + dwHorizontalSize/2 * dwVerticalSize/2;

			// chroma pitch remains pitchY/2 for indirect writes
			// debug writes may not work for MC12, but that's ok
			// because the direct locks of the MC12 surface don't
			// assert the WIN16 lock mutex
			rsdMPEGout.lPitchY = dwHorizontalSize;
			rsdMPEGout.lPitchU = dwHorizontalSize>>1;
			rsdMPEGout.lPitchV = dwHorizontalSize>>1;

			// don't call DDGet unless previous pass completed a picture
			// DDLet will not be called unless picture completes
	
			// If DirectWrite, get video mem surface
			if (!(indirectWrite) && (displayVideo)) {
				DDResult = mRender->DDGet(lFrameToUse, &rsdMPEGout);
				if (DDResult != DD_OK) {
					__asm int 3;	// break out from lock
					AfxMessageBox(mRender->Err2Str(DDResult));
				} else {
					bFrameLocked = TRUE;
				}
			}
		}  // if (uStatus == 0)

		EnterCriticalSection(&_gMemoryPointers);

		// We need to have the buffer filled!!!

		//TRACE("Number of valid buffers = %d\n", muNumberValidBuffers);

		while ((_gHEAPTotalValidPages < NUM_BUFFERS - 2) || (_gPause)) {
			LeaveCriticalSection(&_gMemoryPointers);
			TRACE("Decode stuck with TotalValid = %d\n", _gHEAPTotalValidPages);
			Sleep(10); // Give up the thread
			EnterCriticalSection(&_gMemoryPointers);
		}

		muBuffersFilled = 0;
		muFirstValidVideoBufferOffset = muNextVideoBufferOffsetToWrite;
		
		while (muNumberValidBuffers < NUM_BUFFERS) {
			// Install an condition before start playback
			//TRACE("Array Bounds: NextToWrite %d from Index %d Addr %d\n", 
			//	muNextVideoBufferOffsetToWrite, _gHEAPNextPageToCopy, _gpHEAP[_gHEAPNextPageToCopy]);
			DataInVideoArray[muNextVideoBufferOffsetToWrite].cPacketSz = PAGE_SIZE;
			DataInVideoArray[muNextVideoBufferOffsetToWrite].pVideoPacket = _gpHEAP[_gHEAPNextPageToCopy];
			_gHEAPNextPageToCopy = (_gHEAPNextPageToCopy + 1) % HEAP_SLOTS;
			muNextVideoBufferOffsetToWrite++;
			muNextVideoBufferOffsetToWrite = (muNextVideoBufferOffsetToWrite < NUM_BUFFERS) ? muNextVideoBufferOffsetToWrite : 0;
			muBuffersFilled++;	// number of buffers filled this time around
			muNumberValidBuffers++;	// another valid buffer
		}

		// ugly but necessary until decoder no longer crashes on bad data
		try {
			// Pass the video packet to the decoder.
			uNumberBuffersConsumed = 0;
			Result = MPEG2Decoder->MPEG2DecodeVideoData(
						&muFirstValidVideoBufferOffset,	//uNewBufferOffset,
						&muBuffersFilled,				//uNumberNewValidBuffers,
						&uBufferConsumedStartOffset,
						&uNumberBuffersConsumed,
						&uStatus,
						&rsdMPEGout,
						Line21Data);
		}
		catch (...) {
			//	AfxMessageBox("Access violation exception caught, exiting!!!\n");
			//	bReachedEndOfFile = TRUE;
			uStatus = MPEG2_PICTURE_DATA_ERROR;
		}

		// return used buffers to driver
		//ULONG x=uBufferConsumedStartOffset,y; 
		//for(y=0;y<uNumberBuffersConsumed;y++) {
		//	_gVideoConnection->ReturnBuf(videobufhandles[x]);
		//	x = (x+1)%NUM_BUFFERS;
		//}

		// release framebuffer if decoding was successful and picture is complete
		if (bFrameLocked && Result==NOERROR && uStatus == MPEG2_PICTURE_DONE) {
			DDResult = mRender->DDLet();
			if (DDResult != DD_OK) {
				__asm int 3;	// break out from lock
				AfxMessageBox(mRender->Err2Str(DDResult));
			} else {
				bFrameLocked = FALSE;
			}
		}

		// update buffer handling control parameters

		muNumberValidBuffers -= uNumberBuffersConsumed;
		_gHEAPFirstValidPage = (_gHEAPFirstValidPage + uNumberBuffersConsumed) % HEAP_SLOTS;
		_gHEAPTotalValidPages = _gHEAPLastValidPage - _gHEAPFirstValidPage + 1;
		if (_gHEAPTotalValidPages < 0)
			_gHEAPTotalValidPages += HEAP_SLOTS;
		_gGO = 0;

		// begin SAR restart code
		// if decoder has restarted from error, force a new set of control changes
		if (Result==NOERROR && uStatus==MPEG2_PICTURE_DONE && DecoderIsInRestart) {
			DecoderIsInRestart=0;
		}
		// else decoder has just hit an error -- flush buffers & restart
		else if ((Result != NOERROR) || 
  				 (uStatus!=MPEG2_PICTURE_PENDING && uStatus!=MPEG2_PICTURE_DONE)) {
			//__asm int 3;
			DWORD InterruptTime = timeGetTime();
			DecoderIsInRestart=1;

			// reset decoder 
			if (SUCCESS != MPEG2Decoder->MPEG2Reset()) {
				AfxMessageBox("Unable to reset MPEG2 decoder!!!");
				break;
			}
			MPEG2Decoder->MPEG2SetDecoderRestart();
			uStatus = MPEG2_PICTURE_PENDING;

			// return any buffers that are in testdec 
			_gVideoConnection->ReturnAllBuffers();
			muFirstEmptyBufferOffset = 0;		
			muLastEmptyBufferOffset = NUM_BUFFERS;
			muNextVideoBufferOffsetToWrite = 0;		
			muNumberValidBuffers = 0;

			continue;
		}

		LeaveCriticalSection(&_gMemoryPointers);
		
		// end SAR restart code

		if (MPEG2_PICTURE_PENDING == uStatus) {
			// Nothing to do until the decoder completes a picture.
        } else if (MPEG2_PICTURE_DONE == uStatus) {
			// Pass completed frame to renderer
			// If IndirectWrite, first Copy/Convert to video mem
			if (displayVideo) {	
				if (indirectWrite) {
					// Get pointer to video mem surface
					DDResult = mRender->DDGet(lFrameToUse, &rsdVideoMem);
					if (DDResult != DD_OK) {
						__asm int 3;	// this will kick us out
						AfxMessageBox(mRender->Err2Str(DDResult));
					} else {
						bFrameLocked = TRUE;
					}

					// Copy (and convert) data from system memory to video memory
					switch (dwFourCC) {
					case mmioFOURCC('C','L','P','L'): 
					case mmioFOURCC('Y','V','1','2'):
						CopyFrame420(	&rsdMPEGout,
							dwActualHorizontalSize,
							dwVerticalSize,		
							&rsdVideoMem);
						break;
					case mmioFOURCC('Y','U','Y','2'):
						CopyFrameYUY2(	&rsdMPEGout,
							dwActualHorizontalSize,	
							dwVerticalSize,	
							&rsdVideoMem);
						break;
					}

								
					// Unlock the current pointer to overlay surface(n)
					DDResult = mRender->DDLet();
					if (DDResult != DD_OK) {
						AfxMessageBox(mRender->Err2Str(DDResult));
					} else {
						bFrameLocked = FALSE;
					}
				}	// if (IndirectWrite)
						
				MPEG2Decoder->Quality(sVideoQuality);
				sVideoQuality.lTimeInms = 0;

				// lFrameToUse is not known until during motion comp
				// uFrameNumber is set by dvdDisplay on unlock
				// this will work for MC12 with software motion comp
				// for other surfaces, uFrameNumber will be set on the
				// lock, and will persist to this time.
				// 
				// if the uFrameNumber replaces frame tracking for surfaces
				// other than MC12, then it can move from mcData to the
				// renderSurfaceDesc, EXCEPT that it is currently set inside
				// of the DDLet, which does not have access to the renderSurf...
				// This is a solvable problem.
				lFrameToUse = rsdMPEGout.mcData->uFrameNumber;

				// GetFrameBufferToFill now called after frame decoding is complete.
				// A better name would be SetFrameBufferToDisplay, as the lFrameToUse
				// parameter now contains the index of the buffer just decoded.
				// Note: it no longer needs to be a pointer, but is left so now
				//       to avoid changing the prototype

				mRender->PassFrameToRenderer(lFrameToUse, 0, sRectStore, rsdMPEGout, &sVideoQuality.lTimeInms);
				CallBackOnFrameDone(_gVideoData);
			}	// if(Displaying)

			MPEG2Decoder->MPEG2GetStatistics(
						&dwMPEGStreamType,
						&dwFrameRate,
						&fTargetFrameRate,
						&flSecondsDecoded,
						&fAverageFrameRate,
						&dwAverageBitRate,
						&dwCurrentBitRate,
						&dwMinBitRate,
						&dwMaxBitRate,
						&fCurrentFrameRate,
						&dwInterlacedFrame,
						&dwProgressiveFrame,
						&dwActualHorizontalSize,
						&dwActualVerticalSize,
						&dwIFrames,
						&dwPFrames,
						&dwBFrames,
                        &dwPicturesDecoded,
						&dwPicturesDropped,
                        &dwSequencesDecoded,
                        &dwGOPsDecoded,
                        &dwSlicesDecoded,
                        &dwMacroblocksDecoded);
			} else if (Result == MPEG2_PICTURE_RESIZE) {
			AfxMessageBox("Picture Resize in stream!!!");
			break;
			} else if (Result == MPEG2_NO_VALID_OVERLAY) {
            AfxMessageBox("Tried to decode video data with no free frame buffer!!!");
            break;
        } else if (Result == MPEG2_INPUT_BUFFER_OVERRUN) {
            AfxMessageBox("Tried to decode a block of video data with too few frame buffers!!!");
            break;
		} else if (Result == MPEG2_SCALABLE_LAYER) {
			AfxMessageBox("Encountered a scalable layer.  Scalable Layers are not handled!!!");
            break;
        } else if (Result == MPEG2_MACROBLOCK_DECODE_ERROR) {
            AfxMessageBox("Unable to process a macroblock!!!");
            break;
		} else if (END_OF_FILE_NO_DATA == Result) {
			AfxMessageBox("Unable to get video packet from stream splitter!!!");
			_gReachedEndOfFile = TRUE;
		} else if (INVALID_MPEG1_DATA_STREAM == Result) {
			AfxMessageBox("Encountered an erroneous MPEG1 system stream!!!");
			break;
		} else {
			AfxMessageBox("Error code not coded in TestDlg.cpp!!!");
			break;
		}
	} // end of while

	if (SUCCESS != MPEG2Decoder->MPEG2Delete())
	{
        AfxMessageBox("Unable to uninit MPEG2 decoder!!!");
	}

    delete MPEG2Decoder;
	delete mRender;
	delete _gVideoConnection;

	DTVClose();

    return (0);
}
