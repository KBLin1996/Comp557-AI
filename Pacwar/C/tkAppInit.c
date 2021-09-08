/* 
 * tkAppInit.c --
 *
 *	Provides a default version of the Tcl_AppInit procedure for
 *	use in wish and similar Tk-based applications.
 *
 * Copyright (c) 1993 The Regents of the University of California.
 * Copyright (c) 1994 Sun Microsystems, Inc.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 */

#include <ctype.h>
#include "PacWar.h"
#include "tk.h"

#include "bitmaps/blob16.pbm"
#include "bitmaps/pac16_000.pbm"
#include "bitmaps/pac16_001.pbm"
#include "bitmaps/pac16_002.pbm"
#include "bitmaps/pac16_003.pbm"
#include "bitmaps/pac16_010.pbm"
#include "bitmaps/pac16_011.pbm"
#include "bitmaps/pac16_012.pbm"
#include "bitmaps/pac16_013.pbm"
#include "bitmaps/pac16_020.pbm"
#include "bitmaps/pac16_021.pbm"
#include "bitmaps/pac16_022.pbm"
#include "bitmaps/pac16_023.pbm"
#include "bitmaps/pac16_030.pbm"
#include "bitmaps/pac16_031.pbm"
#include "bitmaps/pac16_032.pbm"
#include "bitmaps/pac16_033.pbm"
#include "bitmaps/pac16_100.pbm"
#include "bitmaps/pac16_101.pbm"
#include "bitmaps/pac16_102.pbm"
#include "bitmaps/pac16_103.pbm"
#include "bitmaps/pac16_110.pbm"
#include "bitmaps/pac16_111.pbm"
#include "bitmaps/pac16_112.pbm"
#include "bitmaps/pac16_113.pbm"
#include "bitmaps/pac16_120.pbm"
#include "bitmaps/pac16_121.pbm"
#include "bitmaps/pac16_122.pbm"
#include "bitmaps/pac16_123.pbm"
#include "bitmaps/pac16_130.pbm"
#include "bitmaps/pac16_131.pbm"
#include "bitmaps/pac16_132.pbm"
#include "bitmaps/pac16_133.pbm"
#include "bitmaps/blob32.pbm"
#include "bitmaps/pac32_000.pbm"
#include "bitmaps/pac32_001.pbm"
#include "bitmaps/pac32_002.pbm"
#include "bitmaps/pac32_003.pbm"
#include "bitmaps/pac32_010.pbm"
#include "bitmaps/pac32_011.pbm"
#include "bitmaps/pac32_012.pbm"
#include "bitmaps/pac32_013.pbm"
#include "bitmaps/pac32_020.pbm"
#include "bitmaps/pac32_021.pbm"
#include "bitmaps/pac32_022.pbm"
#include "bitmaps/pac32_023.pbm"
#include "bitmaps/pac32_030.pbm"
#include "bitmaps/pac32_031.pbm"
#include "bitmaps/pac32_032.pbm"
#include "bitmaps/pac32_033.pbm"
#include "bitmaps/pac32_100.pbm"
#include "bitmaps/pac32_101.pbm"
#include "bitmaps/pac32_102.pbm"
#include "bitmaps/pac32_103.pbm"
#include "bitmaps/pac32_110.pbm"
#include "bitmaps/pac32_111.pbm"
#include "bitmaps/pac32_112.pbm"
#include "bitmaps/pac32_113.pbm"
#include "bitmaps/pac32_120.pbm"
#include "bitmaps/pac32_121.pbm"
#include "bitmaps/pac32_122.pbm"
#include "bitmaps/pac32_123.pbm"
#include "bitmaps/pac32_130.pbm"
#include "bitmaps/pac32_131.pbm"
#include "bitmaps/pac32_132.pbm"
#include "bitmaps/pac32_133.pbm"
#include "bitmaps/blob48.pbm"
#include "bitmaps/pac48_000.pbm"
#include "bitmaps/pac48_001.pbm"
#include "bitmaps/pac48_002.pbm"
#include "bitmaps/pac48_003.pbm"
#include "bitmaps/pac48_010.pbm"
#include "bitmaps/pac48_011.pbm"
#include "bitmaps/pac48_012.pbm"
#include "bitmaps/pac48_013.pbm"
#include "bitmaps/pac48_020.pbm"
#include "bitmaps/pac48_021.pbm"
#include "bitmaps/pac48_022.pbm"
#include "bitmaps/pac48_023.pbm"
#include "bitmaps/pac48_030.pbm"
#include "bitmaps/pac48_031.pbm"
#include "bitmaps/pac48_032.pbm"
#include "bitmaps/pac48_033.pbm"
#include "bitmaps/pac48_100.pbm"
#include "bitmaps/pac48_101.pbm"
#include "bitmaps/pac48_102.pbm"
#include "bitmaps/pac48_103.pbm"
#include "bitmaps/pac48_110.pbm"
#include "bitmaps/pac48_111.pbm"
#include "bitmaps/pac48_112.pbm"
#include "bitmaps/pac48_113.pbm"
#include "bitmaps/pac48_120.pbm"
#include "bitmaps/pac48_121.pbm"
#include "bitmaps/pac48_122.pbm"
#include "bitmaps/pac48_123.pbm"
#include "bitmaps/pac48_130.pbm"
#include "bitmaps/pac48_131.pbm"
#include "bitmaps/pac48_132.pbm"
#include "bitmaps/pac48_133.pbm"


Tcl_Interp *theInterp = NULL;


extern int RunSimTclCmd _ANSI_ARGS_((ClientData clientData, 
				     Tcl_Interp *interp,
				     int argc, const char *argv[]));
extern int SetStatusTclCmd _ANSI_ARGS_((ClientData clientData, 
					Tcl_Interp *interp,
					int argc, const char *argv[]));



/*
 *----------------------------------------------------------------------
 *
 * main --
 *
 *	This is the main program for the application.
 *
 *
 * Side effects:
 *	Whatever the application does.
 *
 *----------------------------------------------------------------------
 */

int
main(argc, argv)
    int argc;			/* Number of command-line arguments. */
    char **argv;		/* Values of command-line arguments. */
{
    Tk_Main(argc, argv, Tcl_AppInit);
    return 0;			/* Needed only to prevent compiler warning. */
}

/*
 *----------------------------------------------------------------------
 *
 * Tcl_AppInit --
 *
 *	This procedure performs application-specific initialization.
 *	Most applications, especially those that incorporate additional
 *	packages, will have their own version of this procedure.
 *
 * Results:
 *	Returns a standard Tcl completion code, and leaves an error
 *	message in interp->result if an error occurs.
 *
 * Side effects:
 *	Depends on the startup script.
 *
 *----------------------------------------------------------------------
 */

int
Tcl_AppInit(interp)
    Tcl_Interp *interp;		/* Interpreter for application. */
{

    if (Tcl_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
    if (Tk_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }

    theInterp = interp;


    /*
     * Call Tcl_CreateCommand for application-specific commands, if
     * they weren't already created by the init procedures called above.
     */
    Tcl_CreateCommand(interp, "RunSim", RunSimTclCmd, 
		      (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
    Tcl_CreateCommand(interp, "SetStatus", SetStatusTclCmd, 
		      (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);

    Tk_DefineBitmap(interp, Tk_GetUid("blob16"), blob16_bits, 
		    blob16_width, blob16_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_000"), pac16_000_bits, 
		    pac16_000_width, pac16_000_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_001"), pac16_001_bits, 
		    pac16_001_width, pac16_001_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_002"), pac16_002_bits, 
		    pac16_002_width, pac16_002_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_003"), pac16_003_bits, 
		    pac16_003_width, pac16_003_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_010"), pac16_010_bits, 
		    pac16_010_width, pac16_010_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_011"), pac16_011_bits, 
		    pac16_011_width, pac16_011_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_012"), pac16_012_bits, 
		    pac16_012_width, pac16_012_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_013"), pac16_013_bits, 
		    pac16_013_width, pac16_013_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_020"), pac16_020_bits, 
		    pac16_020_width, pac16_020_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_021"), pac16_021_bits, 
		    pac16_021_width, pac16_021_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_022"), pac16_022_bits, 
		    pac16_022_width, pac16_022_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_023"), pac16_023_bits, 
		    pac16_023_width, pac16_023_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_030"), pac16_030_bits, 
		    pac16_030_width, pac16_030_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_031"), pac16_031_bits, 
		    pac16_031_width, pac16_031_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_032"), pac16_032_bits, 
		    pac16_032_width, pac16_032_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_033"), pac16_033_bits, 
		    pac16_033_width, pac16_033_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_100"), pac16_100_bits, 
		    pac16_100_width, pac16_100_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_101"), pac16_101_bits, 
		    pac16_101_width, pac16_101_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_102"), pac16_102_bits, 
		    pac16_102_width, pac16_102_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_103"), pac16_103_bits, 
		    pac16_103_width, pac16_103_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_110"), pac16_110_bits, 
		    pac16_110_width, pac16_110_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_111"), pac16_111_bits, 
		    pac16_111_width, pac16_111_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_112"), pac16_112_bits, 
		    pac16_112_width, pac16_112_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_113"), pac16_113_bits, 
		    pac16_113_width, pac16_113_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_120"), pac16_120_bits, 
		    pac16_120_width, pac16_120_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_121"), pac16_121_bits, 
		    pac16_121_width, pac16_121_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_122"), pac16_122_bits, 
		    pac16_122_width, pac16_122_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_123"), pac16_123_bits, 
		    pac16_123_width, pac16_123_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_130"), pac16_130_bits, 
		    pac16_130_width, pac16_130_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_131"), pac16_131_bits, 
		    pac16_131_width, pac16_131_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_132"), pac16_132_bits, 
		    pac16_132_width, pac16_132_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac16_133"), pac16_133_bits, 
		    pac16_133_width, pac16_133_height);

    Tk_DefineBitmap(interp, Tk_GetUid("blob32"), blob32_bits, 
		    blob32_width, blob32_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_000"), pac32_000_bits, 
		    pac32_000_width, pac32_000_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_001"), pac32_001_bits, 
		    pac32_001_width, pac32_001_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_002"), pac32_002_bits, 
		    pac32_002_width, pac32_002_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_003"), pac32_003_bits, 
		    pac32_003_width, pac32_003_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_010"), pac32_010_bits, 
		    pac32_010_width, pac32_010_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_011"), pac32_011_bits, 
		    pac32_011_width, pac32_011_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_012"), pac32_012_bits, 
		    pac32_012_width, pac32_012_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_013"), pac32_013_bits, 
		    pac32_013_width, pac32_013_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_020"), pac32_020_bits, 
		    pac32_020_width, pac32_020_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_021"), pac32_021_bits, 
		    pac32_021_width, pac32_021_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_022"), pac32_022_bits, 
		    pac32_022_width, pac32_022_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_023"), pac32_023_bits, 
		    pac32_023_width, pac32_023_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_030"), pac32_030_bits, 
		    pac32_030_width, pac32_030_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_031"), pac32_031_bits, 
		    pac32_031_width, pac32_031_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_032"), pac32_032_bits, 
		    pac32_032_width, pac32_032_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_033"), pac32_033_bits, 
		    pac32_033_width, pac32_033_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_100"), pac32_100_bits, 
		    pac32_100_width, pac32_100_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_101"), pac32_101_bits, 
		    pac32_101_width, pac32_101_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_102"), pac32_102_bits, 
		    pac32_102_width, pac32_102_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_103"), pac32_103_bits, 
		    pac32_103_width, pac32_103_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_110"), pac32_110_bits, 
		    pac32_110_width, pac32_110_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_111"), pac32_111_bits, 
		    pac32_111_width, pac32_111_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_112"), pac32_112_bits, 
		    pac32_112_width, pac32_112_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_113"), pac32_113_bits, 
		    pac32_113_width, pac32_113_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_120"), pac32_120_bits, 
		    pac32_120_width, pac32_120_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_121"), pac32_121_bits, 
		    pac32_121_width, pac32_121_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_122"), pac32_122_bits, 
		    pac32_122_width, pac32_122_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_123"), pac32_123_bits, 
		    pac32_123_width, pac32_123_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_130"), pac32_130_bits, 
		    pac32_130_width, pac32_130_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_131"), pac32_131_bits, 
		    pac32_131_width, pac32_131_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_132"), pac32_132_bits, 
		    pac32_132_width, pac32_132_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac32_133"), pac32_133_bits, 
		    pac32_133_width, pac32_133_height);

    Tk_DefineBitmap(interp, Tk_GetUid("blob48"), blob48_bits, 
		    blob48_width, blob48_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_000"), pac48_000_bits, 
		    pac48_000_width, pac48_000_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_001"), pac48_001_bits, 
		    pac48_001_width, pac48_001_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_002"), pac48_002_bits, 
		    pac48_002_width, pac48_002_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_003"), pac48_003_bits, 
		    pac48_003_width, pac48_003_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_010"), pac48_010_bits, 
		    pac48_010_width, pac48_010_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_011"), pac48_011_bits, 
		    pac48_011_width, pac48_011_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_012"), pac48_012_bits, 
		    pac48_012_width, pac48_012_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_013"), pac48_013_bits, 
		    pac48_013_width, pac48_013_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_020"), pac48_020_bits, 
		    pac48_020_width, pac48_020_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_021"), pac48_021_bits, 
		    pac48_021_width, pac48_021_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_022"), pac48_022_bits, 
		    pac48_022_width, pac48_022_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_023"), pac48_023_bits, 
		    pac48_023_width, pac48_023_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_030"), pac48_030_bits, 
		    pac48_030_width, pac48_030_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_031"), pac48_031_bits, 
		    pac48_031_width, pac48_031_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_032"), pac48_032_bits, 
		    pac48_032_width, pac48_032_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_033"), pac48_033_bits, 
		    pac48_033_width, pac48_033_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_100"), pac48_100_bits, 
		    pac48_100_width, pac48_100_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_101"), pac48_101_bits, 
		    pac48_101_width, pac48_101_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_102"), pac48_102_bits, 
		    pac48_102_width, pac48_102_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_103"), pac48_103_bits, 
		    pac48_103_width, pac48_103_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_110"), pac48_110_bits, 
		    pac48_110_width, pac48_110_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_111"), pac48_111_bits, 
		    pac48_111_width, pac48_111_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_112"), pac48_112_bits, 
		    pac48_112_width, pac48_112_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_113"), pac48_113_bits, 
		    pac48_113_width, pac48_113_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_120"), pac48_120_bits, 
		    pac48_120_width, pac48_120_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_121"), pac48_121_bits, 
		    pac48_121_width, pac48_121_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_122"), pac48_122_bits, 
		    pac48_122_width, pac48_122_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_123"), pac48_123_bits, 
		    pac48_123_width, pac48_123_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_130"), pac48_130_bits, 
		    pac48_130_width, pac48_130_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_131"), pac48_131_bits, 
		    pac48_131_width, pac48_131_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_132"), pac48_132_bits, 
		    pac48_132_width, pac48_132_height);
    Tk_DefineBitmap(interp, Tk_GetUid("pac48_133"), pac48_133_bits, 
		    pac48_133_width, pac48_133_height);

    /*
     * Specify a user-specific startup file to invoke if the application
     * is run interactively.  Typically the startup file is "~/.apprc"
     * where "app" is the name of the application.  If this line is deleted
     * then no user-specific startup file will be run under any conditions.
     */

    /*    tcl_RcFileName = "~/.wishrc"; */
    return TCL_OK;
}
