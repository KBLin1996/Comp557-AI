#include "tk.h"
#include "PacWar.h"
#include <stdio.h>
#include <string.h>
/*int printf(); */

/* The interpreter & space to store commands for interacting w/ Tcl */
extern Tcl_Interp *theInterp; // global command interpreter
static char tclCmd[100];

/* The possible states the simulator is in */

enum simStatus
{
  STEPPING,
  RUNNING,
  STOPPED,
  RESET,
  NUM_STAT
};
static char simStatusText[NUM_STAT][10] = {"step", "run", "stop", "reset"};

static int status = RESET; /* The current state of the simulator */
static int duel = 1;       /* Is this a duel (2 species) or a test (1) ? */

World w[2];         /* The old and new worlds */
int clockCount = 0; /* Number of rounds simulated */
int count[2];       /* Number of mites in the current world of each species */
int order = 0;      /* Which w is the old one? */
PacGene g[2];       /* Genes for the two species */
PacGenePtr gp[2] = {&g[0], &g[1]};

/* ******************************************************************
 * Execute the tcl command tclCmd; report any error.
 * *******************************************************************/
char *DoTclCmd(char *tclCmd)
{
  int rc;
  if ((rc = Tcl_Eval(theInterp, tclCmd)) != 0)
  {
    printf("For tcl command '%s',\nError %d: %s\n", tclCmd, rc, Tcl_GetStringResult(theInterp));
    return (char *)Tcl_GetStringResult(theInterp);
  }
  return (0);
}

/* *********************************************************************
 * Update the display of cell (x,y) containing c
 * *********************************************************************/
void DrawCell(int x, int y, Cell c)
{

  sprintf(tclCmd, "DrawCell %d %d %d %d %d", x, y, c.kind, c.dir, c.age);
  DoTclCmd(tclCmd);
}

static void (*drawFcn)(int x, int y, Cell c) = &DrawCell;

/***************************************************************************
  
  The following are functions that can be called from a Tcl script
  
***************************************************************************/

/* *********************************************************************
 * This runs a simulation 
 * *********************************************************************/
int RunSimTclCmd _ANSI_ARGS_((ClientData clientData, Tcl_Interp *interp,
                              int argc, char *argv[]))
{
  int newStatus;
  char *kind, *show, *g1, *g2;

  if (argc != 2)
  {
    Tcl_SetResult(interp, "wrong # args", TCL_DYNAMIC);
    return TCL_ERROR;
  }

  /* Verify the new status */
  if (strcmp(argv[1], simStatusText[RUNNING]) == 0)
    newStatus = RUNNING;
  else if (strcmp(argv[1], simStatusText[STEPPING]) == 0)
    newStatus = STEPPING;
  else
  {
    Tcl_SetResult(interp, "Status must be step or run", TCL_DYNAMIC);
    return TCL_ERROR;
  }

  /* Save the interpreter; we'll need it later */
  theInterp = interp;

  /* If starting a simulation, check animation & such */
  if (status == RESET)
  {
    clockCount = 0;
    kind = (char *)Tcl_GetVar(interp, "kind", TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    if (kind == NULL)
    {
      printf("When accessing `kind', TCL reported: %s\n",
             Tcl_GetStringResult(interp));
      Tcl_SetResult(interp, "kind not found", TCL_DYNAMIC);
      return TCL_ERROR;
    }
    show = (char *)Tcl_GetVar(interp, "showRounds",
                              TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    if (show == NULL)
    {
      printf("When accessing `showRounds', TCL reported: %s\n",
             Tcl_GetStringResult(interp));
      Tcl_SetResult(interp, "showRounds not found", TCL_DYNAMIC);
      return TCL_ERROR;
    }
    else if (*show == '1')
      drawFcn = &DrawCell;
    else
      drawFcn = NULL;
    sprintf(tclCmd, "global spec; global spec1name; set spec($spec1name)");
    DoTclCmd(tclCmd);
    g1 = (char *)Tcl_GetStringResult(interp);
    if (g1 == NULL)
    {
      Tcl_SetResult(interp, "gene-string 1 not found", TCL_DYNAMIC);
      return TCL_ERROR;
    }
    if (SetGeneFromString(g1, g) == 0)
    {
      Tcl_SetResult(interp, "Gene-string 1 was bad", TCL_DYNAMIC);
      return TCL_ERROR;
    }
    count[0] = 1;
    order = 0;
    if (strcmp(kind, "duel") == 0)
    {
      sprintf(tclCmd,
              "global spec; global spec2name; set spec($spec2name)");
      DoTclCmd(tclCmd);
      g2 = (char *)Tcl_GetStringResult(interp);
      if (g2 == NULL)
      {
        Tcl_SetResult(interp, "gene-string 2 not found", TCL_DYNAMIC);
        return TCL_ERROR;
      }
      if (SetGeneFromString(g2, g + 1) == 0)
      {
        Tcl_SetResult(interp, "Gene-string 2 was bad", TCL_DYNAMIC);
        return TCL_ERROR;
      }
      count[1] = 1;
      duel = 1;
      PrepDuel(&(w[0]), &(w[1]), &DrawCell);
    }
    else if (strcmp(kind, "test") == 0)
    {
      count[1] = 0;
      duel = 0;
      PrepTest(&(w[0]), &(w[1]), &DrawCell);
    }
    else
    {
      Tcl_SetResult(interp, "kind has invalid value", TCL_DYNAMIC);
      return TCL_ERROR;
    }
    sprintf(tclCmd, "%d", clockCount);
    Tcl_SetVar(interp, "display_clockCount", tclCmd,
               TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    sprintf(tclCmd, "%d", count[0]);
    Tcl_SetVar(interp, "spec1Cnt", tclCmd,
               TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    if (duel)
    {
      sprintf(tclCmd, "%d", count[1]);
      Tcl_SetVar(interp, "spec2Cnt", tclCmd,
                 TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    }
    DoTclCmd("update");
    if (newStatus == STEPPING)
      status = STOPPED;
    else
      status = RUNNING;
  }
  else
    status = RUNNING;

  /* Main loop; keep going until told to stop or nothing left to do */
  while (status == RUNNING)
  {
    ComputeNewWorld(&(w[order]), &(w[1 - order]), gp, count, drawFcn);
    order = 1 - order;
    clockCount++;
    sprintf(tclCmd, "%d", clockCount);
    Tcl_SetVar(interp, "display_clockCount", tclCmd,
               TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    sprintf(tclCmd, "%d", count[0]);
    Tcl_SetVar(interp, "spec1Cnt", tclCmd,
               TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    if (duel)
    {
      sprintf(tclCmd, "%d", count[1]);
      Tcl_SetVar(interp, "spec2Cnt", tclCmd,
                 TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    }
    DoTclCmd("update");
    if (status == RUNNING && (newStatus == STEPPING || clockCount == 500 ||
                              count[0] == 0 || (duel && count[1] == 0)))
      status = STOPPED;
  }
  if (status == STOPPED)
  {
    DoTclCmd("StopSimulation");
    if (drawFcn == NULL)
    {
      int x, y;
      for (x = 1; x < MaxX - 1; x++)
        for (y = 1; y < MaxY - 1; y++)
          DrawCell(x, y, w[order][x][y]);
    }
  }
  return TCL_OK;
}

/* *********************************************************************
 * This sets the status variable 
 * *********************************************************************/
int SetStatusTclCmd _ANSI_ARGS_((ClientData clientData, Tcl_Interp *interp,
                                 int argc, char *argv[]))
{
  int i;

  if (argc != 2)
  {
    Tcl_SetResult(interp, "Wrong # args", TCL_DYNAMIC);
    return TCL_ERROR;
  }
  for (i = 1; i < NUM_STAT; i++)
    if (strcmp(argv[1], simStatusText[i]) == 0)
    {
      status = i;
      return TCL_OK;
    }
  Tcl_SetResult(interp, "arg 1 must be either stop, run, or reset", TCL_DYNAMIC);
  return TCL_ERROR;
}
