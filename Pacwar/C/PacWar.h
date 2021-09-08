/* *******************************************************************
 *
 * Header file for PacWar.
 *
 * ******************************************************************/

#ifndef _PACWAR_H
#define _PACWAR_H

/* A gene for a pac-mite.  
   When refering to turns, we mean ccw, 90-degree turns.
   The difference between directions A and B is how many turns A would have 
   to make to be in the same direction as B*/
typedef struct gene
{
  signed char u[4];    /* If an empty cell has 1 oldest mite facing it,
			    this is how much the new baby should be turned
			    from the direction its "mother" is facing, based
			    on the age of the mother. */
  signed char v[4][4]; /* If a mite is faced by a single, stronger enemy,
			    it is replaced by a baby enemy.  The amount it 
			    is turned from its "mother" is based on the 
			    difference between atackee's and attacker's
			    direction AND the age of the attacker */
  signed char w[3];    /* If a mite survives this round and is facing the
			    edge of the world(!), it will turn this amount
			    based on his age. */
  signed char x[3];    /* If a mite survives this round and is facing an
			    empty cell, it will turn this amount based on 
			    his age. */
  signed char y[4][3]; /* If a mite survives this round and is facing a
			    friendly mite, it will turn this amount based on 
			    the difference between the mite's and his friend's
			    directions AND his age. */
  signed char z[4][3]; /* If a mite survives this round and is facing an
			    enemy mite, it will turn this amount based on 
			    the difference between the mite's and his enemy's
			    directions AND his age. */
} PacGene, *PacGenePtr;

/* What can be in a cell: one of the two species of mites, a blob
   (i.e. an empty cell), or a barrier (a cell off-screen) */
enum CellKind
{
  Species1,
  Species2,
  Blob,
  Barrier
};

/* Possible directions */
enum Direction
{
  East,
  North,
  West,
  South,
  Num_dirs
};

/* A location in the world */
typedef struct cell
{
  signed char kind;
  signed char dir; /* For non-mites, this is meaningless */
  signed char age; /* For non-mites, this is -1 */
} Cell, *CellPtr;

/* The size of the world, and the World itself */
#define MaxX 21
#define MaxY 11
typedef Cell World[MaxX][MaxY];

/* *******************************************************************
 * Initialize a PacGene from a string (50 chars, each '0'..'3').
 * *******************************************************************/
int SetGeneFromString(char *s, PacGenePtr g);

/* *******************************************************************
 * Create a string representation of a PacGene, yielding that string.
 * If s is NULL, the string will be malloc'd (so the caller's
 * responsibility to free it); o/w the string is written into the 51
 * bytes starting at s.
 * *******************************************************************/
char *NewStringFromGene(PacGenePtr g, char *s);

/* *******************************************************************
 * Given the old World, compute the New one where gs is an array of 
 * 2 PacGenes, and count is an array fo 2 integers that will be set to
 * the number of mites of each species that are present in the new
 * world.  Draw is a pointer to a function called to update cells that
 * change.
 * *******************************************************************/
void ComputeNewWorld(World *old, World *new, PacGenePtr *gs, int *count,
                     void (*draw)(int x, int y, Cell c));

/* *******************************************************************
 * Initialize the two worlds (only do barrier for second) 
 * *******************************************************************/
void PrepSim(World *w1, World *w2);

/* *******************************************************************
 * Initialize the two worlds for a test by placing 1 mite in the 
 * center of w1 (the "starting" world).  Draw is a pointer to a 
 * function called to update cells that change.
 * *******************************************************************/
void PrepTest(World *w1, World *w2, void (*draw)(int x, int y, Cell c));

/* *******************************************************************
 * Initialize the two worlds for a duel by placing 1 mite in the 
 * center of each side of w1 (the "starting" world).  Draw is a pointer
 * to a function called to update cells that change.
 * *******************************************************************/
void PrepDuel(World *w1, World *w2, void (*draw)(int x, int y, Cell c));

/* *******************************************************************
 * Run a duel between species with genes g1 & g2 for at most *rounds.
 * When done, *rounds will be the number of rounds actually required,
 * *count1 & *count2 will be the number of each species of mite still
 * standing when done.
 * *******************************************************************/
void FastDuel(PacGenePtr g1, PacGenePtr g2,
              int *rounds, int *count1, int *count2);

/* *******************************************************************
 * Run a test of the species with gene g1 for at most *rounds.
 * When done, *rounds will be the number of rounds actually required,
 * *count1 will be the number of mites still standing when done.
 * *******************************************************************/
void FastTest(PacGenePtr g1, int *rounds, int *count1);

#endif
