#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "PacWar.h"
#include <string.h>

/* The x and y components of the directions, and their opposites */
int dx[4] = {1, 0, -1, 0};
int dy[4] = {0, -1, 0, 1};
int opdir[4] = {West, South, East, North};

/* *******************************************************************
 * Initialize a PacGene from a string (50 chars, each '0'..'3').
 * *******************************************************************/
int SetGeneFromString(char *s, PacGenePtr g)
{
  signed char *gs = (signed char *)g;
  int i;

  if (strlen(s) != 50)
    return 0;

  for (i = 0; i < 50; i++)
  {
    if (!(*s >= '0' && *s <= '4'))
      return 0;
    *gs++ = (*s++) - '0';
  }
  return 1;
}

/* *******************************************************************
 * Create a string representation of a PacGene, yielding that string.
 * If s is NULL, the string will be malloc'd (so the caller's
 * responsibility to free it); o/w the string is written into the 51
 * bytes starting at s.
 * *******************************************************************/
char *NewStringFromGene(PacGenePtr g, char *s)
{
  signed char *gs = (signed char *)g;
  int i;

  if (s == NULL)
    s = (signed char *)malloc(51);
  for (i = 0; i < 50; i++)
  {
    if (*gs < 0 || *gs > 3)
    {
      free(s);
      return NULL;
    }
    s[i] = (*gs++) + '0';
  }
  s[50] = '\0';
  return s;
}

/* *******************************************************************
 * Given the old World, compute the New one where gs is an array of 
 * 2 PacGenes, and count is an array fo 2 integers that will be set to
 * the number of mites of each species that are present in the new
 * world.  Draw is a pointer to a function called to update cells that
 * change.
 * *******************************************************************/
void ComputeNewWorld(World *old, World *new, PacGenePtr *gs, int *count,
                     void (*draw)(int x, int y, Cell c))
{
  int x, y, d;

  count[Species1] = 0;
  count[Species2] = 0;
  for (x = 1; x < MaxX - 1; x++)
  {
    for (y = 1; y < MaxY - 1; y++)
    {
      Cell spot = (*old)[x][y];
      Cell enemy = (*old)[0][0]; /* Who's the strongest attacker? */
      int num_attack = 0;        /* How many are strongest? */
      int other_species = (spot.kind == Species1 ? Species2 : Species1);
      PacGenePtr g = *(gs + spot.kind);

      for (d = 0; d < Num_dirs; d++)
      {
        Cell neighbor = (*old)[x + dx[d]][y + dy[d]];
        if (neighbor.kind <= Species2 && d == opdir[neighbor.dir])
        {
          if (enemy.age < neighbor.age)
          {
            num_attack = 1;
            enemy = neighbor;
          }
          else if (enemy.age == neighbor.age)
          {
            num_attack++;
            if (neighbor.kind != spot.kind)
              enemy = neighbor;
          }
        }
      }

      if (spot.kind == Blob)
      {
        if (num_attack == 1)
        {
          /* We have a brith! */
          enemy.dir = (enemy.dir + gs[enemy.kind]->u[enemy.age]) % 4;
          enemy.age = 0;
          spot = enemy;
        }
      }
      else if (enemy.kind == other_species)
      {
        if (num_attack > 1)
        {
          /* Blow spot away */
          spot.kind = Blob;
          spot.age = -1;
        }
        else
        {
          /* Unique enemy attacker "converts" spot */
          int e = (spot.dir - enemy.dir + 4) % 4;
          enemy.dir = (enemy.dir + gs[enemy.kind]->v[e][enemy.age]) % 4;
          enemy.age = 0;
          spot = enemy;
        }
      }
      else if (spot.age == 3)
      {
        /* Spot dies of old age */
        spot.kind = Blob;
        spot.age = -1;
      }
      else
      {
        /* Spot gets a round older! */
        Cell neighbor = (*old)[x + dx[spot.dir]][y + dy[spot.dir]];
        int e;
        switch (neighbor.kind)
        {
        case Barrier:
          spot.dir = (spot.dir + g->w[spot.age]) % 4;
          break;
        case Blob:
          spot.dir = (spot.dir + g->x[spot.age]) % 4;
          break;
        case Species1:
        case Species2:
          e = (neighbor.dir - spot.dir + 4) % 4;
          if (neighbor.kind == spot.kind)
            spot.dir = (spot.dir + g->y[e][spot.age]) % 4;
          else
            spot.dir = (spot.dir + g->z[e][spot.age]) % 4;
        }
        spot.age++;
      }
      if (spot.kind <= Species2)
        count[spot.kind]++;
      (*new)[x][y] = spot;
      if (draw != NULL && !((*old)[x][y].kind == Blob && spot.kind == Blob))
      {
        draw(x, y, spot);
      }
    }
  }
}

/* *******************************************************************
 * Initialize the two worlds (only do barrier for second) 
 * *******************************************************************/
void PrepSim(World *w1, World *w2)
{
  int x, y;
  Cell c;

  c.kind = Blob;
  c.age = -1;
  c.dir = -1;
  for (x = 1; x < MaxX; x++)
    for (y = 1; y < MaxY; y++)
      (*w1)[x][y] = c;
  c.kind = Barrier;
  for (x = 0; x <= MaxX - 1; x++)
    (*w1)[x][0] = (*w1)[x][MaxY - 1] = (*w2)[x][0] = (*w2)[x][MaxY - 1] = c;
  for (y = 1; y < MaxY - 1; y++)
    (*w1)[0][y] = (*w1)[MaxX - 1][y] = (*w2)[0][y] = (*w2)[MaxX - 1][y] = c;
}

/* *******************************************************************
 * Initialize the two worlds for a test by placing 1 mite in the 
 * center of w1 (the "starting" world).  Draw is a pointer to a 
 * function called to update cells that change.
 * *******************************************************************/
void PrepTest(World *w1, World *w2, void (*draw)(int, int, Cell))
{
  PrepSim(w1, w2);
  (*w1)[10][5].kind = Species1;
  (*w1)[10][5].age = 0;
  (*w1)[10][5].dir = East;
  if (draw != NULL)
    draw(10, 5, (*w1)[10][5]);
}

/* *******************************************************************
 * Initialize the two worlds for a duel by placing 1 mite in the 
 * center of each side of w1 (the "starting" world).  Draw is a pointer
 * to a function called to update cells that change.
 * *******************************************************************/
void PrepDuel(World *w1, World *w2, void (*draw)(int, int, Cell))
{
  PrepSim(w1, w2);
  (*w1)[5][5].kind = Species1;
  (*w1)[5][5].age = 0;
  (*w1)[5][5].dir = East;
  (*w1)[15][5].kind = Species2;
  (*w1)[15][5].age = 0;
  (*w1)[15][5].dir = West;
  if (draw != NULL)
  {
    draw(15, 5, (*w1)[15][5]);
    draw(5, 5, (*w1)[5][5]);
  }
}

/* *******************************************************************
 * Run a duel between species with genes g1 & g2 for at most *rounds.
 * When done, *rounds will be the number of rounds actually required,
 * *count1 & *count2 will be the number of each species of mite still
 * standing when done.
 * *******************************************************************/
void FastDuel(PacGenePtr g1, PacGenePtr g2,
              int *rounds, int *count1, int *count2)
{
  World w[2];
  int round = 0;
  int count[2] = {1, 1};
  int order = 0;
  PacGenePtr g[2] = {g1, g2};

  PrepDuel(&(w[0]), &(w[1]), NULL);
  while (round < *rounds && (count[0] > 0 && count[1] > 0))
  {
    ComputeNewWorld(&(w[order]), &(w[1 - order]), g, count, NULL);
    order = 1 - order;
    round++;
  }

  *count1 = count[0];
  *count2 = count[1];
  *rounds = round;
}

/* *******************************************************************
 * Run a test of the species with gene g1 for at most *rounds.
 * When done, *rounds will be the number of rounds actually required,
 * *count1 will be the number of mites still standing when done.
 * *******************************************************************/
void FastTest(PacGenePtr g1, int *rounds, int *count1)
{
  World w[2];
  int round = 0;
  int count[2] = {1, 0};
  int order = 0;
  PacGenePtr g[2] = {g1, NULL};

  PrepTest(&(w[0]), &(w[1]), NULL);
  while (round < *rounds && (count[0] > 0))
  {
    ComputeNewWorld(&(w[order]), &(w[1 - order]), g, count, NULL);
    order = 1 - order;
    round++;
  }

  *count1 = count[0];
  *rounds = round;
}
