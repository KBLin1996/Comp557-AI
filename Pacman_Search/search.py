# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    state = problem.getStartState() # Get start state
    s = util.Stack()    # A variable of stack
    visited = list() # Record the position

    s.push((state, [], 0))  # Push the initial state
    while not s.isEmpty():
        (pos, direction, cost) = s.pop()
        if problem.isGoalState(pos):
            return direction
        if pos in visited:
            continue
        visited.append(pos)

        # Get information of the successor
        info = problem.getSuccessors(pos)

        for i in info:
            if i[0] not in visited:
                s.push((i[0], direction + [i[1]], cost + i[2]))

    if s.isEmpty():
        print("Error! The stack is empty but no Goal is found.")
        return 0

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    q = util.Queue()
    visited = list()

    q.push((state, [], 0))
    while not q.isEmpty():
        (pos, direction, cost) = q.pop()
        if problem.isGoalState(pos):
            return direction
        if pos in visited:
            continue
        visited.append(pos)

        info = problem.getSuccessors(pos)

        for i in info:
            if i[0] not in visited:
                q.push((i[0], direction + [i[1]], cost + i[2]))

    if q.isEmpty():
        print("Error! The queue is empty but no Goal is found.")

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier_info = dict()
    visited = list()

    frontier.push((state, [], 0), 0)
    while not frontier.isEmpty():
        (pos, direction, cost) = frontier.pop()
        if problem.isGoalState(pos):
            return direction
        if pos in visited:
            continue
        visited.append(pos)

        info = problem.getSuccessors(pos)

        for i in info:
            if i[0] not in visited and i[0] not in frontier_info.keys():
                frontier.push((i[0], direction + [i[1]], cost + i[2]), cost + i[2])
                frontier_info[i[0]] = cost + i[2]
            elif i[0] in frontier_info.keys() and cost + i[2] < frontier_info[i[0]]:
                frontier.update((i[0], direction + [i[1]], cost + i[2]), cost + i[2])
                frontier_info[i[0]] = cost + i[2]

    if frontier.isEmpty():
        print("Error! The priority queue is empty but no Goal is found.")

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier_info = dict()
    visited = list()

    frontier.push((state, [], 0), 0)
    while not frontier.isEmpty():
        (pos, direction, cost) = frontier.pop()
        if problem.isGoalState(pos):
            return direction
        if pos in visited:
            continue
        visited.append(pos)

        info = problem.getSuccessors(pos)

        for i in info:
            heuristic_diff = heuristic(i[0], problem) - heuristic(pos, problem)
            if i[0] not in visited and i[0] not in frontier_info.keys():
                frontier.push((i[0], direction + [i[1]], cost + i[2] + heuristic_diff), cost + i[2] + heuristic_diff)
                frontier_info[i[0]] = cost + i[2] + heuristic_diff
            elif i[0] in frontier_info.keys() and cost + i[2] + heuristic_diff < frontier_info[i[0]]:
                frontier.update((i[0], direction + [i[1]], cost + i[2] + heuristic_diff), cost + i[2] + heuristic_diff)
                frontier_info[i[0]] = cost + i[2] + heuristic_diff

    if frontier.isEmpty():
        print("Error! The priority queue is empty but no Goal is found.")


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
