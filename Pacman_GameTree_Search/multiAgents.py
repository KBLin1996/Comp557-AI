# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        value = 0
        newFood_coordinate = newFood.asList()

        # If the position after we move contains a food => value += 1
        if newFood[newPos[0]][newPos[1]]:
            value += 1

        # If the position after we move contains a ghost => return a value that substracts 9999
        for ghost_states in newGhostStates:
            if util.manhattanDistance(newPos, ghost_states.getPosition()) < 1:
                value -= 9999
                return value

        # Accumulate the estimate value that we eat all food dots in minimum manhattanDistance()
        while len(newFood_coordinate) != 0:
            min_distance = util.manhattanDistance(newPos, newFood_coordinate[0])
            min_coordinate = newFood_coordinate[0]
            for coordinate in newFood_coordinate:
                cur_distance = util.manhattanDistance(newPos, coordinate)
                if cur_distance < min_distance:
                    min_distance = cur_distance
                    min_coordinate = coordinate
            newFood_coordinate.remove(min_coordinate)
            value += 1 / min_distance

        value += successorGameState.getScore()
        return value

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        agentIndex = 0
        depth = self.depth
        value, action = self.value(gameState, agentIndex, depth)
        return action

    def value(self, gameState, agentIndex, depth):
        if  depth == 0 or gameState.isWin() or gameState.isLose():
            return scoreEvaluationFunction(gameState), ""
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        value = float("-inf")
        bestA = ""
        numA = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            curD = depth if (agentIndex+1) % numA else depth-1
            curV = self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex+1)%numA, curD)[0]
            if curV > value:
                value = curV
                bestA = action
        return value, bestA

    def minValue(self, gameState, agentIndex, depth):
        value = float("inf")
        bestA = ""
        numA = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            curD = depth if (agentIndex+1) % numA else depth-1
            curV = self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex+1)%numA, curD)[0]
            if curV < value:
                value = curV
                bestA = action
        return value, bestA

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        agentIndex = 0
        depth = self.depth
        alpha = float("-inf")
        beta = float("inf")
        value, action = self.value(gameState, agentIndex, depth, alpha, beta)
        return action

    def value(self, gameState, agentIndex, depth, alpha, beta):
        if  depth == 0 or gameState.isWin() or gameState.isLose():
            return gameState.getScore(), ""
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        value = float("-inf")
        bestA = ""
        numA = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            curD = depth if (agentIndex+1) % numA else depth-1
            curV = self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex+1)%numA, curD, alpha, beta)[0]
            if curV > value:
                value = curV
                bestA = action
            alpha = max(alpha, value)
            if value > beta:
                break
        return value, bestA

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        value = float("inf")
        bestA = ""
        numA = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            curD = depth if (agentIndex+1) % numA else depth-1
            curV = self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex+1)%numA, curD, alpha, beta)[0]
            if curV < value:
                value = curV
                bestA = action
            beta = min(beta, value)
            if value < alpha:
                break
        return value, bestA

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        agentIndex = 0
        depth = self.depth
        value, action = self.value(gameState, agentIndex, depth)
        return action
        
    def value(self, gameState, agentIndex, depth):
        if  depth == 0 or gameState.isWin() or gameState.isLose():
            return scoreEvaluationFunction(gameState), ""
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.expectValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        value = float("-inf")
        bestA = ""
        numA = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            curD = depth if (agentIndex+1) % numA else depth-1
            curV = self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex+1)%numA, curD)[0]
            if curV > value:
                value = curV
                bestA = action
        return value, bestA

    def expectValue(self, gameState, agentIndex, depth):
        value = 0
        bestA = ""
        numA = gameState.getNumAgents()
        LegalActions = gameState.getLegalActions(agentIndex)
        numLA = len(LegalActions)
        for action in LegalActions:
            curD = depth if (agentIndex+1) % numA else depth-1
            value += self.value(gameState.generateSuccessor(agentIndex, action), (agentIndex+1)%numA, curD)[0]
        return value/numLA, bestA
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    flag = False
    value = 0
    # weight = [food_weight, len_power_food_weight, score_weight, food_left_weight]
    weight = [10, -10, 200, -100]

    # Get information on currentGameState (agents' position, capsules' position and food coordinate)
    pacman_pos = currentGameState.getPacmanPosition()
    ghosts_pos = currentGameState.getGhostPositions()
    capsule_pos = currentGameState.getCapsules()
    food_coordinate = currentGameState.getFood().asList()

    # If the ghosts are near us => adjust the weight of finding power food and normal food finding
    for ghost_pos in ghosts_pos:
        if util.manhattanDistance(pacman_pos, ghost_pos) < 2:
            flag = 1

    # Determine the food finding value (find the closest food distance)
    if not flag:
        min_distance = util.manhattanDistance(pacman_pos, food_coordinate[0])
        for coordinate in food_coordinate:
            cur_distance = util.manhattanDistance(pacman_pos, coordinate)
            if cur_distance < min_distance:
                min_distance = cur_distance
    else:
        min_distance = 10000
    food_value = (1 / min_distance) * weight[0] + len(food_coordinate) * weight[3]

    # Determine the capsule value
    capsule_value = len(capsule_pos) * weight[1]

    # Determine the gamescore value
    game_value = currentGameState.getScore() * weight[2]

    value = food_value + capsule_value + game_value
    return value

# Abbreviation
better = betterEvaluationFunction
