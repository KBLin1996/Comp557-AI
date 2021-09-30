import collections, util, math, random, copy

############################################################
# Problem 4.1.1

def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).  Note that |V| is a dictionary.  
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    action_val = 0
    all_movement_info = list()
    all_movement_info = mdp.succAndProbReward(state, action)

    for newState, prob, reward in all_movement_info:
        action_val += prob * (reward + mdp.discount() * V[newState])
    return action_val
# END_YOUR_CODE

############################################################
# Problem 4.1.2

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.  Note that |V| and |pi| are
    dictionaries.
    """
    # BEGIN_YOUR_CODE (around 7 lines of code expected)
    while True:
        val_changed = 0
        for state in mdp.states:
            state_val = computeQ(mdp, V, state, pi[state])
            val_changed = max(val_changed, abs(V[state] - state_val))
            V[state] = state_val
        if val_changed < epsilon: break
    return V
# END_YOUR_CODE

############################################################
# Problem 4.1.3

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().  Note that |V| is a
    dictionary.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    pi = dict()
    for state in mdp.states:
        max_val = -math.inf
        for action in mdp.actions(state):
            if max_val < computeQ(mdp, V, state, action):
                pi[state] = action
    return pi
# END_YOUR_CODE

############################################################
# Problem 4.1.4

class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # compute |V| and |pi|, which should both be dicts
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        V = dict()
        pi = dict()
        for state in mdp.states:
            V[state] = 0
            pi[state] = 1

        while True:
            prev_pi = copy.deepcopy(pi)
            V = policyEvaluation(mdp, V, pi, epsilon)
            pi = computeOptimalPolicy(mdp, V)
            if prev_pi == pi:
                break
        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 4.1.5

class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # BEGIN_YOUR_CODE (around 10 lines of code expected)
        V = collections.defaultdict(float)
        while True:
            val_changed = 0
            for state in mdp.states:
                state_val = max([computeQ(mdp, V, state, action) for action in mdp.actions(state)])
                val_changed = max(val_changed, abs(V[state] - state_val))
                V[state] = state_val
            if val_changed < epsilon:
                break
        pi = computeOptimalPolicy(mdp, V)
        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 4.1.6

# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().
class CounterexampleMDP(util.MDP):
    def __init__(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
    # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
    # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
    # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
    # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
    # END_YOUR_CODE

def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    raise Exception("Not implemented yet")
# END_YOUR_CODE

############################################################
# Problem 4.2.1

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 50 lines of code expected)
        result = []
        totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts = state

        if deckCardCounts is None:
            return []

        elif action == "Peek":
            if nextCardIndexIfPeeked != None:
                return []
            else:
                for idx, count in enumerate(deckCardCounts):
                    if count > 0:
                        prob = count / sum(deckCardCounts)
                        nextState = (totalCardValueInHand, idx, deckCardCounts)
                        result.append((nextState, prob, -self.peekCost))

        elif action == "Take":
            if nextCardIndexIfPeeked != None:
                deckCardCountsList = list(deckCardCounts)
                deckCardCountsList[nextCardIndexIfPeeked] -= 1
                totalCardValueInHand += self.cardValues[nextCardIndexIfPeeked]

                if totalCardValueInHand > self.threshold:
                    nextState = (totalCardValueInHand, None, None)
                    result.append((nextState, 1.0, 0))
                elif sum(deckCardCounts) == 0:
                    nextState = (totalCardValueInHand, None, None)
                    result.append((nextState, 1.0, totalCardValueInHand))
                else:
                    nextState = (totalCardValueInHand, None, tuple(deckCardCountsList))
                    result.append((nextState, 1.0, 0))
            else:
                for idx, count in enumerate(deckCardCounts):
                    if count > 0:
                        prevCardValueInHand = totalCardValueInHand
                        prob = count / sum(deckCardCounts)
                        prevCardValueInHand += self.cardValues[idx]
                        deckCardCountsList = list(deckCardCounts)
                        deckCardCountsList[idx] -= 1

                        if prevCardValueInHand > self.threshold:
                            nextState = (prevCardValueInHand, None, None)
                            result.append((nextState, prob, 0))
                        elif sum(deckCardCountsList) == 0:
                            nextState = (prevCardValueInHand, None, None)
                            result.append((nextState, 1.0, prevCardValueInHand))
                        else:
                            nextState = (prevCardValueInHand, None, tuple(deckCardCountsList))
                            result.append((nextState, prob, 0))

        elif action == "Quit":
            nextState = (totalCardValueInHand, None, None)
            result.append((nextState, 1, totalCardValueInHand))

        return result
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 4.2.2

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE
