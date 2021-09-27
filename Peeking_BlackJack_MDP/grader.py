#!/usr/bin/env python
import random, util, collections

import graderUtil
grader = graderUtil.Grader()
submission = grader.load('submission')


############################################################
# Manual problems

def testQ(f, V):
    mdp = util.NumberLineMDP()
    goldQ = {}
    values = [l.split() for l in open(f)]
    for state, action, value in values:
        goldQ[(int(state), int(action))] = float(value)
    for state in range(-5,6):
        for action in [-1,1]:
            if not grader.requireIsEqual(goldQ[(state, action)],
                                         submission.computeQ(mdp, V, state,
                                             action)):
                print ('   state: {}, action: {}'.format(state, action))

def test1a_0():
    V = collections.defaultdict(lambda: 1)
    testQ('1a_0.gold', V)
grader.addBasicPart('4.1.1-0', test1a_0,2)

def test1a_1():
    V = collections.Counter()  # state -> value of state
    for state in range(-5,6):
        V[state] = state
    testQ('1a_1.gold', V)
grader.addBasicPart('4.1.1-1', test1a_1,3)

def test1b():
    V = collections.defaultdict(int)
    pi = collections.defaultdict(lambda: -1)
    mdp = util.NumberLineMDP()
    mdp.computeStates()
    goldV = {}
    values = [l.split() for l in open('1b.gold')]
    for state, value in values:
        goldV[int(state)] = float(value)
    V = submission.policyEvaluation(mdp, V, pi, .0001)
    for state in range(-5,6):
        if not grader.requireIsLessThan(.001, abs(goldV[state] - V[state])):
            print ('   state: {}'.format(state))
grader.addBasicPart('4.1.2', test1b,15)

def test1c():
    V = collections.Counter()  # state -> value of state
    for state in range(-5,6):
        V[state] = state
    mdp = util.NumberLineMDP()
    mdp.computeStates()
    goldPi = collections.defaultdict(lambda: 1)
    pi = submission.computeOptimalPolicy(mdp, V)
    for state in range(-5,6):
        if not grader.requireIsEqual(goldPi[state], pi[state]):
            print ('   state: {}'.format(state))
grader.addBasicPart('4.1.3', test1c,5)

def testIteration(algorithm):
    mdp = util.NumberLineMDP()
    goldPi = collections.defaultdict(lambda: 1)
    goldV = {}
    values = [l.split() for l in open('1d.gold')]
    for state, value in values:
        goldV[int(state)] = float(value)
    algorithm.solve(mdp, .0001)
    for state in range(-5,6):
        if not grader.requireIsEqual(goldPi[state], algorithm.pi[state]):
            print ('   action for state: {}'.format(state))
        if not grader.requireIsLessThan(.001, abs(goldV[state] - algorithm.V[state])):
            print ('   value for state: {}'.format(state))

def test1d():
    testIteration(submission.PolicyIteration())
grader.addBasicPart('4.1.4', test1d,10)

def test1e():
    testIteration(submission.ValueIteration())
grader.addBasicPart('4.1.5', test1e,10)

def test2a():
    mdp1 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=10, peekCost=1)
    startState = mdp1.startState()
    preBustState = (6, None, (1, 1))
    postBustState = (11, None, None)

    mdp2 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=15, peekCost=1)
    preEmptyState = (11, None, (1,0))

    mdp3 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                               threshold=10, peekCost=1)


    tests = [([((1, None, (1, 2)), 0.5, 0), ((5, None, (2, 1)), 0.5, 0)],
              mdp1, startState, 'Take'),
             ([((0, 0, (2, 2)), 0.5, -1), ((0, 1, (2, 2)), 0.5, -1)],
              mdp1, startState, 'Peek'),
             ([((0, None, None), 1, 0)], mdp1, startState, 'Quit'),
             ([((7, None, (0, 1)), 0.5, 0), ((11, None, None), 0.5, 0)],
              mdp1, preBustState, 'Take'),
             ([], mdp1, postBustState, 'Take'),
             ([], mdp1, postBustState, 'Peek'),
             ([], mdp1, postBustState, 'Quit'),
             ([((12, None, None), 1, 12)], mdp2, preEmptyState, 'Take'),
             ([((5, None, (2, 1)), 1, 0)], mdp3, (0, 1, (2, 2)), 'Take')]
    for gold, mdp, state, action in tests:
        if not grader.requireIsEqual(gold,
                                     mdp.succAndProbReward(state, action)):
            print ('   state: {}, action: {}'.format(state, action))
grader.addBasicPart('4.2.1', test2a,15)

def test2b():
    mdp = submission.peekingMDP()
    vi = submission.ValueIteration()
    vi.solve(mdp)
    grader.requireIsEqual(mdp.threshold, 20)
    grader.requireIsEqual(mdp.peekCost, 1)
    f = len([a for a in vi.pi.values() if a == 'Peek']) / float(len(vi.pi.values()))
    grader.requireIsGreaterThanOrEqualTo(.1, f)
grader.addBasicPart('4.2.2', test2b,5)

grader.grade()

