#!/usr/bin/env python
from __future__ import print_function

"""
Grader for template assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""


import random

import graderUtil
import util
import collections
import copy
grader = graderUtil.Grader()
submission = grader.load('submission')


############################################################
# Manual problems

# .addBasicPart is a basic test and always run.
grader.addBasicPart('writeupValid', lambda : grader.requireIsValidPdf('writeup.pdf'),0)


############################################################
# Problem 3.1a: N-Queens

def test31a():
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(8))
    grader.requireIsEqual(1.0, nQueensSolver.optimalWeight)
    grader.requireIsEqual(92, nQueensSolver.numOptimalAssignments)
    grader.requireIsEqual(2057, nQueensSolver.numOperations)

grader.addBasicPart('3.1a-0', test31a,5)

############################################################
# Problem 3.1b: Most constrained variable

def test31b():
    mcvSolver = submission.BacktrackingSearch()
    mcvSolver.solve(submission.create_nqueens_csp(8), mcv = True)
    grader.requireIsEqual(1.0, mcvSolver.optimalWeight)
    grader.requireIsEqual(92, mcvSolver.numOptimalAssignments)
    grader.requireIsLessThan(1500, mcvSolver.numOperations)

grader.addBasicPart('3.1b-0', test31b,10)

############################################################
# Problem 3.1c: Least constraining value

def test31c():
    lcvSolver = submission.BacktrackingSearch()
    lcvSolver.solve(submission.create_nqueens_csp(8), lcv = True)
    grader.requireIsEqual(92, lcvSolver.numOptimalAssignments)
    grader.requireIsLessThan(60, lcvSolver.firstAssignmentNumOperations)

grader.addBasicPart('3.1c', test31c,10)

############################################################
# Problem 3.1d: Arc consistency

def test31d_0():
    acSolver = submission.BacktrackingSearch()
    acSolver.solve(submission.create_nqueens_csp(8), mac = True)
    grader.requireIsEqual(92, acSolver.numOptimalAssignments)
    grader.requireIsLessThan(22, acSolver.firstAssignmentNumOperations)
    grader.requireIsLessThan(1000, acSolver.numOperations)

grader.addBasicPart('3.1d-0', test31d_0,7)

def test31d_1():
    acSolver = submission.BacktrackingSearch()
    acSolver.solve(submission.create_nqueens_csp(8), mcv = True, lcv = True, mac = True)
    grader.requireIsEqual(92, acSolver.numOptimalAssignments)
    grader.requireIsLessThan(20, acSolver.firstAssignmentNumOperations)
    grader.requireIsLessThan(1000, acSolver.numOperations)

grader.addBasicPart('3.1d-1', test31d_1,8)


############################################################
# Problem 3.2b: Sum potential

def test32b_0():
    csp = util.CSP()
    csp.add_variable('A', [0, 1, 2, 3])
    csp.add_variable('B', [0, 6, 7])
    csp.add_variable('C', [0, 5])

    sumVar = submission.get_sum_variable(csp, 'sum-up-to-15', ['A', 'B', 'C'], 15)
    csp.add_unary_potential(sumVar, lambda n: n in [12, 13])
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    grader.requireIsEqual(4, sumSolver.numOptimalAssignments)

    csp.add_unary_potential(sumVar, lambda n: n == 12)
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    grader.requireIsEqual(2, sumSolver.numOptimalAssignments)

grader.addBasicPart('3.2b-0', test32b_0,10)


def verify_schedule(bulletin, profile, schedule, checkUnits = True):
    """
    Returns true if the schedule satisifies all requirements given by the profile.
    """
    goodSchedule = True
    all_courses_taking = dict((s[1], s[0]) for s in schedule)

    # No course can be taken twice.
    goodSchedule *= len(all_courses_taking) == len(schedule)
    if not goodSchedule:
        print ('course repeated')
        return False

    # Each course must be offered in that semester.
    goodSchedule *= all(bulletin.courses[s[1]].is_offered_in(s[0]) for s in schedule)
    if not goodSchedule:
        print ('course not offered')
        return False

    # If specified, only take the course at the requested time.
    for req in profile.requests:
        if len(req.semesters) == 0: continue
        goodSchedule *= all([s[0] in req.semesters for s in schedule if s[1] in req.cids])
    if not goodSchedule:
        print ('course taken at wrong time')
        return False

    # If a request has multiple courses, at most one is chosen.
    for req in profile.requests:
        if len(req.cids) == 1: continue
        goodSchedule *= len([s for s in schedule if s[1] in req.cids]) <= 1
    if not goodSchedule:
        print ('more than one exclusive group of courses is taken')
        return False

    # Must take a course after the prereqs
    for req in profile.requests:
        if len(req.prereqs) == 0: continue
        cids = [s for s in schedule if s[1] in req.cids] # either empty or 1 element
        if len(cids) == 0: continue
        semester, cid, units = cids[0]
        for prereq in req.prereqs:
            if prereq in profile.taking:
                goodSchedule *= prereq in all_courses_taking
                if not goodSchedule:
                    print ('not all prereqs are taken')
                    return False
                goodSchedule *= profile.semesters.index(semester) > \
                    profile.semesters.index(all_courses_taking[prereq])
    if not goodSchedule:
        print ('course is taken before prereq')
        return False

    if not checkUnits: return goodSchedule
    # Check for unit loads
    unitCounters = collections.Counter()
    for semester, c, units in schedule:
        unitCounters[semester] += units
    goodSchedule *= all(profile.minUnits <= u and u <= profile.maxUnits \
        for k, u in unitCounters.items())
    if not goodSchedule:
        print ('unit count out of bound for semester')
        return False

    return goodSchedule

# Load all courses.
bulletin = util.CourseBulletin('rice_cs_courses.json')

############################################################
# Problem 3.3a: Semester specification

def test33a_0():
    profile = util.Profile(bulletin, 'profile3a_rice.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_semester_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    grader.requireIsEqual(2, alg.numOptimalAssignments)
    #solution = util.extract_course_scheduling_solution(profile, alg.optimalAssignment)
    for assignment in alg.allAssignments:
        solution = util.extract_course_scheduling_solution(profile, assignment)
        grader.requireIsTrue(verify_schedule(bulletin, profile, solution, False))

grader.addBasicPart('3.3a-0', test33a_0,10)

############################################################
# Problem 3.3b: Weighting

def test33b_0():
    profile = util.Profile(bulletin, 'profile3b_rice.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_request_weights(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    grader.requireIsEqual(1, alg.numOptimalAssignments)
    grader.requireIsEqual(2, alg.numAssignments)
    grader.requireIsEqual(5, alg.optimalWeight)
    for assignment in alg.allAssignments:
        solution = util.extract_course_scheduling_solution(profile, assignment)
        grader.requireIsTrue(verify_schedule(bulletin, profile, solution, False))

grader.addBasicPart('3.3b-0', test33b_0,10)

############################################################
# Problem 3.3c: Prerequisites

def test33c_0():
    profile = util.Profile(bulletin, 'profile3h_rice.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_prereq_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    #import pdb; pdb.set_trace()
    grader.requireIsEqual(3, alg.numOptimalAssignments)
    for assignment in alg.allAssignments:
        solution = util.extract_course_scheduling_solution(profile, assignment)
        grader.requireIsTrue(verify_schedule(bulletin, profile, solution, False))

grader.addBasicPart('3.3c-0', test33c_0,10)

############################################################
# Problem 3.3d: Credit load

def test33d_0():
    profile = util.Profile(bulletin, 'profile3d_rice.txt')
    cspConstructor = submission.SchedulingCSPConstructor(bulletin, copy.deepcopy(profile))
    csp = cspConstructor.get_basic_csp()
    cspConstructor.add_credit_constraints(csp)
    alg = submission.BacktrackingSearch()
    alg.solve(csp)

    # Verify correctness.
    grader.requireIsEqual(6, alg.numOptimalAssignments)
    for assignment in alg.allAssignments:
        solution = util.extract_course_scheduling_solution(profile, assignment)
        grader.requireIsTrue(verify_schedule(bulletin, profile, solution))

grader.addBasicPart('3.3d-0', test33d_0,10)

############################################################
# Check that profile.txt is valid.

def valid_profile_txt():
    try:
        profile = util.Profile(bulletin, 'profile.txt')
    except:
        grader.fail('profile.txt is not valid')
    grader.assignFullCredit()

grader.addBasicPart('profile.txt', valid_profile_txt, 10)

grader.grade()
