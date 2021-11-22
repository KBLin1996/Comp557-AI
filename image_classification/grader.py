#!/usr/bin/env python
"""
Grader for template assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""


import random
import numpy as np

import graderUtil
grader = graderUtil.Grader()
submission = grader.load('submission')



############################################################
# Problem 4.1: runKMeans

# Use lambda for short tests
def testKMeansBasic():
    np.random.seed(5)
    x = np.array([[1., 1., 2.],[-1., -1., 2.],[-2., -1., 1.]])
    ans1 = np.array([[2., 1.],[ 2., -1.],[1., -1.5]])
    ans2 = submission.runKMeans(2,x,5)
    n1 = np.sum(ans1**2)
    n2 = np.sum(ans2**2)
    grader.requireIsEqual(n1, n2)
grader.addPart('4.1', testKMeansBasic, 10)


############################################################
# Problem 4.2: extractFeatures

patches = np.array([[1.,1.5,3.,-2.5],[2.,2.5,1.,-1.],[3.,3.5,1.,4.]])
centroids = np.array([[2.0,2.3,2.5],[-1.0,-2.5,1.0],[1.5,-3.,-1.5]])
features = np.array([[1.81983758, 0., 0.47215773],[1.81019111, 0., 0.5571374],
       [1.10930137, 0., 0.85107947],[2.0069789, 0., 0.]])
grader.addBasicPart('4.2', lambda : grader.requireIsEqual(features,submission.extractFeatures(patches,centroids)),10)



############################################################
# Problem 4.3: Supervised Training


theta = np.array([1.5,2.5])
fv1 = np.array([1.0,1.0])
fv2 = np.array([0.2,0.2])
grader.addBasicPart('4.3.1.1', lambda : grader.requireIsEqual(np.array([-0.01798621, -0.01798621]), submission.logisticGradient(theta,fv1, 1)),3)
grader.addBasicPart('4.3.1.2', lambda : grader.requireIsEqual(np.array([-0.0620051, -0.0620051]), submission.logisticGradient(theta,fv2, 1)),4)
grader.addBasicPart('4.3.1.3', lambda : grader.requireIsEqual(np.array([0.1379949, 0.1379949]), submission.logisticGradient(theta,fv2, 0)),3)



theta = np.array([1.5,2.5])
fv1 = np.array([1.0,1.0])
fv2 = np.array([0.2,0.2])
grader.addBasicPart('4.3.2.1', lambda : grader.requireIsEqual(np.array([0.,0.]), submission.hingeLossGradient(theta,fv1, 1)),3)
grader.addBasicPart('4.3.2.2', lambda : grader.requireIsEqual(-fv2, submission.hingeLossGradient(theta,fv2, 1)),4)
grader.addBasicPart('4.3.2.3', lambda : grader.requireIsEqual(fv2, submission.hingeLossGradient(theta,fv2, 0)),3)


grader.grade()
