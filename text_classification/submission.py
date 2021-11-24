"""
Text classification
"""

import util
import operator
from collections import Counter

class Classifier(object):
    def __init__(self, labels):
        """
        @param (string, string): Pair of positive, negative labels
        @return string y: either the positive or negative label
        """
        self.labels = labels

    def classify(self, text):
        """
        @param string text: e.g. email
        @return double y: classification score; >= 0 if positive label
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, text):
        """
        @param string text: the text message
        @return string y: either 'ham' or 'spam'
        """
        if self.classify(text) >= 0.:
            return self.labels[0]
        else:
            return self.labels[1]

class RuleBasedClassifier(Classifier):
    def __init__(self, labels, blacklist, n=1, k=-1):
        """
        @param (string, string): Pair of positive, negative labels
        @param list string: Blacklisted words
        @param int n: threshold of blacklisted words before email marked spam
        @param int k: number of words in the blacklist to consider
        """
        super(RuleBasedClassifier, self).__init__(labels)
        # BEGIN_YOUR_CODE (around 3 lines of code expected) 
        self.blackSet = set()
        
        lenBlackList = k 
        if k == -1 or k > len(blacklist):
            lenBlackList = len(blacklist)

        for i in range(lenBlackList):
            self.blackSet.add(blacklist[i])
        self.n = n
        # END_YOUR_CODE

    def classify(self, text):
        """
        @param string text: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        words = text.split()
        cnt = 0

        for word in words:
            if word in self.blackSet:
                cnt += 1
            if cnt > self.n:
                return -1
        return 1
        # END_YOUR_CODE

def extractUnigramFeatures(x):
    """
    Extract unigram features for a text document $x$. 
    @param string x: represents the contents of an text message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    featureVec = Counter()
    words = x.split()
    for word in words:
        featureVec[word] += 1
    return featureVec
    # END_YOUR_CODE


class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
        @param dict params: the parameter weights used to predict
        """
        super(WeightedClassifier, self).__init__(labels)
        self.featureFunction = featureFunction
        self.params = params

    def classify(self, x):
        """
        @param string x: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        total = 0
        feature = self.featureFunction(x)
        for featureElement in feature:
            if featureElement in self.params:
                total += self.params[featureElement] * feature[featureElement]
        if total >= 0:
            return 1
        else:
            return -1
        # END_YOUR_CODE

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('pos', 'neg'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    w = Counter()
    featureList = list()

    for x, y in trainExamples:
        featureList.append(featureExtractor(x))

    for i in range(iters):
        for j in range(len(trainExamples)):
            features = featureList[j]
            y = trainExamples[j][1]
            wTotal = 0
            for feature in features:
                # Calculating our expected wTotal according to our w record
                if feature in w:
                    wTotal += w[feature] * features[feature]
            # We adjust our w if we have the wrong prediction
            # The answer is "ham" but we predict "spam" => loosing w by using the feature vector of current x
            if wTotal >= 0 and y == labels[1]:
                for feature in features:
                    w[feature] -= features[feature]
            # The answer is "spam" but we predict "ham" => tighten w by using the feature vector of current x
            if wTotal < 0 and y == labels[0]:
                for feature in features:
                    w[feature] += features[feature]
    return w
    # END_YOUR_CODE

def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$. 

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    result = Counter()
    # Append a signal at the very first of the sentence
    words = ["."] + x.split()
    puncuation = [".", "!", "?"]

    for i in range(len(words)-1):
        if words[i+1] not in puncuation:
            result[words[i+1]] += 1
            if words[i] not in puncuation:
                result[words[i] + " " + words[i+1]] += 1
            else:
                result["-BEGIN-" + " " + words[i+1]] += 1
    return result
    # END_YOUR_CODE

class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); each classifier is a WeightedClassifier that detects label vs NOT-label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self.classifiers = {}
        for label, classifier in classifiers:
            self.classifiers[label] = classifier    
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        score = self.classify(x)
        label = score[0][0]
        bestScore = score[0][1]
        for i in range(1, len(score)):
            if score[i][1] > bestScore:
                label = score[i][0]
                bestScore = score[i][1]
        return label
        # END_YOUR_CODE

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        result = list()

        for label, _ in self.classifiers:
            result.append(label, self.classifiers[label].classify(x))
        return result
        # END_YOUR_CODE

def learnOneVsAllClassifiers(trainExamples, featureFunction, labels, perClassifierIters = 10):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label (an entry from the list of labels)
    @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each classifier
    @return list (label, Classifier)
    """
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
    result = list()

    for label in labels:
        newTrainingData = list()
        newLabel = ("pos", "neg")
        for x, y in trainExamples:
            if y == label:
                newTrainingData.append((x, "pos"))
            else:
                newTrainingData.append((x, "neg"))
            result.append((label, WeightedClassifier(newLabel, featureFunction, learnWeightsFromPerceptron(newTrainingData, featureFunction, newLabel, perClassifierIters))))
    return result
    # END_YOUR_CODE
