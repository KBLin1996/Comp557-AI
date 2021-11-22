import os, random, operator
from collections import Counter

LABELS_SPAM = ('ham', 'spam')
LABELS_SENTIMENT = ('pos', 'neg')
LABELS_TOPICS = ( 'comp', 'rec', 'talk', 'god', 'sci' )

def loadExamples(path):
    """
    Reads examples from disk.
    @param string path: a directory containing subdirectories, one for each
    label.  Each file in each of those directories is an example:
        <path>/<label>/<example1>
    @return list of examples: (x, y) pairs, where x is an email (string) and y is
    a label (string).
    """
    examples = []
    for label in os.listdir(path):
#        print (label)
        i = 0
        for emailFile in os.listdir(os.path.join(path, label)):
            x = open(os.path.join(path, label, emailFile),encoding="latin-1").read()
            y = label
            examples.append((x, y))
            i += 1
#            if i%100 == 0: print i
    # Randomly shuffle the examples
    random.seed(41)
    random.shuffle(examples)
    return examples

def loadBlacklist(path="data/spam-classification/blacklist.txt"):
    """
    Loads the blacklisted words from path
    @param string path: a file containing blacklisted words
    @return list of blacklisted words
    """
    return [ word.strip() for word in open(path,encoding="latin-1").readlines() ]

def holdoutExamples(examples, frac=0.2):
    """
    @param list examples
    @param float frac: fraction of examples to holdout.
    @return (examples1, examples2): two lists of examples.
    """
    examples1 = []
    examples2 = []
    random.seed(42)
    for ex in examples:
        if random.random() < frac:
            examples2.append(ex)
        else:
            examples1.append(ex)
    return (examples1, examples2)

def computeConfusionMatrix(examples, classifier): 
    """
    @param list examples
    @param Classifier classifier: 
    @return float[][]: confusion matrix; rows are true labels, columns are predicted
    """
    # First extract all keys
    keys = set([])
    for _, y in examples:
        keys.add(y)

    confusion = {}
    for y in keys:
        confusion[y] = dict( ( (y_, 0) for y_ in keys ) )
    for x, y in examples:
        y_ = classifier.classifyWithLabel(x) 
        confusion[y][y_] = confusion[y].get( y_, 0 ) + 1
    return confusion

def printConfusionMatrix(confusion): 
    """
    @param list examples
    @param Classifier classifier: 
    @return float[][]: confusion matrix; rows are true labels, columns are predicted
    """
    print ("\t" + "\t".join(confusion.keys()))
    for key in confusion.keys():
            print (key + "\t" + "\t".join(map(str, confusion[key].values())))

def computeErrorRate(examples, classifier): 
    """
    @param list examples
    @param dict params: parameters
    @param function predict: (params, x) => y
    @return float errorRate: fraction of examples we make a mistake on.
    """
    numErrors = 0
    for x, y in examples:
        if classifier.classifyWithLabel(x) != y:
            numErrors += 1
    return 1.0 * numErrors / len(examples)

def readParameters(path):
    """
    @param string path: location to read the parameters (tab separated file)
    @return dict: parameters
    """
    params = {}
    for line in open(path):
        feature, value = line.split("\t")
        params[feature] = float(value)
    return params

def writeParameters(params, path):
    """
    @param dict params: parameters to write
    @param string path: location to write the parameters to (tab separated file)
    """
    for feature, value in sorted(params.items(), key=operator.itemgetter(1)):
      print (feature + "\t" + str(value),file=open(path,"a"))
#    out.close()

