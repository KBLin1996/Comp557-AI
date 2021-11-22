import graderUtil
import util
import time

if __name__ == "__main__":

    grader = graderUtil.Grader()
    submission = grader.load('submission')

    TRAIN_PATH_SPAM = 'data/spam-classification/train'
    TRAIN_PATH_SENTIMENT = 'data/sentiment/train'
    TRAIN_PATH_TOPICS = 'data/topics/train'
    TRAIN_SIZE = 5000


    ############################################################
    # Problem 2.1: Spam Classification
    ############################################################
    # Load training examples
    trainSpamExamples, devSpamExamples = util.holdoutExamples(util.loadExamples(TRAIN_PATH_SPAM)[:TRAIN_SIZE])


    ############################################################
    # -- 2.1.1a: RuleBasedClassifier
    def test1_1a_0():
        blacklist = util.loadBlacklist()
        classifier = submission.RuleBasedClassifier( 
                util.LABELS_SPAM, blacklist )
        for (ex, _) in trainSpamExamples[:10]:
            grader.requireIsOneOf( util.LABELS_SPAM, classifier.classifyWithLabel(ex) )
        grader.requireIsEqual( "spam", classifier.classifyWithLabel(" ".join(blacklist)) )
        grader.requireIsEqual( "ham", classifier.classifyWithLabel("") )
    grader.addBasicPart('2.1.1a-0', test1_1a_0, 1)

    def test1_1a_1():
        blacklist = util.loadBlacklist()
        classifier = submission.RuleBasedClassifier( 
                util.LABELS_SPAM, blacklist )
        trainErr = util.computeErrorRate( trainSpamExamples, classifier )
        devErr = util.computeErrorRate( devSpamExamples, classifier )
        print ("trainErr: ", trainErr)
        print ("devErr: ", devErr)
        grader.requireIsLessThan( 0.50, trainErr ) # We got 0.487
        grader.requireIsLessThan( 0.50, devErr ) # We got 0.4807

    grader.addPart('2.1.1a-1', test1_1a_1, 1)

    ############################################################
    # -- 2.1.1b: Experiment with thresholds n and k
    grader.addManualPart('2.1.1b-0', 1)

    ############################################################
    # -- 2.1.2a: extractUnigramFeatures
    def test_1_2a():
        sentence = "The quick dog chased the lazy fox over the brown fence"
        features = {'brown': 1, 'lazy': 1, 'fence': 1, 'fox': 1, 'over':
                1, 'chased': 1, 'dog': 1, 'quick': 1, 'the': 2, 'The':
                1}
        grader.requireIsEqual(features, submission.extractUnigramFeatures(sentence))
    grader.addBasicPart('2.1.2a', test_1_2a, 2)

    ############################################################
    # -- 2.1.2b: WeightedClassifier.classify
    def test1_2b():
        weights = {'quick': 1.0, 'fox': -1.0}
        classifier = submission.WeightedClassifier(util.LABELS_SPAM, 
                submission.extractUnigramFeatures, weights)
        grader.requireIsEqual(
                "ham",
                classifier.classifyWithLabel( "The quick dog chased")) 
        grader.requireIsEqual( 
                "spam",
                classifier.classifyWithLabel("The dog chased the lazy fox over the brown fence"))
    grader.addBasicPart('2.1.2b', test1_2b, 2)

    ############################################################
    # -- 2.1.3a: learnWeightsFromPerceptron
    def test1_3a_0():
        weights = submission.learnWeightsFromPerceptron(
                trainSpamExamples[:100], 
                submission.extractUnigramFeatures,
                util.LABELS_SPAM,
                3)
        grader.requireIsTrue( isinstance(weights, dict) )
        for (k,v) in weights.items(): grader.requireIsNumeric(v)
    grader.addBasicPart('2.1.3a-0', test1_3a_0, 0)

    def test1_3a_1():
        weights = submission.learnWeightsFromPerceptron(
                trainSpamExamples, 
                submission.extractUnigramFeatures,
                util.LABELS_SPAM,
                10)
        classifier = submission.WeightedClassifier(util.LABELS_SPAM, 
                submission.extractUnigramFeatures, weights)
        trainErr = util.computeErrorRate( trainSpamExamples, classifier )
        devErr = util.computeErrorRate( devSpamExamples, classifier )
        print ("trainErr: ", trainErr)
        print ("devErr: ", devErr)


        grader.requireIsLessThan( 0.02, trainErr )
        grader.requireIsLessThan( 0.06, devErr )

    grader.addPart('2.1.3a-1', test1_3a_1, 6, 45)

    ############################################################
    # -- 2.1.3b:  extractBigramFeatures
    def test_1_3b_0():
        sentence = "The quick dog chased the lazy fox over the brown fence"
        features = {'the': 2, 'over': 1, 'brown': 1, 'lazy': 1, 'fox': 1, 'fence': 1, 'brown fence': 1, 'chased the': 1, 'quick dog': 1, 'fox over': 1, 'chased': 1, 'dog': 1, 'lazy fox': 1, 'The quick': 1, 'the lazy': 1, '-BEGIN- The': 1, 'quick': 1, 'The': 1, 'over the': 1, 'dog chased': 1, 'the brown': 1}
        grader.requireIsEqual(features, submission.extractBigramFeatures(sentence))
    grader.addBasicPart('2.1.3b-0', test_1_3b_0, 2)


    ############################################################
    # -- 2.1.3c: Vary number of examples with bigram features
    grader.addManualPart('2.1.3c', 5)

    ############################################################
    # Problem 2.2: Sentiment Classification
    ############################################################
    # Load training examples
    trainSentimentExamples, devSentimentExamples = util.holdoutExamples(util.loadExamples(TRAIN_PATH_SENTIMENT)[:TRAIN_SIZE])


    ############################################################
    # -- 2a: Use perceptron for sentiment classification
    grader.addManualPart('2.2a', 3)


    ############################################################
    # -- 2b: Vary number of iterations
    grader.addManualPart('2.2b', 5)

    ############################################################
    # Problem 2.3: Document Categorization
    ############################################################
    
    # Load training examples
    trainDocumentExamples, devDocumentExamples = util.holdoutExamples(util.loadExamples(TRAIN_PATH_TOPICS)[:TRAIN_SIZE])


    ############################################################
    # -- 3a: Implement OneVsAll
    def test_3a_0():
        labels = ["A", "B", "C"]
        weightsA = {'quick': 1, 'lazy' : 1, 'dog' : -1, 'field' : -1 }
        weightsB = {'dog': 1, 'fox' : 1, 'quick' : -1, 'fence' : -1 }
        weightsC = {'dog': -1, 'quick' : -1, 'fence' : 1, 'field' : 1 }
        classifierA = submission.WeightedClassifier( ["A", "!A"], 
                submission.extractUnigramFeatures, weightsA )
        classifierB = submission.WeightedClassifier( ["B", "!B"], 
                submission.extractUnigramFeatures, weightsB)
        classifierC = submission.WeightedClassifier( ["C", "!C"], 
                submission.extractUnigramFeatures, weightsC)
        classifier = submission.OneVsAllClassifier( labels,
                zip( labels, [classifierA, classifierB, classifierC] ) )
        grader.requireIsEqual( "A",
                classifier.classifyWithLabel( "The quick dog was lazy") ) # 1, 0, -2
        grader.requireIsEqual( "B",
                classifier.classifyWithLabel( "The dog was quick unlike the fox") ) # 0, 1, -2
        grader.requireIsEqual( "C",
                classifier.classifyWithLabel( "The dog jumped over the fence and on to the field") ) # -2, 0, 1
    grader.addBasicPart('2.3a-0', test_3a_0, 9)

    ############################################################
    # -- 2b: Report accuracies for one-vs-all
    grader.addManualPart('2.3b-0', 3)



    grader.grade()

