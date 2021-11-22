import sys, operator
import util, submission

# Main entry point to test your spam classifier.

TRAIN_PATH_SPAM = 'data/spam-classification/train'
TRAIN_PATH_SENTIMENT = 'data/sentiment/train'
TRAIN_PATH_TOPICS = 'data/topics/train'

def evaluateClassifier(trainExamples, devExamples, classifier):
    util.printConfusionMatrix(util.computeConfusionMatrix(trainExamples, classifier))
    trainErrorRate = util.computeErrorRate(trainExamples, classifier) 
    print ('trainErrorRate: %f' % trainErrorRate)
    util.printConfusionMatrix(util.computeConfusionMatrix(devExamples, classifier))
    devErrorRate = util.computeErrorRate(devExamples, classifier) 
    print ('devErrorRate: %f' % devErrorRate)

def part1_1(args):
    print ("Part 2.1.1 RuleBasedClassifier")

    examples = util.loadExamples(TRAIN_PATH_SPAM)[:args.examples]
    labels = util.LABELS_SPAM
    trainExamples, devExamples = util.holdoutExamples(examples)
    classifier = submission.RuleBasedClassifier( 
            labels, util.loadBlacklist(), args.n, args.k)

    evaluateClassifier(trainExamples, devExamples, classifier)

def part1_3(args):
    print ("Part 2.1.3 learnWeightsFromPerceptron")

    examples = util.loadExamples(TRAIN_PATH_SPAM)[:args.examples]
    labels = util.LABELS_SPAM
    trainExamples, devExamples = util.holdoutExamples(examples)
    weights = submission.learnWeightsFromPerceptron(trainExamples, submission.extractUnigramFeatures, labels, args.iters)
    classifier = submission.WeightedClassifier(labels, submission.extractUnigramFeatures, weights)

    evaluateClassifier(trainExamples, devExamples, classifier)

    print ("The bigram feature extractor")
    print (submission.extractBigramFeatures('The quick dog chased the lazy fox over the brown fence.'))

    print ("Varying the number of examples")
    for i in range(500,5500,500):
        weights = submission.learnWeightsFromPerceptron(trainExamples[:i], submission.extractUnigramFeatures, labels, args.iters)
        classifier = submission.WeightedClassifier(labels, submission.extractUnigramFeatures, weights)
        
        evaluateClassifier(trainExamples[:i], devExamples, classifier)

def part2(args):
    print ("Part 2.2 Sentiment Analysis")

    examples = util.loadExamples(TRAIN_PATH_SENTIMENT)[:args.examples]
    labels = util.LABELS_SENTIMENT
    trainExamples, devExamples = util.holdoutExamples(examples)
    weights = submission.learnWeightsFromPerceptron(trainExamples, submission.extractUnigramFeatures, labels, args.iters)
    classifier = submission.WeightedClassifier(labels, submission.extractUnigramFeatures, weights)
    weights = submission.learnWeightsFromPerceptron(trainExamples, submission.extractBigramFeatures, labels, args.iters)
    classifier = submission.WeightedClassifier(labels, submission.extractBigramFeatures, weights)
    evaluateClassifier(trainExamples, devExamples, classifier)
    for i in range(1,21):
        print ("Iters = ",i)
        weights = submission.learnWeightsFromPerceptron(trainExamples, submission.extractBigramFeatures, labels, i)
        classifier = submission.WeightedClassifier(labels, submission.extractBigramFeatures, weights)
        evaluateClassifier(trainExamples, devExamples, classifier)
        
def part3(args):
    print ("Part 2.3 Topic Classification")
    examples = util.loadExamples(TRAIN_PATH_TOPICS)[:args.examples]
    labels = util.LABELS_TOPICS
    trainExamples, devExamples = util.holdoutExamples(examples)

    classifiers = submission.learnOneVsAllClassifiers( trainExamples, submission.extractBigramFeatures, labels, 10 )
    classifier = submission.OneVsAllClassifier(labels, classifiers)

    evaluateClassifier(trainExamples, devExamples, classifier)
   

def main():
    import argparse
    parser = argparse.ArgumentParser( description='Spam classifier' )
    parser.add_argument('--examples', type=int, default=10000, help="Maximum number of examples to use" )
    subparsers = parser.add_subparsers()

    # Part 2.1.1
    parser1_1 = subparsers.add_parser('part2.1.1', help = "Part 2.1.1")
    parser1_1.add_argument('-n', type=int, default="1", help="Number of words to consider" )
    parser1_1.add_argument( '-k', type=int, default="-1", help="Number of words in blacklist to choose" )
    parser1_1.set_defaults(func=part1_1)

    # Part 2.1.3
    parser1_3 = subparsers.add_parser('part2.1.3', help = "Part 2.1.3")
    parser1_3.add_argument('--iters', type=int, default="20", help="Number of iterations to run perceptron" )
    parser1_3.set_defaults(func=part1_3)

    # Part 2.2
    parser2 = subparsers.add_parser('part2.2', help = "Part 2.2")
    parser2.add_argument('--iters', type=int, default="20", help="Number of iterations to run perceptron" )
    parser2.set_defaults(func=part2)

    # Part 2.3
    parser3 = subparsers.add_parser('part2.3', help = "Part 2.3")
    parser3.add_argument('--iters', type=int, default="20", help="Number of iterations to run perceptron" )
    parser3.set_defaults(func=part3)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

