#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator
import matplotlib.pyplot as plt



def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)
    logRegClassifier = {}
    lr_list = [0.5, 0.05, 0.005, 0.0005]
    for lr in lr_list:
        logRegClassifier[lr] = LogisticRegression(data.trainingSet,
                                                  data.validationSet,
                                                  data.testSet,
                                                  learningRate=lr)

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nPerceptron training..")
    myPerceptronClassifier.train()
    print("Done..")

    print("\nLogistic Neuron training...")
    fig,ax = plt.subplots()
    logRegErrors = {}
    logRegEpochs = {}
    for lr in lr_list:
        epochs_range, errors_range = logRegClassifier[lr].train()
        logRegErrors[lr] = errors_range
        logRegEpochs[lr] = epochs_range
        ax.plot(epochs_range, errors_range, label="Learning Rate: "+str(lr))
    print("Done..")
    ax.legend()
    
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    perceptronPred = myPerceptronClassifier.evaluate()
    logRegPreds = {}
    for lr in lr_list:
        logRegPreds[lr] = logRegClassifier[lr].evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, perceptronPred)

    print("\nResult of the Logistic Regression recognizer(s):")
    for lr in lr_list:
        # evaluator.printComparison(data.testSet, logRegPreds[lr])
        print "Leraning Rate:", lr
        accuracy = evaluator.printAccuracy(data.testSet, logRegPreds[lr])
        ax.text(logRegEpochs[lr][-1]-2,logRegErrors[lr][-1]*1.3,
                "acc:"+str(accuracy)+":"+str(lr))
    ax.set_ylim([-50,200])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

if __name__ == '__main__':
    main()
