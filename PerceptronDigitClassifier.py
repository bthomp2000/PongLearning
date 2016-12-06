from random import randint
from copy import deepcopy
import random


K = .2
decay = .4
# 0 is white/gray (foreground), 1 is black
trainingData = list() #[row][col]
testData = list() #[row][col]

weightVectors = [] #An array of pairs (digit, weight vector)
traininglabels = []
classCount = [0] * 10
traininglabels = []
testlabels = []
resultlabels = []
results = [0]*10
testcount = [0.0] * 10
confusion = []
pairs = []
testImages = []



def trainPercepton(numEpochs, usingBias, randomOrder):
    for i in range(0, numEpochs):
        readTrainingInput(usingBias, randomOrder)

def readTrainingInput(usingBias, randomOrder):
    with open('digitdata/traininglabels') as input_file:
        for i, line in enumerate(input_file):
            line = line.rstrip()
            for label in line:
                traininglabels.append(int(label))

    trueNumber = -1
    isFirstIteration = True
    numberCorrect = 0.0
    numberProcessed = 0.0
    with open('digitdata/trainingimages') as input_file:
        imageNum = 0
        count = 0
        trainingData = initArray()
        inputLines = generateLinesArray('digitdata/trainingimages', randomOrder)
        # for intputLineNum, line in enumerate(input_file):
        for intputLineNum in range(0, len(inputLines)):
            currRow = intputLineNum % 28
            imageNum = int(inputLines[intputLineNum][1] / 28)
            line = inputLines[intputLineNum][0]
            if currRow == 0:
                if not isFirstIteration:
                    if processTrainingDigit(trainingData, trueNumber, usingBias):
                        numberCorrect += 1.0
                    numberProcessed += 1.0
                trainingData = initArray()
                trueNumber = traininglabels[imageNum]  # 0-9
                classCount[trueNumber] += 1.0
            for currCol in range(len(line)):
                count+=1
                feature = line[currCol]
                if feature == "#" or feature == "+":
                    trainingData[currRow][currCol] = 0
                elif feature == " ":
                    trainingData[currRow][currCol] = 1
            isFirstIteration = False
        numberProcessed += 1.0
        if processTrainingDigit(trainingData, trueNumber, usingBias): #Process the last digit
            numberCorrect += 1.0
        print("Accuracy on epoch: " + str(numberCorrect) + " / " + str(numberProcessed) + " = " + str(numberCorrect / numberProcessed))


def readTestInput(usingBias):
    with open('digitdata/testlabels') as input_file:
        for i, line in enumerate(input_file):
            line = line.rstrip()
            for label in line:
                testlabels.append(int(label))

    trueNumber = -1
    isFirstIteration = True
    numberCorrect = 0.0
    numberProcessed = 0.0
    with open('digitdata/testimages') as input_file:
        imageNum = 0
        count = 0
        testData = initArray()
        for intputLineNum, line in enumerate(input_file):
            currRow = intputLineNum % 28
            imageNum = int(intputLineNum / 28)
            if currRow == 0:
                if not isFirstIteration:
                    if processTestDigit(testData, trueNumber, usingBias):
                        numberCorrect += 1.0
                    numberProcessed += 1.0
                testData = initArray()
                trueNumber = testlabels[imageNum]  # 0-9
                testcount[trueNumber] += 1.0
                classCount[trueNumber] += 1.0
            for currCol in range(len(line)):
                count+=1
                feature = line[currCol]
                if feature == "#" or feature == "+":
                    testData[currRow][currCol] = 0
                elif feature == " ":
                    testData[currRow][currCol] = 1
            isFirstIteration = False
        numberProcessed += 1.0
        if processTestDigit(testData, trueNumber, usingBias): # Process the last digit
            numberCorrect += 1.0
        print("Accuracy on test set: " + str(numberCorrect) + " / " + str(numberProcessed) + " = " + str(numberCorrect / numberProcessed))


def generateLinesArray(filename, randomSelection):
    if not randomSelection:
        # code adapted from http://cmdlinetips.com/2011/08/three-ways-to-read-a-text-file-line-by-line-in-python/
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        augmentedLines = []
        for i in range(0, len(lines)):
            augmentedLines.append((lines[i], i))
        return augmentedLines
    else:
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        remainingDigits = []
        for i in range(0, int(len(lines) / 28)):
            remainingDigits.append(i)
        random.shuffle(remainingDigits)
        shuffledLines = []
        while remainingDigits:
            nextDigit = remainingDigits.pop(0)
            startingLine = nextDigit * 28
            for i in range(startingLine, startingLine + 28):
                shuffledLines.append((lines[i], i))
        return shuffledLines


def processTrainingDigit(trainingData, trueNumber, usingBias):
    chosenBestDigit = chooseBestDigit(trainingData, usingBias)
    if chosenBestDigit[0] != trueNumber:
        # lower score of wrong answer, raise score of right answer
        # weight of incorrect = weight of incorrect - f(x)
        # weight of correct = weight of correct + f(x)
        augmentWeightVector(chosenBestDigit[0], trainingData, False, usingBias)
        augmentWeightVector(trueNumber, trainingData, True, usingBias)
        # print("Incorrect. Chosen " + str(chosenBestDigit[0]) + ", correct was " + str(trueNumber))
        return False
    else:
        # print("Correct on number " + str(chosenBestDigit[0]))
        return True

def processTestDigit(testData, trueNumber, usingBias):
    chosenBestDigit = chooseBestDigit(testData, usingBias)
    confusion[chosenBestDigit[0]][trueNumber] += 1.0
    if chosenBestDigit[0] != trueNumber:
        # on test data, don't update weight vectors
        print("Incorrect. Chosen " + str(chosenBestDigit[0]) + ", correct was " + str(trueNumber))
        return False
    else:
        print("Correct on number " + str(chosenBestDigit[0]))
        return True


def augmentWeightVector(digitNumber, dataVector, shouldIncrement, usingBias):
    for i in range(0, 28):
        for j in range(0, 28):
            if shouldIncrement:
                weightVectors[digitNumber][i][j] += dataVector[i][j] * decay
            else:
                weightVectors[digitNumber][i][j] -= dataVector[i][j] * decay
        if usingBias:
            weightVectors[digitNumber][i][28] += 1


#randomInitialValues ia bool. If true, then weight vectors initialized randomly
def initializeWeightVectors(randomInitialValues):
    for digit in range (0, 10):
        digitWeight = [[] for i in range (0, 28)]
        for i in range(0, 28):
            digitWeight[i] = [0] * 29
            if randomInitialValues:
                for j in range(0, 28):
                    digitWeight[i][j] = randint(-10, 10)
        weightVectors.append(digitWeight)


def dotProductDigitVectors(testDigitVector, currentDigitVector, usingBias):
    currentTotal = 0
    for i in range(0, 28):
        for j in range(0, 28):
            currentTotal += testDigitVector[i][j] * currentDigitVector[i][j]
        if usingBias:
            currentTotal += currentDigitVector[i][28]
    return currentTotal


#testDigitVector is an array of size [28][28] of the features of a digit to be checked
def chooseBestDigit(testDigitVector, usingBias):
    bestDigit = -1
    bestDigitValue = -99999999
    for currentDigit in range(0,10):
        currentDigitValue = dotProductDigitVectors(testDigitVector, weightVectors[currentDigit], usingBias)
        if currentDigitValue > bestDigitValue:
            bestDigit = currentDigit
            bestDigitValue = currentDigitValue
    return (bestDigit, bestDigitValue)


def initArray():
    colors = K * 2
    twodim = list()
    for i in range(28):
        temp = list()
        for j in range(28):
            temp.append(colors)
        twodim.append(deepcopy(temp))
    return deepcopy(twodim)

def initializeConfusionMatrix():
    for i in range(10):
        temp = [0.0] * 10
        confusion.append(deepcopy(temp))

def initializeDataStructures():
    initializeWeightVectors(False)
    initializeConfusionMatrix()

def printConfusionMatrix():
    for i in range(len(testcount)):
        for j in range(10):
            confusion[i][j] /= testcount[i]
    for i in range(10):
        for j in range(10):
            value = confusion[j][i]
            value = format(value, '2.3f')
            print(value, end=" ")
        print('')


initializeDataStructures()
trainPercepton(10, True, True)
readTestInput(True)
printConfusionMatrix()
