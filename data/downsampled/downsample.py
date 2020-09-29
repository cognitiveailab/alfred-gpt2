import random

def loadData(filename):
    print(" * loadData(): filename = " + filename)

    tasks = []

    numLines = 0
    with open(filename) as fp:

        lastCommandSeq = ""
        lastLines = []
        for line in fp:
            
            fields = line.split("[SEP]")
            taskDesc = fields[0]
            commandSeq = fields[1]

            print(taskDesc)
            print(commandSeq)
            print(line)

            if (commandSeq != lastCommandSeq):
                # Store
                tasks.append(lastLines)
                lastLines = []

            lastLines.append(line)
            lastCommandSeq = commandSeq

            numLines += 1

    print("Number of lines: " + str(numLines))
    print("Number of tasks: " + str(len(tasks)))

    return tasks


def saveData(dataIn, filenameOut):
    print(" * saveData(): filename = " + filenameOut)

    numLinesOut = 0
    with open(filenameOut, 'w') as outFile:
        for task in dataIn:
            for line in task:
                outFile.write(line)
                numLinesOut += 1

    print(" * saveData(): Wrote " + str(len(dataIn)) + " tasks / " + str(numLinesOut) + " lines.")



def getRandomSubsample(dataIn, subsampleProportion=1.00):
    print(" * getRandomSubsample(): started... (subsamplePropotion = " + str(subsampleProportion) + ")")
    out = []

    numSamples = int(len(dataIn) * subsampleProportion)
    print("numSamples: " + str(numSamples))
    indicies = random.sample(range(len(dataIn)), numSamples)

    for index in indicies:
        out.append(dataIn[index])

    return out



#
# Main
#

print("Initializing...")

filenameIn = "alfred.train.gpt2.txt"

subsampleProportion = 0.02
numSubsamples = 10

# Load original data
data = loadData(filenameIn)

for subsampleIdx in range(0, numSubsamples):

    # Subsample it
    subsampledData = getRandomSubsample(data, subsampleProportion)

    # Save the subsampled data
    filenameOut = filenameIn + ".subsampled" + str(subsampleProportion) + "-" + str(subsampleIdx+1) + ".txt"
    saveData(subsampledData, filenameOut)