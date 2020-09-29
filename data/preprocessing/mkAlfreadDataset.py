# test1.py

from AlfredData import AlfredData
from AlfredDataLoader import AlfredDataLoader

import os
import re
from typing import *



def getUniqueActionsHighLevel(data:List[AlfredData]):
    out = set()
    for elem in data:
        actions = elem.getActionsHighLevel()
        for action in actions:
            out.add(action)

    return out    

def getUniqueActionsLowLevel(data:List[AlfredData]):
    out = set()
    for elem in data:
        actions = elem.getActionsLowLevel()
        for action in actions:
            out.add(action)

    return out    


def getUniqueObjects(data:List[AlfredData]):
    out = set()
    for elem in data:
        #objects = elem.sceneObjects
        objects = elem.getObjectsHighLevel()

        for objectName in objects:
            objName = objectName.split("_")[0]
            out.add(objName)

    return out    

#
#   Template Generation
#
def strToTemplate(strInRaw):
    filterWords = ["the", "in", "on", "with"]    
    strIn = strInRaw

    # Remove filler words
    for filterWord in filterWords:
        #strIn = strIn.replace(filterWord + " ", "")
        strIn = re.sub(rf"\b{filterWord}\b", "", strIn)

    # Split into arguments
    strIn = strIn.replace("</s>", "\t")

    # Parse string
    commands = parseStrFormat(strIn)

    # Replace arguments with variable names
    strOut = ""
    lexiconIdx = 0
    word2idxLUT = {}
    idx2wordLUT = {}    
    for cTuple in commands:
        #print(cTuple)
        command = cTuple["command"]
        strOut += command

        arg1 = cTuple["arg1"]
        if (len(arg1) > 0):
            if (arg1 in word2idxLUT):
                arg1 = "<VAR" + str(word2idxLUT[arg1]) + ">"
            else:
                word2idxLUT[arg1] = lexiconIdx
                idx2wordLUT[lexiconIdx] = arg1
                lexiconIdx += 1
                arg1 = "<VAR" + str(word2idxLUT[arg1]) + ">"
            strOut += " <arg1> " + arg1

        arg2 = cTuple["arg2"]
        if (len(arg2) > 0):
            if (arg2 in word2idxLUT):
                arg2 = "<VAR" + str(word2idxLUT[arg2]) + ">"
            else:
                word2idxLUT[arg2] = lexiconIdx
                idx2wordLUT[lexiconIdx] = arg2
                lexiconIdx += 1
                arg2 = "<VAR" + str(word2idxLUT[arg2]) + ">"
            strOut += " <arg2> " + arg2

        strOut += " </s> "

    strOut = strOut.strip()

    # Encode variable names and their values in a string
    strOutVariables = ""
    for i in range(lexiconIdx):
        strOutVariables += "<VAR" + str(i) + "> " + idx2wordLUT[i] + " " + "<varsep> "
    strOutVariables = strOutVariables.strip()

    return (strOut, strOutVariables)


# Parse string format into an object
def parseStrFormat(strIn):
    out = list()
    goldSplit = strIn.strip().split('\t')

    for actionTuple in goldSplit:
        #print("actionTuple: " + str(actionTuple))

        indexOfArg1 = actionTuple.find("<arg1>")
        indexOfArg2 = actionTuple.find("<arg2>")        
        command = ""
        arg1 = ""
        arg2 = ""

        # All action, no arguments
        if (indexOfArg1 == -1) and (indexOfArg2 == -1):
            command = actionTuple.strip()
        
        # action and arg1, no arg2
        elif (indexOfArg1 != -1) and (indexOfArg2 == -1):
            command = actionTuple[:indexOfArg1].strip()
            arg1 = actionTuple[indexOfArg1+6:].strip()
        
        # action and arg1 and arg2
        elif (indexOfArg1 != -1) and (indexOfArg2 != -1):
            command = actionTuple[:indexOfArg1].strip()
            arg1 = actionTuple[indexOfArg1+6:indexOfArg2].strip()
            arg2 = actionTuple[indexOfArg2+6:].strip()

        # Unusual cases (signifying something is likely wrong, but must still be parsed)
        elif (indexOfArg1 == -1) and (indexOfArg2 != -1):
            command = actionTuple[:indexOfArg2].strip()
            arg2 = actionTuple[indexOfArg2+6:].strip()

        # Pack
        actionDict = {
            "command": command,
            "arg1": arg1,
            "arg2": arg2
        }
        
        out.append(actionDict)
    
    return out

#
#   Summary Statistics
#
def summaryStatistics(foldIn):
    minCmd = 999
    maxCmd = 0
    sumCmd = 0

    sumTaskDesc = 0

    for elem in foldIn:
        numCommands = len(elem.actionsHighLevelNormalized)
        sumCmd += numCommands
        minCmd = min(minCmd, numCommands)
        maxCmd = max(maxCmd, numCommands)

        sumTaskDesc += len(elem.textTaskDescs)



    avgCmd = float(sumCmd) / len(foldIn)
    avgTaskDesc = float(sumTaskDesc) / len(foldIn)

    print("Summary Statistics: ")
    print("Fold size: " + str(len(foldIn)))
    print("Min commands: " + str(minCmd))
    print("Max commands: " + str(maxCmd))
    print("Average commands: " + str(avgCmd))
    print("Average task descriptions per gold command sequence: " + str(avgTaskDesc))


#
#   Exporting train/dev/test set for fine tuning
#

def exportFold(fold):
    outTaskDesc = list()
    outTransformerText = list()
    outTemplateText = list()
    outVarText = list()

    for trial in fold:
        for trialText in trial.textTaskDescs:
            outTaskDesc.append(trialText)
            outTransformerText.append(trial.getTransformerTextT5())

            (templateStr, varStr) = strToTemplate(trial.getTransformerTextT5())
            outTemplateText.append(templateStr)
            outVarText.append(varStr)

            #print( trial.getTransformerTextT5() )
            #print( trial.actionsHighLevelTransformerTextT5 )

        #print (trial.toString())

    return (outTaskDesc, outTransformerText, outTemplateText, outVarText)


def writeListToFile(listIn, filename):
    with open(filename, 'w') as fp:
        for listElem in listIn:
            fp.write('%s\n' % listElem)    

def writeParallelListsToFile(listIn1, listIn2, filename):
    with open(filename, 'w') as fp:
        for i in range(len(listIn1)):        
            #fp.write('%s\t%s\n' % listIn1[i])    
            fp.write(listIn1[i])
            fp.write("\t")
            fp.write(listIn2[i])
            fp.write("\n")

def writeParallelListsToFileGPT2(listIn1, listIn2, filename):
    with open(filename, 'w') as fp:
        for i in range(len(listIn1)):        
            #fp.write('%s\t%s\n' % listIn1[i])    
            listIn1[i] = listIn1[i].replace(".", "").strip()
            fp.write(listIn1[i])
            fp.write(" [SEP] ")
            fp.write(listIn2[i].strip())
            fp.write(" [EOS]")
            fp.write("\n")

def writeParallelListsToFileTemplates(listIn1, listIn2, listIn3, listIn4, filename):
    with open(filename, 'w') as fp:
        for i in range(len(listIn1)):        
            #fp.write('%s\t%s\n' % listIn1[i])    
            listIn1[i] = listIn1[i].replace(".", "").strip()
            fp.write(listIn1[i].strip())
            fp.write("\t")
            fp.write(listIn2[i].strip())
            fp.write("\t")
            fp.write(listIn3[i].strip())
            fp.write("\t")
            fp.write(listIn4[i].strip())            
            fp.write("\n")

def writeParallelListsToFileGPT2Templates(listIn1, listIn2, listIn3, listIn4, filename):
    with open(filename, 'w') as fp:
        for i in range(len(listIn1)):        
            #fp.write('%s\t%s\n' % listIn1[i])    
            listIn1[i] = listIn1[i].replace(".", "").strip()
            fp.write(listIn1[i].strip())    # Task description
            fp.write(" [SEP] ")
            fp.write(listIn3[i].strip())    # Template
            fp.write("\n")

#
#
#

# Path to Alfred data
path = "/home/peter/github/alfred/data/json_2.1.0/train/"
filename = "pick_heat_then_place_in_recep-PotatoSliced-None-SideTable-28/trial_T20190906_212031_031197/traj_data.json"

# Load Alfred data
alfredDataLoader = AlfredDataLoader()

# Load word normalization
alfredDataLoader.loadWordNormalization("wordNormalization.tsv")
alfredDataLoader.loadInOnPrefixNormalization("prefixNormalization.tsv")

# Load train/dev/test folds
(filenamesTrain, filenamesDev, filenamesTest) = alfredDataLoader.makeTrainDevTest(path)
foldTrain = alfredDataLoader.loadFiles(path, filenamesTrain)
foldDev = alfredDataLoader.loadFiles(path, filenamesDev)
foldTest = alfredDataLoader.loadFiles(path, filenamesTest)


# Collect all objects in scenes
uniqueActionsHighLevel = getUniqueActionsHighLevel(foldTrain)
uniqueActionsLowLevel = getUniqueActionsLowLevel(foldTrain)
uniqueObjects1 = getUniqueObjects(foldTrain)

uniqueObjects2 = getUniqueObjects(foldDev)
uniqueObjects3 = getUniqueObjects(foldTest)

uniqueObjects = uniqueObjects1.union(uniqueObjects2).union(uniqueObjects3)


print("Actions (High-level): ")
print(uniqueActionsHighLevel)
print("")

print("Actions (Low-level): ")
print(uniqueActionsLowLevel)
print("")

print("Objects: " )
print(uniqueObjects)


for task in foldTrain:
    print(task.toString())
    print("\n\n")

# Make templates



# Export folds to file
(trainTaskDesc, trainTransformerText, trainTemplates, trainVars) = exportFold(foldTrain)
writeListToFile(trainTaskDesc, "alfred.train.taskdesc.txt")
writeListToFile(trainTransformerText, "alfred.train.transformertext.txt")
writeParallelListsToFile(trainTaskDesc, trainTransformerText, "alfred.train.txt")
writeParallelListsToFileGPT2(trainTaskDesc, trainTransformerText, "alfred.train.gpt2.txt")
writeParallelListsToFileTemplates(trainTaskDesc, trainTransformerText, trainTemplates, trainVars, "alfred.train.templates.txt")
writeParallelListsToFileGPT2Templates(trainTaskDesc, trainTransformerText, trainTemplates, trainVars, "alfred.train.gpt2.templates.txt")

(devTaskDesc, devTransformerText, devTemplates, devVars) = exportFold(foldDev)
writeListToFile(devTaskDesc, "alfred.dev.taskdesc.txt")
writeListToFile(devTransformerText, "alfred.dev.transformertext.txt")
writeParallelListsToFile(devTaskDesc, devTransformerText, "alfred.dev.txt")
writeParallelListsToFileGPT2(devTaskDesc, devTransformerText, "alfred.dev.gpt2.txt")
writeParallelListsToFileTemplates(devTaskDesc, devTransformerText, devTemplates, devVars, "alfred.dev.templates.txt")
writeParallelListsToFileGPT2Templates(devTaskDesc, devTransformerText, devTemplates, devVars, "alfred.dev.gpt2.templates.txt")


(testTaskDesc, testTransformerText, testTemplates, testVars) = exportFold(foldTest)
writeListToFile(testTaskDesc, "alfred.test.taskdesc.txt")
writeListToFile(testTransformerText, "alfred.test.transformertext.txt")
writeParallelListsToFile(testTaskDesc, testTransformerText, "alfred.test.txt")
writeParallelListsToFileGPT2(testTaskDesc, testTransformerText, "alfred.test.gpt2.txt")
writeParallelListsToFileTemplates(testTaskDesc, testTransformerText, testTemplates, testVars, "alfred.test.templates.txt")
writeParallelListsToFileGPT2Templates(testTaskDesc, testTransformerText, testTemplates, testVars, "alfred.test.gpt2.templates.txt")


# Summary statistics
summaryStatistics(foldTrain)