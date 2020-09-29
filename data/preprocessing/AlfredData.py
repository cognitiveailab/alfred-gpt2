# AlfredData.py

import json
import re

class AlfredData:
    # Text descriptions of this task
    textTaskDescs = list()

    # Metadata
    taskID = ""
    taskType = ""

    # Descriptions of task from Alfred task data
    actionsHighLevel = list()
    actionsLowLevel = list()
    sceneObjects = list()
    textAnnotation = list()

    # Normalized versions of above
    actionsHighLevelNormalized = list()
    actionsLowLevelNormalized = list()

    actionsHighLevelTransformerTextT5 = list()

    #
    #   Constructors
    # 

    # Main method of use -- instantiate with filename
    def __init__(self, filename, wordNormalizationList, prefixMap):
        self.loadFromJSON(filename, wordNormalizationList, prefixMap)

    #
    #   Load ALFRED task data file
    #
    def clear(self):
        self.textTaskDescs = list()
        self.taskID = ""
        self.taskType = ""
    
        self.actionsHighLevel = list()
        self.actionsLowLevel = list()
        self.sceneObjects = list()
        self.textAnnotation = list()


    def loadFromJSON(self, filename, normalizationList, prefixMap):        
        self.clear()

        # Step 1: Load JSON file
        with open(filename) as jsonFile:
            jsonData = json.load(jsonFile)

            # Step 2: Parse JSON
            #print(jsonData)

            plan = jsonData['plan']
            planHighLevel = plan['high_pddl']
            planLowLevel = plan['low_actions']
            scene = jsonData['scene']
            objectPoses = scene['object_poses']

            self.taskID = jsonData['task_id']
            self.taskType = jsonData['task_type']

            turkAnnotation = jsonData['turk_annotations']['anns']

            # Step 3: Parse high-level actions
            actionsHighLevel = list()
            for actionElem in planHighLevel:
                # Read JSON
                apiAction = actionElem['discrete_action']
                action = apiAction['action']
                args = apiAction['args']

                # Parse into a list of action/argument information
                actionOut = {}
                actionOut['action'] = action
                actionArgs = []

                for actionArg in args:
                    actionArgs.append(actionArg)
                actionOut['args'] = actionArgs

                # Store element                
                actionsHighLevel.append( actionOut )

                #debug
                #print(actionOut)
            self.actionsHighLevel = actionsHighLevel


            # Step 4: Parse low-level actions
            actionsLowLevel = list()
            for actionElem in planLowLevel:
                # Read JSON
                apiAction = actionElem['api_action']
                action = apiAction['action']

                actionOut = {}
                actionOut['action'] = action

                #print("actionElem: ")
                #print(actionElem)
                #print("\n")                

                # Optional object this action refers to
                if ('objectId' in apiAction):
                    objectID = apiAction['objectId']
                    actionOut['objectID'] = objectID                    

                
                # Optional recepticle this action refers to
                if ('receptacleObjectId' in apiAction):
                    receptacleID = apiAction['receptacleObjectId']
                    actionOut['receptacleID'] = receptacleID

                # Store element
                actionsLowLevel.append( actionOut )
            self.actionsLowLevel = actionsLowLevel


            # Step 5: Parse textual descriptions of task
            textAnnotation = []
            for tAnnotation in turkAnnotation:
                assignmentID = tAnnotation['assignment_id']
                taskDesc = tAnnotation['task_desc']
                taskDescTokens = re.findall(r"[\w']+|[.,!?;]", taskDesc)
                taskDesc = ' '.join(taskDescTokens)
                taskDesc = taskDesc.rstrip('.').strip()

                votes = tAnnotation['votes']
                highDescs = tAnnotation['high_descs']
                
                tAnnotationOut = {}
                tAnnotationOut['turkAssignmentID'] = assignmentID
                tAnnotationOut['taskDesc'] = taskDesc
                tAnnotationOut['votes'] = votes
                tAnnotationOut['highDescs'] = highDescs
                
                textAnnotation.append(tAnnotationOut)
            self.textAnnotation = textAnnotation


            # Step 6: Scene objects
            sceneObjects = []
            for objectElem in objectPoses:
                objectName = objectElem['objectName']
                sceneObjects.append(objectName)
            self.sceneObjects = sceneObjects


            # Step 7: Store text descriptions in a more easily accessed structure
            textTaskDescs = []
            for taskAnnotation in self.textAnnotation:
                textTaskDescs.append( taskAnnotation['taskDesc'] )
            self.textTaskDescs = textTaskDescs

            # print("\n\n")
            # print("High Level:\n")
            # print(actionsHighLevel)

            # print("\n\n")
            # print("Low Level:\n")
            # print(actionsLowLevel)

            # print("\n\n")
            # print(turkAnnotation)

            # Step 8: Normalize
            self.actionsHighLevelNormalized = self.normalize(self.actionsHighLevel, normalizationList)
            self.actionsLowLevelNormalized = self.normalize(self.actionsLowLevel, normalizationList)

            # Step 9: Convert to transformer text
            self.actionsHighLevelTransformerTextT5 = self.convertActionToTransformerHighLevelT5(self.actionsHighLevelNormalized, normalizationList, prefixMap)



    # Normalize words 
    def normalize(self, listIn, normalizationLUT):
        out = []
        for elem in listIn:
            norm = {}            
            #print(norm)
            for key in elem:
                value = elem[key]
                if (isinstance(value, str)):
                    norm[key] = self.normalizeStrHelper(value, normalizationLUT)
                elif (isinstance(value, list)):
                    listOut = []
                    for listElem in value:
                        listOut.append(self.normalizeStrHelper(listElem, normalizationLUT))
                    norm[key] = listOut                

                #print(norm[key])

            out.append(norm)
        
        return out

    def normalizeStrHelper(self, stringIn, normalizationLUT):        
        valueStr = stringIn        
        for i in range(len(normalizationLUT)):
            valueStr = valueStr.replace(normalizationLUT[i][0], normalizationLUT[i][1])

        #print("in: " + stringIn + "  out: " + valueStr)
        return valueStr


    #
    #   Accessors
    #
    def getActionsHighLevel(self):
        actionsOut = []
        for actionElem in self.actionsHighLevel:
            actionsOut.append(actionElem['action'])

        return actionsOut

    def getActionsLowLevel(self):
        actionsOut = []
        for actionElem in self.actionsLowLevel:
            actionsOut.append(actionElem['action'])

        return actionsOut

    def getObjectsHighLevel(self):
        objectsOut = []
        for actionElem in self.actionsHighLevel:
            if 'args' in actionElem:
                for arg in actionElem['args']:
                    objectsOut.append(arg)

        return objectsOut
    
    def getObjectsLowLevel(self):
        objectsOut = []
        for actionElem in self.actionsLowLevel:
            if 'args' in actionElem:
                for arg in actionElem['args']:
                    objectsOut.append(arg)

        return objectsOut


    #
    #   Converting API actions to free text
    #
    def convertActionToTransformerHighLevelT5(self, normalizedActionList, normalizationList, prefixMap):
        out = []
        for listElem in normalizedActionList:
            out.append( self.convertActionToTextHighLevelT5(listElem, normalizationList, prefixMap) )

        return out

    def getTransformerTextT5(self):        
        return " ".join(self.actionsHighLevelTransformerTextT5)
            

    def reverseLookup(self, stringIn, normalizationLUT):                     
        for i in range(len(normalizationLUT)):
            if (stringIn == normalizationLUT[i][1]):
                return normalizationLUT[i][0]            
        
        return stringIn


    def convertActionToTextHighLevelT5(self, elem, normalizationList, prefixMap):
        ACTIONSTART = "<ACTIONSTART>"
        ARG1START = "<ARG1>"
        ARG2START = "<ARG2>"
        ENDOFSEQUENCE = "</s>"

        strOut = ""
        #strOut += ACTIONSTART + " "
        strOut += elem['action']

        # First argument
        if (len(elem['args']) > 0):
            arg1 = elem['args'][0]
            strOut += " " + ARG1START + " the " + arg1

        if (len(elem['args']) > 1):
            arg2 = elem['args'][1]
            prenormalized = self.reverseLookup(arg2, normalizationList)
            prefix = ""
            if (prenormalized in prefixMap):
                prefix = prefixMap[prenormalized]

            strOut += " " + ARG2START + " " + prefix + " the " + arg2

        strOut += " " + ENDOFSEQUENCE

        return strOut



    #
    #   String methods
    #
    def toString(self):
        os = ""

        os += "Text Description of Task:\n"
        for i in range(len(self.textTaskDescs)):
            os += f'\t{i}: {self.textTaskDescs[i]}\n'

        os += "Commands (High-level):\n"
        for i in range(len(self.actionsHighLevel)):
            os += f'\t{i}: {self.actionsHighLevel[i]}\n'

        os += "Commands (Low-level):\n"
        for i in range(len(self.actionsLowLevel)):
            os += f'\t{i}: {self.actionsLowLevel[i]}\n'

        os += "Commands (High-level, transformer):\n"
        for i in range(len(self.actionsHighLevelTransformerTextT5)):
            os += f'\t{i}: {self.actionsHighLevelTransformerTextT5[i]}\n'

        os += "Commands (Transformer text):\n"        
        os += "\t" + self.getTransformerTextT5()

        return os


    




#
# Test
#

# path = "/home/peter/github/alfred/data/json_2.1.0/train/"
# filename = "pick_heat_then_place_in_recep-PotatoSliced-None-SideTable-28/trial_T20190906_212031_031197/traj_data.json"

# testObj = AlfredData(path + "/" + filename)

# print( testObj.toString() )
# print( testObj.getActionsHighLevel() )