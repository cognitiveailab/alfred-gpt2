# AlfredDataLoader.py

from AlfredData import AlfredData
import csv
import os


class AlfredDataLoader: 
    wordNormalizationMap = {}
    wordNormalizationList = list()
    prefixMap = {}

    # Get a list of all JSON files in a given path, recursively
    def loadAlfredFiles(self, path):
        filenamesOut = []

        for dirpath, dirs, files in os.walk(path):  
            for filename in files: 
                fname = os.path.join(dirpath, filename) 
                if fname.endswith('.json'): 
                    filenamesOut.append(fname)
        
        print (f'* {len(filenamesOut)} JSON files found in {path}')

        return filenamesOut


    # Return a list of files that go in train, dev, and test folds, automatically split 
    def makeTrainDevTest(self, path):
        outTrain = {}
        outDev = {}
        outTest = {}

        # Step 1: Get list of filenames to load
        filenames = self.loadAlfredFiles(path)

        filenamesByProblem = {}
        for filenameWithPath in filenames:         
            # Filename with user-supplied path removed
            filename = filenameWithPath[len(path):]   
            filenamePrefix = filename[:filename.index("/")]

            #print("filename: " + str(filename))
            #print("filenamePrefix: " + str(filenamePrefix))

            if (filenamePrefix not in filenamesByProblem):
                filenamesByProblem[filenamePrefix] = list()
            filenamesByProblem[filenamePrefix].append(filename)

        # Step 2: Separate into train/dev/test sets
        for key in filenamesByProblem:
            outTrain[key] = list()
            outDev[key] = list()
            outTest[key] = list()

            samples = filenamesByProblem[key]
            if (len(samples) == 0):
                # Do nothing
                pass
            elif (len(samples) == 1):
                # All to train
                outTrain[key].append( samples[0] )
            elif (len(samples) == 2):
                # One to train, one to test
                outTrain[key].append( samples[0] )
                outTest[key].append( samples[1] )
            else:
                # One to dev, one to test, all others to train
                outDev[key].append( samples[0] )
                outTest[key].append( samples[1] )
                for i in range(2, len(samples)):
                    outTrain[key].append( samples[i] )


        # Summary statistics
        countTrain = self.countFilesInFold(outTrain)
        countDev = self.countFilesInFold(outDev)   
        countTest = self.countFilesInFold(outTest)

        print (f'* Automatically generated folds completed ({countTrain} train, {countDev} development, {countTest} test).') 


        return (outTrain, outDev, outTest)

    # Folds are stored as problemType (key) -> array(filename) (value) pairs.  This utility quickly sums the total number of files across all keys.
    def countFilesInFold(self, fold):
        count = 0
        for key in fold:
            count += len(fold[key])
        return count


    def loadFiles(self, path, fold):
        out = []

        print (f'* loadFiles: Loading {self.countFilesInFold(fold)} files.')

        for problemType in fold:
            # print("Problem Type: " + problemType)
            for filename in fold[problemType]:
                # print("filename: " + filename)
                data = AlfredData(path + "/" + filename, self.wordNormalizationList, self.prefixMap)
                out.append( data )

        return out


    #
    #   Load word normalization
    # 
    def loadWordNormalization(self, filename):
        outHashMap = {}
        outList = list()

        with open(filename) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                key = row[0]
                value = row[1]
                outHashMap[key] = value

                outList.append(row)

        self.wordNormalizationMap = outHashMap
        self.wordNormalizationList = outList        

    #
    #   Load in-vs-on prefix
    # 
    def loadInOnPrefixNormalization(self, filename):
        outHashMap = {}        

        with open(filename) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                key = row[0]
                value = row[1]
                outHashMap[key] = value
                
        self.prefixMap = outHashMap        
