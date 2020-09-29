#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging

import numpy as np
import torch

import re
import json
import math
import time
from collections import defaultdict


from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    LineByLineTextDataset,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed
)

# from transformers import (
#     CONFIG_MAPPING,
#     MODEL_WITH_LM_HEAD_MAPPING,
#     AutoConfig,
#     AutoModelWithLMHead,
#     AutoTokenizer,
#     ,
#     HfArgumentParser,
# )

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


#
#   Loading ALFRED data
#
# Alfred
def readLangsAlfred(filenameIn, useTemplates=False):
    print("Reading lines... (" + filenameIn + ")")

    # Read the file and split into lines
    lines = open(filenameIn, encoding='utf-8').\
        read().strip().split('\n')

    # Replace separator tokens with our own separators
    #lines = [line.replace("</s>", "<sep>") for line in lines]

    # Split every line into pairs and normalize
    #pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = list()
    for line in lines:
        #print("line: " + str(line))
        fields = line.lower().split('\t')
        #print("Fields: " + str(fields))
        if (len(fields) >= 2):
            pairs.append( [fields[0], fields[1]] )

    # Make sure each pair contains a pair (e.g. not a blank line)
    pairs = [item for item in pairs if len(item) == 2]        

    # Convert strings into templates (if enabled)
    if (useTemplates == True):
        for i in range(len(pairs)):
            (templateStr, varStr) = strToTemplate(pairs[i][1])
            pairs[i] = [pairs[i][0], templateStr, varStr]        

    return pairs


#
#   Data preparation
#

def strToTemplate(strInRaw):
    filterWords = ["the", "in", "on", "with"]    
    strIn = strInRaw

    # Remove filler words
    for filterWord in filterWords:
        #strIn = strIn.replace(filterWord + " ", "")
        strIn = re.sub(rf"\b{filterWord}\b", "", strIn)

    # Split into arguments
    strIn = strIn.replace("<sep>", "\t")

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

        strOut += " <sep> "

    strOut = strOut.strip()

    # Encode variable names and their values in a string
    strOutVariables = ""
    for i in range(lexiconIdx):
        strOutVariables += "<VAR" + str(i) + "> " + idx2wordLUT[i] + " " + "<varsep> "
    strOutVariables = strOutVariables.strip()

    return (strOut, strOutVariables)
    


    


# Alfred
def prepareDataAlfred(filenameIn, giveFirstN = 0, useTemplates = False):
    pairs = readLangsAlfred(filenameIn, useTemplates)
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    #print("Trimmed to %s sentence pairs" % len(pairs))
        
    #print(input_lang.name, input_lang.n_words)
    #print(output_lang.name, output_lang.n_words)

    # Clue mode: Append first N commands to end of sentence as hints
    if (giveFirstN > 0):
        #print ("ERROR: Clue disabled in templates")
        #exit()

        for i in range(len(pairs)):
            goldTaskDesc = pairs[i][0]
            goldCommands = pairs[i][1].split("<sep>")

            for j in range(giveFirstN):
                goldTaskDesc += " <sep> "
                goldTaskDesc += goldCommands[j]

            pairs[i][0] = goldTaskDesc
            pairs[i][1] = pairs[i][1]
    
    return pairs


#
#   Scoring/Evaluation
#

# Post-filtering on the BiLSTM output to correct common errors/issues
def filterOutputString(strIn):
    stopWords = ["the", "in", "on", "with", "noop", "<EOS>"]
    startTokens = ["clean", "toggle", "put", "noop", "go", "heat", "cool", "slice", "pick"]
    endActionTokens = ["clean", "toggle", "put", "to", "heat", "cool", "slice", "up"]
    bigrams = [["pick", "up"], ["go", "to"]]

    # Replace separator tokens with tabs
    filtered = strIn.replace("<sep>", "\t")
    filtered = filtered.replace("</s>", "\t")

    # Split into tokens
    tokens = filtered.split(" ")

    # Remove adjacent duplicate tokens
    filteredTokens = list()
    for i in range(len(tokens)):

        # Strip whitespace, but keep a single tab if it exists. 
        tabIdx = tokens[i].find("\t")
        elem = tokens[i].strip()
        if (len(elem) == 0) and (tabIdx > -1):
            elem = "\t"

        if (i == 0):
            filteredTokens.append(elem)
        else:            
            if (len(filteredTokens) > 0) and (elem != filteredTokens[-1]):
                if (elem not in stopWords):
                    filteredTokens.append(elem)



    # Ensure bigrams are complete
    filteredTokens1 = list()
    # Bigrams -- back completions
    for i in range(len(filteredTokens)):  
        elem = filteredTokens[i]      
        filteredTokens1.append(elem)
        
        nextElem = ""
        if (i < len(filteredTokens)-1):
            nextElem = filteredTokens[i+1]
        
        for j in range(len(bigrams)):            
            if (elem == bigrams[j][0]) and (nextElem != bigrams[j][1]):
                filteredTokens1.append(bigrams[j][1])

    filteredTokens = filteredTokens1                

    filteredTokens1 = list()
    # Bigrams -- front completions
    for i in range(len(filteredTokens)):  
        elem = filteredTokens[i]              
        
        lastElem = ""
        if (i > 0):
            lastElem = filteredTokens1[-1]
        
        for j in range(len(bigrams)):            
            if (elem == bigrams[j][1]) and (lastElem != bigrams[j][0]):
                filteredTokens1.append(bigrams[j][0])

        filteredTokens1.append(elem)

    filteredTokens = filteredTokens1                


    # Ensure that start tokens have an delimiter in front of them. 
    filteredTokens1 = list()
    for i in range(len(filteredTokens)):
        elem = filteredTokens[i]
        if (len(filteredTokens1) > 0) and (filteredTokens1[-1] != "\t") and (elem in startTokens):
            filteredTokens1.append("\t")
        filteredTokens1.append(elem)

    filteredTokens = filteredTokens1

    # Ensure that action commands are followed by an <arg1> tag
    filteredTokens1 = list()
    for i in range(len(filteredTokens)):  
        elem = filteredTokens[i]      
        filteredTokens1.append(elem)
        if (len(filteredTokens1) > 0) and (filteredTokens1[-1] in endActionTokens):
            if (i < len(filteredTokens)-1) and (filteredTokens[i+1] != "<arg1>"):
                filteredTokens1.append("<arg1>")
        
    filteredTokens = filteredTokens1


    # Generate string
    strOut = " ".join(filteredTokens)
    strOut = strOut.replace("\t ", "\t")
    strOut = strOut.replace(" \t", "\t")

    # Fix strange bug in transformer that sometimes starts strings with "o\t"
    strOut = removePrefix(strOut, "o\t")


    return strOut


def removePrefix(strIn, prefix):
    if strIn.startswith(prefix):
        return strIn[len(prefix):]
    return strIn

def evaluateSetTSV(pairsIn, pathOut, filenameTSVOut, model, args, tokenizer, limit = 0, batchSize = 8):
    outPairs = list()    
    fp = open(pathOut + filenameTSVOut, "w")
       
    # Step 1: Pre-tokenize task descriptions
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'    

    # Separate prompts and gold text
    promptStrs = [p[0] + " [SEP] " for p in pairsIn]
    goldStrs = [p[1] for p in pairsIn]

    
    # Encode prompts with tokenizer
    prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
    #encodedPairs = tokenizer.batch_encode_plus(pairsIn, add_special_tokens=False, pad_to_max_length=False, return_tensors="pt")
    #encodedPairsInputIDs = encodedPairs["input_ids"]

    # Step 2: Sort encoded task descriptions by length
    sortedByLength = {}
    for i in range(len(pairsIn)):
        promptStr = promptStrs[i]
        goldStr = goldStrs[i]

        encoded = tokenizer.encode_plus(
            promptStr, add_special_tokens=False, pad_to_max_length=False, return_tensors="pt", add_space_before_punct_symbol=True
        )
        encoded = encoded["input_ids"]

        packed = {
            "idx": i,
            "encoded": encoded,
            "taskDesc": promptStr,
            "gold": goldStr,
        }

        encodedLength = encoded.size()[1]
        if (encodedLength not in sortedByLength):
            sortedByLength[encodedLength] = list()
        sortedByLength[encodedLength].append(packed)

        #print("size: " + str(encodedLength))

    for key in sortedByLength:
        print("Length " + str(key) + ": " + str(len(sortedByLength[key])) + " tasks.")



    # Step N: Generate predictions for each task description
    numCompleted = 0
    numTotal = len(pairsIn)
    startTime = time.time()

    for key in sortedByLength:
        tasksOfKeyLength = sortedByLength[key]
        print(" * Procesing tasks of length " + str(key) + " (Total: " + str(len(tasksOfKeyLength)) + ")")        

        # Step 2: Post-process completions
        for idx in range(0, len(tasksOfKeyLength), batchSize):            
            # Limiting (for debug purposes)
            if (limit > 0) and (idx > limit) :
                print("Limit reached (" + str(limit) + ").  Exiting. ")
                break

            # Get the set of pairs for this batch
            packedPairsBatch = tasksOfKeyLength[idx:min(idx+batchSize, len(pairsIn))]

            # Assemble the 1D tensors in this batch into a 2D tensor
            tensorList = list()
            for elem in packedPairsBatch:
                tensorList.append( torch.flatten(elem["encoded"]) )
            tensorsPacked = torch.stack(tensorList)

            #print("tensorsPacked: " + str(tensorsPacked.size()))

            # Generate prediction            
            predictedBatch = generate(tensorsPacked, model, args, tokenizer)            

            # Strip the prompt (task description) text from the beginning of the output
            for j in range(0, len(predictedBatch)):
                taskDesc = packedPairsBatch[j]["taskDesc"]
                predictedBatch[j] = predictedBatch[j][len(taskDesc):]

            # Remove leading/trailing whitespace, normalize to lower case
            predictedBatch = [p.strip().lower() for p in predictedBatch] 

            # Filtering/normalization
            for batchIdx in range(len(packedPairsBatch)):   
                numCompleted += 1
                absoluteIdx = packedPairsBatch[batchIdx]["idx"]

                taskDesc = packedPairsBatch[batchIdx]["taskDesc"]
                gold = packedPairsBatch[batchIdx]["gold"]
                origIdx = packedPairsBatch[batchIdx]["idx"]
                predicted = predictedBatch[batchIdx]

                filteredGold = filterOutputString(gold)
                filteredPredicted = filterOutputString(predicted)

                print("")
                print(str(numCompleted) + " / " + str(numTotal))
                print(absoluteIdx)        
                print("taskDesc  = " + str(taskDesc))    
                print("gold      = " + str(filteredGold))
                print("predicted = " + str(filteredPredicted))

                outPairs.append( {  "taskDesc": taskDesc, 
                                    "gold": filteredGold, 
                                    "predicted": filteredPredicted,
                                    "idx": origIdx,
                                } )

                fp.write("\n")    
                fp.write(str(absoluteIdx) + "\n")
                fp.write("taskDesc\t" + taskDesc + "\n")
                fp.write("gold\t" + filteredGold + "\n")
                fp.write("predicted\t" + filteredPredicted + "\n")
                fp.flush()
            
            # Status bar
            showStatusBar(startTime, numCompleted, numTotal)        
    
    fp.close()

    # Return
    return outPairs


# Save pairs
def savePairs(pairs, filenameOut):
    print(" Saving evaluation data... ")
    with open(filenameOut, 'w') as outfile:
        json.dump(pairs, outfile)

# Load pairs
def loadPairs(filenameIn):
    print(" Loading evaluation data... (" + filenameIn + ")")
    with open(filenameIn) as jsonFile:
        jsonData = json.load(jsonFile)
        return jsonData
    
    # If not successful, return list
    return list()


def isNumber(inVar):
    if (isinstance(inVar, int)):
        return True
    if (isinstance(inVar, float)):
        return True
    if (isinstance(inVar, str)):
        if (inVar.isdigit()):
            return True
        else:
            return False
    
    # Otherwise
    print("Not sure what this is: " + str(inVar))
    return False
    

# Save TSV File
def saveTSVFile(pairsIn, pathOut, filenameTSVOut):
    fp = open(pathOut + filenameTSVOut, "w")

    # Sort by index  
    pairsIn1 = pairsIn  
    if (isNumber(pairsIn1[0]["idx"])):
        pairsIn1 = sorted(pairsIn1, key = lambda i: i['idx'])

    for pair in pairsIn1:
        fp.write("\n")    
        idx = pair["idx"]
        if (isNumber(idx)):
            fp.write(str(pair["idx"]) + "\n")
        else:
            fp.write("0\n")
        fp.write("taskDesc\t" + pair["taskDesc"] + "\n")
        fp.write("gold\t" + pair["gold"] + "\n")
        fp.write("predicted\t" + pair["predicted"] + "\n")
        fp.flush()    

    fp.close()

def saveErrorAnalysisTSVFile(errorsIn, pathOut, filenameTSVOut):
    print ("* Saving error output: " + pathOut + filenameTSVOut)
    fp = open(pathOut + filenameTSVOut, "w")

    count = 0

    # Sort by index
    #pairsIn1 = pairsIn    
    #if (isNumber(pairsIn1[0]["pair"]["idx"])):
    #    pairsIn1 = sorted(pairsIn1, key = lambda i: i['idx'])

    for pairTuple in errorsIn:
        scoresOut = pairTuple["scoresOut"]        
        perfectElem = scoresOut["perfectTriples_Same"]

        fp.write("\n")    
        pair = pairTuple["pair"]
        idx = pair["idx"]
        if (isNumber(idx)):
            fp.write(str(count) + "\t" + str(pair["idx"]) + "\n")
        else:
            fp.write(str(count) + "\t0\n")
        fp.write(str(count) + "\ttaskDesc\t" + pair["taskDesc"] + "\n")

        #fp.write(str(count) + "\tgold\t" + pair["gold"] + "\n")
        fp.write(str(count) + "\tgold\t")
        elems = pair["gold"].split("\t")
        for i in range(len(elems)):
            if (i < len(perfectElem)) and (perfectElem[i] < 1.0):
                fp.write("## ")
            fp.write(elems[i] + "\t")
        fp.write("\n")

        #fp.write(str(count) + "\tpredicted\t" + pair["predicted"] + "\n")
        fp.write(str(count) + "\tpredicted\t")
        elems = pair["predicted"].split("\t")
        for i in range(len(elems)):
            if (i < len(perfectElem)) and (perfectElem[i] < 1.0):
                fp.write("## ")
            fp.write(elems[i] + "\t")
        fp.write("\n")


        fp.flush()    

        count += 1

    fp.close()


# def evaluateSetTSV(pairsIn, pathOut, filenameTextOut, limit = 0):
#     outPairs = list()

#     fp = open(pathOut + "/" + filenameTextOut, "w")

#     for i in range(len(pairsIn)):
#         # Limiting (for debug purposes)
#         if (limit > 0) and (i > limit) :
#             print("Limit reached (" + str(limit) + ").  Exiting. ")
#             break

#         pair = pairsIn[i]
#         sentIn = pair[0]
#         goldTemplate = pair[1]
#         goldVars = pair[2]

#         outputWords, attentions = evaluate(encoder1, attn_decoder1, sentIn)
#         outputStrTemplate = ' '.join(outputWords)

#         outputWordsVars, attentionsVars = evaluate(encoder1, attn_decoder2, sentIn)
#         outputStrVars = ' '.join(outputWordsVars)

#         sentInSanitized = sentIn.replace(".", "").replace(" ", "-").strip()
#         filenameOutAttention = pathOutAttention + "/" + sentInSanitized + ".pdf"        
        
#         # Produce filtered gold and output strings
#         goldFilteredTemplate = filterOutputString(goldTemplate)
#         outputFilteredTemplate = filterOutputString(outputStrTemplate)        

#         goldVarsFiltered = filterOutputStringVars(goldVars.replace("<varsep>", "\t"))       # Shouldn't be required, but done anyway in case there are subtle formatting changes introduced
#         outputStrVarsFiltered = filterOutputStringVars(outputStrVars.replace("<varsep>", "\t"))

#         # Reassemble command string from templates and variables
#         reassembledGold = reassembleTemplateVars(goldFilteredTemplate, goldVarsFiltered)
#         reassembledPred = reassembleTemplateVars(outputFilteredTemplate, outputStrVarsFiltered)

#         # print("")    
#         # print(i)
#         # print('input =', sentIn)
#         # print('goldTemplate =', goldTemplate)
#         # print('outputTemplate =', outputStrTemplate)
#         # print('goldVars =', goldVarsFiltered)
#         # print('outputVars =', outputStrVarsFiltered)

#         print("")     
#         print(str(i))
#         print("taskDesc\t" + sentIn)
#         print("gold\t" + reassembledGold)
#         print("predicted\t" + reassembledPred)
#         print("goldTemplate\t" + goldFilteredTemplate)
#         print("predictedTemplate\t" + outputFilteredTemplate)
#         print("goldVar\t" + goldVarsFiltered)
#         print("predictedVar\t" + outputStrVarsFiltered)
#         # Reassemble command from template and variable string


#         outPairs.append( {  "taskDesc": sentIn, 
#                             "gold": reassembledGold, 
#                             "predicted": reassembledPred,
#                             "goldTemplate": goldFilteredTemplate, 
#                             "predictedTemplate": outputFilteredTemplate,
#                             "goldVar": goldVarsFiltered,
#                             "predictedVar": outputStrVarsFiltered
#                         } )

#         fp.write("\n")    
#         fp.write(str(i) + "\n")
#         fp.write("taskDesc\t" + sentIn + "\n")
#         fp.write("gold\t" + reassembledGold + "\n")
#         fp.write("predicted\t" + reassembledPred + "\n")
#         fp.write("goldTemplate\t" + goldFilteredTemplate + "\n")
#         fp.write("predictedTemplate\t" + outputFilteredTemplate + "\n")
#         fp.write("goldVar\t" + goldVarsFiltered + "\n")
#         fp.write("predictedVar\t" + outputStrVarsFiltered + "\n")
        
#         fp.flush()

#         if (attentionExportLimit == 0) or (i < attentionExportLimit):
#             print("filenameOutAttention: " + filenameOutAttention)
#             #fp.write("filenameOut: " + filenameOutAttention + "\n")
#             fp.flush()
#             showAttention(sentIn, outputWords, attentions, filenameOutAttention)

#     fp.close()

#     return outPairs


#
#   Scorer
#
INVALID_SCORE = -999
possibleActions = ["clean", "toggle", "put", "go to", "heat", "cool", "slice", "pick up", "unknown"]

def scoreOnePair(goldStr, predictedStr, scoreMode):    
    MAX_SEQ_LENGTH = 10     # Maximum number of commands in sequence, for scoring by sequence index

    gold = parseStrFormat(goldStr)
    pred = parseStrFormat(predictedStr)

    commandsSame = 0
    commandsTotal = 0
    arg1Same = 0
    arg1Total = 0
    arg2Same = 0
    arg2Total = 0

    isPerfectAfterSecond = True

    commandsSameByIdx = [0]*MAX_SEQ_LENGTH
    commandsTotalByIdx = [0]*MAX_SEQ_LENGTH
    arg1SameByIdx = [0]*MAX_SEQ_LENGTH
    arg1TotalByIdx = [0]*MAX_SEQ_LENGTH
    arg2SameByIdx = [0]*MAX_SEQ_LENGTH
    arg2TotalByIdx = [0]*MAX_SEQ_LENGTH

    perfectTripleSameByIdx = [0]*MAX_SEQ_LENGTH
    perfectTripleTotalByIdx = [0]*MAX_SEQ_LENGTH

    tripleSuccessByActionSame = [0]*(len(possibleActions)+1)
    tripleSuccessByActionTotal = [0]*(len(possibleActions)+1)

    maxLength = max(len(gold), len(pred))

    for i in range(len(gold)):
        goldElem = {"command": "", "arg1": "", "arg2": ""}
        predElem = {"command": "", "arg1": "", "arg2": ""}

        # Perfect triple calculation
        correctCommand = True
        correctArg1 = True
        correctArg2 = True


        if (i < len(gold)):
            goldElem = gold[i]  
        if (i < len(pred)):
            predElem = pred[i]

        # Command
        #if (goldElem["command"] == predElem["command"]):
        if (compareArguments(goldElem["command"], predElem["command"], scoreMode)):        
            commandsSame += 1
            commandsSameByIdx[min(i, MAX_SEQ_LENGTH-1)] += 1
        else:
            correctCommand = False

        commandsTotal += 1        
        commandsTotalByIdx[min(i, MAX_SEQ_LENGTH-1)] += 1

        # Argument 1
        if (len(goldElem["arg1"]) > 0) or (len(predElem["arg1"]) > 0):
            #if (goldElem["arg1"] == predElem["arg1"]):
            if (compareArguments(goldElem["arg1"], predElem["arg1"], scoreMode)):        
                arg1Same += 1
                arg1SameByIdx[min(i, MAX_SEQ_LENGTH-1)] += 1
            else:
                correctArg1 = False

            arg1Total += 1
            arg1TotalByIdx[min(i, MAX_SEQ_LENGTH-1)] += 1

        # Argument 2
        if (len(goldElem["arg2"]) > 0) or (len(predElem["arg2"]) > 0):
            #if (goldElem["arg2"] == predElem["arg2"]):                
            if (compareArguments(goldElem["arg2"], predElem["arg2"], scoreMode)):        
                arg2Same += 1
                arg2SameByIdx[min(i, MAX_SEQ_LENGTH-1)] += 1
            else:
                correctArg2 = False

            arg2Total += 1
            arg2TotalByIdx[min(i, MAX_SEQ_LENGTH-1)] += 1


        # Perfect triple score calculation
        if (correctCommand == True) and (correctArg1 == True) and (correctArg2 == True):
            perfectTripleSameByIdx[min(i, MAX_SEQ_LENGTH-1)] += 1
        perfectTripleTotalByIdx[min(i, MAX_SEQ_LENGTH-1)] += 1

        # Perfect after second triple and beyond
        if (((correctCommand != True) or (correctArg1 != True) or (correctArg2 != True)) and (i > 0)):
            # There was an error in this triple, and it's not in the first position -- mark this sequence as not being perfect past 2
            isPerfectAfterSecond = False

        # Correctness by command
        commandIdx = possibleActions.index("unknown")
        if (goldElem["command"] in possibleActions):
            commandIdx = possibleActions.index(goldElem["command"])
        if (correctCommand == True) and (correctArg1 == True) and (correctArg2 == True):
            tripleSuccessByActionSame[commandIdx] += 1
        tripleSuccessByActionTotal[commandIdx] += 1

        

    commandsProp = INVALID_SCORE
    arg1Prop = INVALID_SCORE
    arg2Prop = INVALID_SCORE
    if (commandsTotal > 0):
        commandsProp = commandsSame / commandsTotal
    
    if (arg1Total > 0): 
        arg1Prop = arg1Same / arg1Total

    if (arg2Total > 0):        
        arg2Prop = arg2Same / arg2Total

    # Perfect after second triple (second onward)
    perfectAfterSecondSame = [0.0]
    if (isPerfectAfterSecond == True):
        perfectAfterSecondSame = [1.0]
    perfectAfterSecondTotal = [1.0]


    # Pack scores
    scoresOut = {
        "commandsProp": commandsProp,
        "arg1Prop": arg1Prop,
        "arg2Prop": arg2Prop,
        "commandsByIdx_Same": commandsSameByIdx,
        "commandsByIdx_Total": commandsTotalByIdx,
        "arg1ByIdx_Same": arg1SameByIdx,
        "arg1ByIdx_Total": arg1TotalByIdx,
        "arg2ByIdx_Same": arg2SameByIdx,
        "arg2ByIdx_Total": arg2TotalByIdx,
        "perfectTriples_Same": perfectTripleSameByIdx, 
        "perfectTriples_Total": perfectTripleTotalByIdx,
        "perfectTriplesByCommand_Same": tripleSuccessByActionSame,
        "perfectTriplesByCommand_Total": tripleSuccessByActionTotal,
        "perfectAfterSecond_Same": perfectAfterSecondSame,
        "perfectAfterSecond_Total": perfectAfterSecondTotal,
    }

    return scoresOut
    
# Helper function to compare to strings (arguments) using exact ("butter knife" to "butter knife" or permissive ("knife" to "butter knife") criteria.
# Permissive requires only one token between arguments to be identical. 
def compareArguments(argA, argB, scoreMode):
    if (scoreMode == MODE_SCORE_EXACT):
        if (argA == argB):
            return True
        else:
            return False

    elif (scoreMode == MODE_SCORE_PERMISSIVE):
        tokensA = argA.split(" ")
        tokensB = argB.split(" ")
        count = 0
        for token in tokensA:
            if (token in tokensB):
                count += 1
        
        if (count > 0):
            return True
        else:
            return False

    else:
        print("ERROR: Unknown score mode (" + scoreMode + ").")
        exit()
    

# Score a list of (gold, predicted) sequence pairs
def scorePairs(pairs, scoreMode, goldKey="gold", predictedKey="predicted"):
    scoreSums = {} #defaultdict(lambda:0)
    counts = defaultdict(lambda:0)

    countPerfect = 0
    countTotal = 0

    # List of errorful predictions for later analysis
    errorsOut = list()

    for pair in pairs:
        #print("pair: " + str(pair))
        inputStr = pair["taskDesc"]
#        goldStr = pair["gold"]
#        predStr = pair["predicted"]
        goldStr = pair[goldKey]
        predStr = pair[predictedKey]

        scoresOut = scoreOnePair(goldStr, predStr, scoreMode)
        
        print("input\t" + inputStr)
        print("gold\t" + goldStr)
        print("pred\t" + predStr)
        print("Scores: " + str(scoresOut))        

        # sum scores
        for key in scoresOut:
            score = scoresOut[key]
            if (score != INVALID_SCORE):

                # If key doesn't exist, add blank key
                if (key not in scoreSums):
                    if (type(score) is str):
                        scoreSums[key] = 0
                    elif (type(score) is float):
                        scoreSums[key] = 0.0
                    elif (type(score) is list):
                        scoreSums[key] = [0.0] * len(score)
                    else:
                        print("ERROR: Unrecognized type in scores (" + str(type(score)) + ")")

                # Add key
                if (type(score) is int) or (type(score) is float):
                    # Sum score
                    scoreSums[key] += score
                elif (type(score) is list):
                    # Sum array element-wise
                    curScores = scoreSums[key]
                    for idx in range(len(score)):
                        curScores[idx] += score[idx]
                    scoreSums[key] = curScores

                counts[key] += 1

        if (((scoresOut["commandsProp"] == 1.0) or (scoresOut["commandsProp"] == INVALID_SCORE)) and 
            ((scoresOut["arg1Prop"] == 1.0) or (scoresOut["arg1Prop"] == INVALID_SCORE)) and
            ((scoresOut["arg2Prop"] == 1.0) or (scoresOut["arg2Prop"] == INVALID_SCORE)) ):
            countPerfect += 1
            print("PERFECT!")

        countTotal += 1    

        # Append errors to a list for later analysis
        if (scoresOut["perfectAfterSecond_Same"][0] != 1.0):
            errorsOut.append({
                "pair": pair,
                "scoresOut": scoresOut,
            })

        print("")


    # Average scores
    avgScores = {}
    for key in scoreSums:
        if (type(scoreSums[key]) is int) or (type(scoreSums[key]) is float):
            avgScores[key] = scoreSums[key] / counts[key]
        else:
            # Temporary -- do not average non-int non-float types (e.g. lists)
            if (key.endswith("_Same")):
                # Compute the element-wise average of list-based scores
                sums = scoreSums[key]                                       # Get sums
                totalCountsKey = key.split("_")[0] + "_Total"               
                totalCounts = scoreSums[totalCountsKey]                     # Get total counts                
                avgList = [safeDivideElemHelper(i, j) for i, j in zip(sums, totalCounts)]        # Average
                avgKey = key.split("_")[0] + "_Avg"
                avgScores[avgKey] = avgList                                 # Save as new key *_Avg

                # Also compute an average across all elements/sums
                scoreSum = sum(sums)
                totalSum = sum(totalCounts)
                elementwiseAvg = scoreSum / totalSum
                eAvgKey = key.split("_")[0] + "_ElementwiseAvg"
                avgScores[eAvgKey] = elementwiseAvg
                
            avgScores[key] = scoreSums[key]                                 # Retain original counts for debugging purposes

    # Perfect triples by command -- post-processing
    avgScoresByCommand = avgScores["perfectTriplesByCommand_Avg"]
    avgScoresByCommandDict = {}
    for i in range(len(possibleActions)):
        actionName = possibleActions[i]
        avgScore = avgScoresByCommand[i]
        avgScoresByCommandDict[actionName] = avgScore

    avgScores["perfectTriplesByCommand_Dict"] = avgScoresByCommandDict


    # Proportion of perfect scoring trials
    propPerfect = countPerfect / countTotal
    avgScores["perfectProp"] = propPerfect
    avgScores["countTotal"] = countTotal

    # Store scoring mode
    avgScores["scoreMode"] = scoreMode


    # Return
    return (avgScores, errorsOut)


# Helper for list-comprehension division (above)
def safeDivideElemHelper(i, j):
    if (j != 0.0):
        return i/j
    else:
        return 0.0


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
#   Generation (GPT-2)
#

# promptsIn is a list of sequence prompts to generate completions for

## Broken -- batch conversion is generating unexpected sequences

def generate(tensorList, model, args, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token    
    tokenizer.padding_side = 'left'    

    #print("promptsIn:")
    #print(promptsIn)

    # # Different models need different input formatting and/or extra arguments
    # requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    # #requires_preprocessing = True
    # if requires_preprocessing:
    #     print("REQUIRES PREPROCESSING")
    #     prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
    #     preprocessed_prompt_text = prepare_input(args, model, tokenizer, promptsIn)      # PJ: Check that prepare_input works on a list
    #     #encoded_prompt = tokenizer.batch_encode_plus(
    #     encoded_prompt = tokenizer.batch_encode_plus(
    #         preprocessed_prompt_text, add_special_tokens=False, pad_to_max_length=True, return_tensors="pt", add_space_before_punct_symbol=True
    #     )
    # else:
    #     #encoded_prompt = tokenizer.encode(promptIn, add_special_tokens=False, return_tensors="pt")
    #     encoded_prompt = tokenizer.batch_encode_plus(promptsIn, add_special_tokens=False, pad_to_max_length=True, return_tensors="pt")
    #     #tokenizer.encode_batch

    # encoded_prompt = encoded_prompt["input_ids"]    # Grab input_ids tensor from BatchEncoding
    # encoded_prompt = encoded_prompt.to(args.device)

    #if encoded_prompt.size()[-1] == 0:
    #    input_ids = None
    #else:
    #input_ids = encoded_prompt    
    #input_ids = input_ids.to(args.device)   ###
    #input_ids = encoded_prompt["input_ids"]         
    
    input_ids = tensorList    
    input_ids = input_ids.to(args.device)


    #print("input_ids:")
    #print(input_ids)
    ##print(str(type(input_ids)))
    #for i in range(input_ids.size()[0]):
    #    text = tokenizer.decode(input_ids[i], clean_up_tokenization_spaces=True)
    #    print(str(i) + ": " + text)

    #print(str(type(input_ids)))
    #print("size:")
    #print(str(input_ids.size()))
    maxSize = input_ids.size()[1]


    output_sequences = model.generate(
        input_ids=input_ids,
        #max_length=args.length + len(encoded_prompt[0]),
        max_length=args.length + maxSize,
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
        num_beans=1,
    )

    #print("output sequences:")
    #print(str(type(output_sequences)))
    #print(str(output_sequences.size()))

    # Remove the batch dimension when returning multiple sequences
#    if len(output_sequences.shape) > 2:
#        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        # total_sequence = (            
        #     text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        #     #text[len(tokenizer.decode(input_ids[generated_sequence_idx], clean_up_tokenization_spaces=True)) :]
        # )
        #promptsIn + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        total_sequence = text

        generated_sequences.append(total_sequence)
        #print(total_sequence)
    
    return generated_sequences

#
#   Status Bar
# 
def showStatusBar(startTime, curProgress, maxProgress, barLength=40):        
    numLength = len(str(maxProgress))
    proportion = curProgress / maxProgress
    numCompleted = math.floor(proportion*barLength)
    numTodo = barLength - numCompleted
    timePassed = float(int(time.time() - startTime))
    timeRequired = timePassed / proportion
    timeLeft = math.floor(timeRequired - timePassed)

    minsLeft = math.floor(timeLeft / 60)
    secsLeft = timeLeft - (minsLeft*60)

    outStr = "  " + str(curProgress).zfill(numLength) + "/" + str(maxProgress) + ": ["
    outStr += "="*numCompleted + ">" + "."*numTodo
    outStr += "]  "
    outStr += "%.1f" % (proportion*100) + "% "
    outStr += " - ETA " + str(minsLeft).zfill(2) + ":" + str(secsLeft).zfill(2)

    outStr += "\n"
    outStr += " timePassed: " + str(timePassed) + "  timeRequired: " + str(timeRequired) + "  timeLeft: " + str(timeLeft) + "  proportion: " + str(proportion)

    #return outStr
    print(outStr)


MODE_SCORE_EXACT    =   9
MODE_SCORE_PERMISSIVE   =   10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--output_filename", type=str, default="out")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_filename", type=str, default="", required=True)


    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--stop_token", type=str, default="[EOS]", help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args)

    print("Generation Parameters: ")
    print("temperature = " + str(args.temperature))
    print("top_k = " + str(args.k))
    print("top_p = " + str(args.p))
    print("repetition_penalty = " + str(args.repetition_penalty))
    print("num_return_sequences = " + str(args.num_return_sequences))

    #exit()

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)


    # Parameters    
    giveTopN            = 0        
    #giveTopN            = 1    
    useTemplates        = False
    #useTemplates        = True
    modelName = args.model_name_or_path.split("/")[-1]

    # Load the evaluation data
    #filenameInEval = "../alfred.dev.txt"
    #filenameInEval = "../alfred.test.txt"
    filenameInEval = args.eval_filename

    evalSetName = args.output_filename
    #evalSetName = "dev"
    #if ("test" in filenameInEval):
    #    evalSetName = "test"

    print("Loading Evaluation Set... ( " + filenameInEval + ")")
    #input_langEval, output_langEval, pairsEval = prepareDataAlfred(filenameInDev, 'eng', 'command', pretrainedEmbeddingModel, reverse=False)
    pairsEval = prepareDataAlfred(filenameInEval, giveTopN, useTemplates)

    #
    # Dataset Test
    


    #print(pairsEval)
    #exit()

    #def evaluateSetTSV(pairsIn, pathOut, filenameTSVOut, model, args, tokenizer, limit = 0):
    pathOut = ""
    pathOutEval = ""
    filenameTSVOut = "evalOut"
    if (len(args.output_filename)):
        filenameTSVOut += "." + args.output_filename

    # Evaluation
    print ("Evaluation...")

    # An evalBatchSize of 32 takes approximately 16gb of GPU RAM
    #evalBatchSize = 32    # GPT-2 
    #evalBatchSize = 8     # XLNet
    evalBatchSize = args.batch_size


    #evaluateData = False     # Evaluate: True,  load saved previously evaluated data: False
    evaluateData = True
    pairs = list()
    if (evaluateData == True):
        print("Running evaluation... ")
        pairs = evaluateSetTSV(pairsEval, "", filenameTSVOut + "." + modelName + "." + evalSetName +".tsv", model, args, tokenizer, limit = 0, batchSize = evalBatchSize)    
        savePairs(pairs, pathOutEval + "" + filenameTSVOut + "." + modelName + "." + evalSetName +".predicted.json")
    else:
        print ("Loading saved evaluation data... ")
        pairs = loadPairs(pathOutEval + "" + filenameTSVOut + "." + modelName + "." + evalSetName +".predicted.json")


    ## Debug -- Re-filter predicted strings to remove the leading "o\t" bug sometimes present
    print("Re-filtering predictions... ")
    for i in range(len(pairs)):
        predictedStr = pairs[i]["predicted"]
        print("before: " + predictedStr)
        predictedStr = removePrefix(predictedStr, "o\t")
        pairs[i]["predicted"] = predictedStr
        print("after:  " + predictedStr)
    saveTSVFile(pairs, "", filenameTSVOut + "." + modelName + "." + evalSetName +".tsv")
    savePairs(pairs, pathOutEval + "" + filenameTSVOut + "." + modelName + "." + evalSetName +".predicted.json")

    print("Scoring...")
    # Load evaluation pairs, if not generated above
#    if (len(pairs) == 0):
#        pairs = loadPairs(pathOutEval + "/" + filenameEvalOut + ".predicted." + evalSetName + ".json")

    # Perform scoring
    (avgScoresExact, errorsOutExact) = scorePairs(pairs, MODE_SCORE_EXACT, goldKey="gold", predictedKey="predicted")
    (avgScoresPermissive, errorsOutPermissive) = scorePairs(pairs, MODE_SCORE_PERMISSIVE, goldKey="gold", predictedKey="predicted")

    #avgScoresExactTemplate = scorePairs(pairs, MODE_SCORE_EXACT, goldKey="goldTemplate", predictedKey="predictedTemplate")
    
    avgScores = {"exact": avgScoresExact,
                "permissive": avgScoresPermissive,
                #"templateOnlyExact": avgScoresExactTemplate,
                #"learningRate": learningRate,
                #"hidden_size": hidden_size, 
                "giveTopN": giveTopN,
                #"USE_PRETRAINED_EMB": USE_PRETRAINED_EMB,
                "filenameInEval": filenameInEval,
                "useTemplates": useTemplates,
                "pairsEvalLength": len(pairsEval),
                }

    print("")
    print("Average scores for " + str(len(pairs)) + " pairs in evaluation set: ")
    scoreStr = json.dumps(avgScores, indent=2)
    print(scoreStr)
    

    fpOut = open(pathOutEval + "scoresOut." + modelName + "." + evalSetName + ".json", "w")
    fpOut.write(scoreStr)
    fpOut.close()

    # Save error analysis
    saveErrorAnalysisTSVFile(errorsOutPermissive, "", filenameTSVOut + "." + modelName + "." + evalSetName +".errorsOut.tsv")

    # while True:
    #     print("")
    #     prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

    #     if (len(prompt_text) < 1):
    #         exit()

    #     #def generate(promptIn, model, args, tokenizer):
    #     sequences = generate(prompt_text, model, args, tokenizer)
    #     print("Sequences: ")
    #     print(sequences)



if __name__ == "__main__":
    main()
