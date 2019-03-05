# -*- coding: utf-8 -*-

baseDir = "/mnt/hgfs/VMWareShare/"

# mode = "EMA"
# mode = "Shooter"
NER_SIA = "NER_SIA"
NER_SPLON = "NER_SPLON"
mode = NER_SIA
#mode = NER_SPLON

wordVecDim = 50

# expName = "20181003-21-PosTag_to_state"
# expName = "20181003-23-PosTag_to_state-NNPs"
# flagOfPosTagToState = True

flagOfCompareNNP = False
flagOfCtxProbs = False

# expName = "20181011-19-TaggerFe-CIS-qs0"
# flagOfGaze = False
# flagOfPos_se = False
# flagOfPos_ctx = 0
# flagOfFilter = False
# flagOfPLO = False
# flagOfCIS = True
# flagOfQS = False

# expName = "20181011-19-TaggerFe-CIS-qs0-diff0"
# flagOfGaze = False
# flagOfPos_se = False
# flagOfPos_ctx = 0
# flagOfFilter = False
# flagOfPLO = False
# flagOfCIS = True
# flagOfQS = False
# flagOfConfDiff = False

# expName = "20181011-19-TaggerFe-CIS-qs1-diff0"
# flagOfGaze = False
# flagOfPos_se = False
# flagOfPos_ctx = 0
# flagOfFilter = False
# flagOfPLO = False
# flagOfCIS = True
# flagOfQS = True
# flagOfConfDiff = False

# expName = "20181011-19-TaggerFe-CIS-qs1-diff1-overlap1"
# flagOfGaze = False
# flagOfPos_se = False
# flagOfPos_ctx = 0
# flagOfFilter = False
# flagOfPLO = False
# flagOfCIS = True
# flagOfQS = True
# flagOfConfDiff = True
# flagOfOverlap = True

# expName = "20190110-08-T3-Pse-Pctx1-CIS-qs1-diff1-overlap1"
# flagOfGaze = False
# flagOfPos_se = True
# flagOfPos_ctx = 1
# flagOfFilter = False
# flagOfPLO = False
# flagOfCIS = True
# flagOfQS = True
# flagOfConfDiff = True
# flagOfOverlap = True

# expName = "20190111-01-T3-Vse-Vctx1-CIS-qs1-diff1-overlap1"
flagOfGaze = False
flagOfPos_se = False
flagOfPos_ctx = 0
flagOfVec_se = False
flagOfVec_ctx = 0
flagOfFilter = False
flagOfPLO = False
flagOfCIS = True
flagOfQS = True
flagOfConfDiff = True
flagOfOverlap = True
flagOfPartTag = False

# expName = "20190111-14-T3-Vse-Vctx1-Penalty0.01"
# penalty = 0.01

# expName = "20190112-19-T7-Vse-Vctx1"
# expName = "20190112-19-T7-Vse-Vctx1-Train2Test"  # test overfitting
# expName = "20190112-19-T7-Vse-Vctx1-Train2Test-50-50"  # test overfitting
# expName = "20190112-19-T7-Vse-Vctx1-Train2Test-10-10"  # test overfitting
# expName = "20190112-19-T7-Vse-Vctx1-Train2Test-6-18"  # test overfitting
# expName = "20190113-09-T7-Pse-Pctx1-Vse-Vctx1"  # test overfitting
# expName = "20190113-18-T7-Pse-Pctx1-10-10-PartTag1"
# expName = "20190113-18-T7-Pse-Pctx1-10-10-R1.0"
# expName = "20190113-18-T7-Pse-Pctx1-20-20"
penalty = 0.1
port = 7000

# expName = "20190113-18-T7-Pse-Pctx1-20-20-PartTag1-EO1"
# expName = "20190113-18-T7-Pse-Pctx1-20-20-PartTag1-QO0"
# expName = "20190113-18-T7-Pse-Pctx1-20-20-PartTag0-QO1"
# expName = "20190113-18-T7-Pse-Pctx1-20-20-PartTag0-QO0"
expName = "20190113-18-T7-Pse-Pctx1-20-20-PartTag1-EO1-DEMO"

# expName = "20190217-14-T7-Vse-Vctx1-20-20-PartTag1-EO1"
# expName = "20190217-14-T7-Pse-Pctx1-Vse-Vctx1-20-20-PartTag1-EO1"

dataset = "AKWS1News_E"
# dataset = "CoNLLTest_RR"
# dataset = "Ontonotes_EN_Sub_R"

# dataset = "CoNLLTest_R"
# dataset = "CoNLLTestB_R"
# dataset = "CoNLL"
# dataset = "IEER"
# dataset = "Ontonotes_EN"

classifierModel='Bagging'
# classifierModel="DT_J48"
# classifierModel='RandomForest' #running
# classifierModel='SVM_LibSVM_RBF' #running

# classifierModel='LogisticRegression'
# classifierModel='NaiveBayes'
# classifierModel='MultilayerPerceptron'

if "-Pse-" in expName:
    flagOfPos_se= True

if "-Pctx1-" in expName:
    flagOfPos_ctx = True

if "-Vse-" in expName:
    flagOfVec_se = True

if "-Vctx1-" in expName:
    flagOfVec_ctx = True

if "-PartTag1" in expName:
    flagOfPartTag = True
    port = 17000

if "-EO1" in expName:
    port = 27000

# expName = "20190112-19-T7-Vse-Vctx1-50-50"  # test overfitting
# expName = "20190112-19-T7-Vse-Vctx1-10-10"  # test overfitting
# port = 17000
# expName = "20190112-19-T7-Vse-Vctx1-6-18"  # test overfitting
# port = 27000

# expName = "20181011-19-TaggerFe-CIS-qs1-diff1-overlap1-SPLON"
# flagOfGaze = False
# flagOfPos_se = False
# flagOfPos_ctx = 0
# flagOfFilter = False
# flagOfPLO = False
# flagOfCIS = True
# flagOfQS = True
# flagOfConfDiff = True
# flagOfOverlap = True
# mode = NER_SPLON


# expName = "20181011-19-TaggerFe-CIS"
# flagOfGaze = False
# flagOfPos_se = False
# flagOfPos_ctx = 0
# flagOfFilter = False
# flagOfPLO = False
# flagOfCIS = True



print "-------------------------------"
print "expName = ", expName
print "-------------------------------"
if flagOfFilter:
    expName+="-filter"


if dataset=="AKWS1News":
    port = port + 0
elif dataset=="AKWS1News_E":
    port = port + 0
elif dataset=="CoNLLTest_R":
    port = port + 100
elif dataset=="CoNLLTest_RR":
    port = port + 100
elif dataset=="CoNLLTestB_R":
    port = port + 100
elif dataset=="CoNLL":
    port = port + 400
elif dataset=="IEER":
    port = port + 200
elif dataset=="Ontonotes_EN_Sub_R":
    port = port + 300
else:
    raise RuntimeError("unexpected branch")


if classifierModel=='AdaBoostM1':
    port=port+10
elif classifierModel=='Bagging':
    port=port+20
elif classifierModel=='DT_J48':
    port=port+30
elif classifierModel=='LogisticRegression':
    port=port+40
elif classifierModel=='MultilayerPerceptron':
    port=port+50
elif classifierModel=='NaiveBayes':
    port=port+60
elif classifierModel=='RandomForest':
    port=port+70
elif classifierModel=='SVM_LibSVM_RBF':
    port=port+80
else:
    raise RuntimeError("unexpected branch")

port=port+flagOfPos_ctx
port=port+flagOfVec_ctx
if flagOfPos_se or flagOfVec_se:
    port=port+1000
if flagOfPLO:
    port=port+2000
if flagOfCIS:
    port=port+4000
if not flagOfConfDiff:
    port=port+8000
if flagOfQS:
    port=port+16000
if flagOfOverlap:
    port=port+1000


if mode == NER_SIA:
    # 从数字转换为各个槽
    int2slots = ['neType']
    tags2int = {
        'TAG':0,
        'neType':1
    }
    NUM_QUERY_TYPES = 3
    mapOfIntToQueryName = {0: 'Identical', 1: 'Container', 2: 'Part'}
    # NUM_QUERY_TYPES = 2
    # mapOfIntToQueryName = {0: '<1,1>CDQuerierByWord', 1: '<2,2>CDQuerierByWord'}
    mapOfIntToActionName = {1: "STOP_ACTION", 2: "IGNORE_ALL", 999: "ACCEPT_ALL"}

if mode == NER_SPLON:
    # 从数字转换为各个槽
    int2slots = ['neType']
    tags2int = {
        'TAG':0,
        'neType':1
    }
    NUM_QUERY_TYPES = 3
    mapOfIntToQueryName = {0: 'Identical', 1: 'Container', 2: 'Part'}
    mapOfIntToActionName = {0: "STOP_ACTION", 1: "PER", 2: "LOC", 3: "ORG", 4: "O"}
    mapOfActionNameToInt = {"STOP_ACTION": 0, "PER": 1, "LOC": 2, "ORG": 3, "O": 4}

elif mode == "Shooter":
    #for Shooter DB
    int2slots = ['shooterName', 'killedNum', 'woundedNum', 'city']
    tags2int = {'TAG':0,\
    'shooterName':1, \
    'killedNum':2, \
    'woundedNum':3, \
    'city' : 4 }

elif mode == "EMA":
    # for EMA
    int2slots = \
    ['Affected-Food-Product',\
    'Produced-Location',\
    'Adulterant(s)']
    tags2int = \
    {'TAG':0,\
    'Affected-Food-Product':1, \
    'Produced-Location':2, \
    'Adulterant(s)':3}
    int2citationFeilds = ['Authors', 'Date', 'Title', 'Source']
    generic = ["city", "centre", "county", "street", "road", "and", "in", "town", "village"]

posTags = [
"START",    # 0
":",
"CD",
"NNP",
"(",
"IN",       # 5
".",
"DT",
"CC",
",",
"VBD",		# 10
"JobTitle",
"TO",
"VB",
"RP",
"``",		# 15
")",
"JJ",
"VBZ",
"POS",
"NN",		# 20
"WDT",
"NNS",
"VBG",
"VBN",
"VBP",		# 25
"WRB",
"PRP$",
"JJS",
"RB",
"WP$",		# 30
"JJR",
"PRP",
"RBR",
"$",
"WP",		# 35
"MD",
"''",
"FW",
"SYM",
"NNPS",		# 40
"#",
"EX",
"LS",
"UH",
"END",		# 45
"RBS",
"PDT"
]