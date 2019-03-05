# -*- coding: utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import zmq, time
import numpy as np
import copy
import sys, json, pdb, pickle, operator, collections
import helper
#import predict as predict
from train import load_data
from itertools import izip
#import inflect
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from random import shuffle
from operator import itemgetter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
#from classifier import Classifier
import constants
import re

import random

import fileUtil

#DEBUG = False
ANALYSIS = False
COUNT_ZERO = False

#Global variables
int2slots = constants.int2slots
resDir = "/mnt/hgfs/VMWareShare/" + constants.expName + "/" + constants.dataset + "/" + constants.classifierModel + "/"
outDir = "/mnt/hgfs/VMWareShare/" + constants.expName + "/" + constants.dataset + "/" + constants.classifierModel + "/"
detailDir = outDir + "detail/"
fileUtil.mkdir(outDir)
fileUtil.mkdir(detailDir)
print "dataset=", constants.dataset
print "classifierModel=", constants.classifierModel
print "outDir=", outDir
print "detailDir=", detailDir
print "wordVecDim=", constants.wordVecDim
print "flagOfPLO=", constants.flagOfPLO
print "flagOfCompareNNP=", constants.flagOfCompareNNP
print "flagOfCtxProbs=", constants.flagOfCtxProbs
print "flagOfGaze=", constants.flagOfGaze
print "flagOfCIS=", constants.flagOfCIS
print "flagOfQS=", constants.flagOfQS
print "flagOfConfDiff=", constants.flagOfConfDiff
print "flagOfOverlap=", constants.flagOfOverlap
print ""
print "flagOfPos_se=", constants.flagOfPos_se
print "flagOfPos_ctx=", constants.flagOfPos_ctx
print "flagOfVec_se=", constants.flagOfVec_se
print "flagOfVec_ctx=", constants.flagOfVec_ctx
print "flagOfPartTag=", constants.flagOfPartTag
print "mode=", constants.mode

NUM_SLOTS = len(int2slots)        # 1
NUM_NETYPES = 4 # PER, LOC, ORG, O
#NUM_QUERY_TYPES = NUM_ENTITIES + 1  # title, title+entityType1, ...
NUM_QUERY_TYPES = constants.NUM_QUERY_TYPES  #
WORD_LIMIT = 1000
CONTEXT_LENGTH = 0
CONTEXT_TYPE = None
#STATE_SIZE = (1+NUM_NETYPES)*2 + 1 + NUM_QUERY_TYPES*2 # 5*2 + 1 + 3*2 = 17
STATE_SIZE = 17
if constants.flagOfConfDiff:
    STATE_SIZE += 1
if constants.flagOfQS:
    STATE_SIZE += 3
if constants.flagOfCtxProbs:
    STATE_SIZE += 16
if constants.flagOfPos_se:
    STATE_SIZE += (len(constants.posTags)*2*2)    # (src, ext) * (start, end)
if constants.flagOfPos_ctx>0:
    STATE_SIZE += (len(constants.posTags)*2*2*constants.flagOfPos_ctx)    # (src, ext) * (prev, succ) * ctxSize
if constants.flagOfVec_se:
    STATE_SIZE += (constants.wordVecDim*2*2)    # (src, ext) * (start, end)
if constants.flagOfVec_ctx>0:
    STATE_SIZE += (constants.wordVecDim*2*2*constants.flagOfVec_ctx)    # (src, ext) * (prev, succ) * ctxSize
if constants.flagOfGaze>0:
    STATE_SIZE += (2*4*3)    # (src, ext) * (prev, succ) * ctxSize
if constants.flagOfCompareNNP:
    STATE_SIZE += 2
if constants.flagOfPartTag:
    STATE_SIZE += 2*3*3

print "state_size=", STATE_SIZE

# 1,2,999
if constants.mode == "NER_SIA":
    STOP_ACTION = 1  # 停止、接受、拒绝，一共三个动作，之前的设计是：每个实体一个动作
    IGNORE_ALL = 2
    ACCEPT_ALL = 999 #arbitrary
else:
    STOP_ACTION = 0
    PER_ACTION  = 1
    LOC_ACTION  = 2
    ORG_ACTION  = 3
    O_ACTION    = 4

PENALTY = constants.penalty
# PENALTY = 0.001
#PENALTY = 0

# trained_model = None
# tfidf_vectorizer = TfidfVectorizer()
# inflect_engine = inflect.engine()

#resDir = "D:\\Research\\NER-RL\\Code\\lutm\\res\\"


def dd():
    return {}               # lutm: 小括号 元组；中括号 列表；大括号 键值对map

def ddd():
    return collections.defaultdict(dd)

#global caching to speed up
# TRAIN_TFIDF_MATRICES = {}
# TRAIN_EXT_ENTITIES = collections.defaultdict(dd)
# TRAIN_EXT_CONFIDENCES = collections.defaultdict(dd)

# TEST_TFIDF_MATRICES = {}
# TEST_EXT_ENTITIES = collections.defaultdict(dd)
# TEST_EXT_CONFIDENCES = collections.defaultdict(dd)

TRAIN_COSINE_SIM = collections.defaultdict(dd)
TRAIN_SRC_ENTITIES = collections.defaultdict(ddd)
TRAIN_SRC_CONFIDENCES = collections.defaultdict(ddd)
TRAIN_EXT_ENTITIES = collections.defaultdict(ddd)
TRAIN_EXT_CONFIDENCES = collections.defaultdict(ddd)
TRAIN_CONTEXT = collections.defaultdict(ddd) #final value will be a vector


TEST_COSINE_SIM =  collections.defaultdict(dd)
TEST_SRC_ENTITIES = collections.defaultdict(ddd)
TEST_SRC_CONFIDENCES = collections.defaultdict(ddd)
TEST_EXT_ENTITIES = collections.defaultdict(ddd)
TEST_EXT_CONFIDENCES = collections.defaultdict(ddd)
TEST_CONTEXT = collections.defaultdict(ddd) #final value will be a vector

CORRECT = collections.defaultdict(lambda:0.)
GOLD = collections.defaultdict(lambda:0.)
PRED = collections.defaultdict(lambda:0.)
EVALCONF = collections.defaultdict(lambda:[])
EVALCONF2 = collections.defaultdict(lambda:[])
QUERY = collections.defaultdict(lambda:0.)
ACTION = collections.defaultdict(lambda:0.)
CHANGES = 0
evalMode = False
STAT_POSITIVE, STAT_NEGATIVE = 0, 0 #stat. sign.

CONTEXT = None

def splitBars(w):
    return [q.strip() for q in w.split('|')]

#Environment for each episode
class Environment:
    # srcArticle: boundary
    # extArticles: a[qIdx][dldIdx]
    # goldEntities: [label]
    # srcArticleIdx
    # args
    # evalMode
    def __init__(self, srcMention, extArticles, goldEntities, srcArticleIdx, args, evalMode):
        self.srcArticleIdx = srcArticleIdx
        self.srcMention = srcMention
        self.extArticles = extArticles #extra articles to process
        self.goldEntities = goldEntities
        self.ignoreDuplicates = args.ignoreDuplicates
        self.entity = args.entity
        self.aggregate = args.aggregate
        self.delayedReward = args.delayedReward
        self.shooterLenientEval = args.shooterLenientEval
        self.queryIdx = 0 #start off with first list
        self.rlbasicEval = args.rlbasicEval    
        self.rlqueryEval = args.rlqueryEval    

        self.shuffledIndxs = [range(len(q)) for q in self.extArticles]  #newAtricles中第一维是查询模板类型q，q中有k篇文章
        if not evalMode and args.shuffleArticles:
            for q in self.shuffledIndxs:
                shuffle(q)

        self.state = [0 for i in range(STATE_SIZE)]
        self.terminal = False

        self.slotsCurrentlyHeld = collections.defaultdict(lambda: '') #current best entities
        self.confsCurrentlyHeld = collections.defaultdict(lambda:0.)
        self.bestEntitySet = None
        if self.aggregate == 'majority':
            self.bestEntitySet = collections.defaultdict(lambda:[]) #key不存在时的默认值
        self.bestIndex = (0,0)
        # ?
        self.prevQueryIdx = 0
        # ?
        self.prevArticleIndx = 0

        # to keep track of extracted values from previousArticle
        # 一开始ACCEPT_ALL接受的就是src_mention中的值和置信度
        self.slotsFromPrevArticle, self.confsFromPrevArticle = SRC_ENTITIES[self.srcArticleIdx], SRC_CONFIDENCES[self.srcArticleIdx]

        #store the original entities before updating state
        # 这里直接赋值为SRC_ENTITIES[self.srcArticleIdx]会更加清楚
        # 这个变量只是在每个情节一开始赋值，后面不再赋值，用于计算滞后的回报，也就是整个情节的回报
        self.entitiesInSrcArticle = SRC_ENTITIES[self.srcArticleIdx]

        # 这段代码的作用是构建数组COSINE_SIM[self.srcArticleIdx][qIdx]，即源文章和下载文章的相似度
        # 修改为直接返回1
        if self.srcArticleIdx not in COSINE_SIM:
            for qIdx, extArticleIdx in enumerate(self.extArticles):
                COSINE_SIM[self.srcArticleIdx][qIdx] = [1.]
        # 修改结束

        #update the initial state
        self.arrayOfNumOfUsedMentionsByQuery = [0 for q in range(NUM_QUERY_TYPES)]          #每个查询模板，有一个数字(被选择次数)，被初始化为0

        # 初始的查询，应该是有查询结果的查询中，随机选一个
        # query = queryIdx + 1
        mapOfMentionsByQueries = self.srcMention["mapOfMentionsByQueries"]
        queryIdxsHavingResults = []

        if evalMode and DEBUG:
            detailOutFile.write("\tAvailable  :")

        for queryIdx in range(NUM_QUERY_TYPES):
            queryName = constants.mapOfIntToQueryName[queryIdx]
            mentionsByQuery = mapOfMentionsByQueries[queryName]
            if evalMode and DEBUG:
                detailOutFile.write("\t" + str(len(mentionsByQuery)))
            if len(mentionsByQuery)>0:
                queryIdxsHavingResults.append(queryIdx)
        if len(queryIdxsHavingResults)==0:
            # arbitary
            # todo: -1
            initialQueryIdx = -1
            if evalMode and DEBUG:
                detailOutFile.write("\tNo available query: use " + str(initialQueryIdx))
        else:
            initialQueryIdx = random.choice(queryIdxsHavingResults)
            if evalMode and DEBUG:
                detailOutFile.write("\tUse " + str(initialQueryIdx))
        initialQuery = initialQueryIdx + 1
        if evalMode and DEBUG:
            detailOutFile.write("\n")

        if constants.mode==constants.NER_SIA:
            # 一开始ACCEPT_ALL接受的就是src_mention中的值和置信度
            initialActionIdx=ACCEPT_ALL
        if constants.mode==constants.NER_SPLON:
            initialActionName = self.srcMention["neType"]
            initialActionIdx= constants.mapOfActionNameToInt[initialActionName]

        self.updateState(initialActionIdx, initialQuery, self.ignoreDuplicates)


        return


    # def extractEntitiesWithConfidences4Ner(self, boundaryKey):
    #     clsRst = trainingMentionMap[boundaryKey]
    #     return clsRst.classifierLabel, clsRst.conf
    #
    #
    # def extractEntitiesWithConfidences4NerWithEK(self, boundary, queryIdx):
    #     clsRst = testingMentionMap[queryIdx][boundary]
    #     return [clsRst['classifierLabel']], [clsRst['conf']]


    # lutm: OK
    #find the article similarity between original and newArticle[i] (=allArticles[i+1])
    def articleSim(self, indx, queryIdx, i):
        # return cosine_similarity(self.tfidf_matrix[0:1], self.tfidf_matrix[i+1:i+2])[0][0]
        return COSINE_SIM[indx][queryIdx][i]

    # update the state based on the decision from DQN
    def updateState(self, action, query, ignoreDuplicates=False):
        global CONTEXT, CONTEXT_TYPE

        #use query to get next article
        extArticleIdx = None

        if self.rlbasicEval:    #false
            #ignore the query decision from the agent
            queryIdx = self.queryIdx  #在Environment类中，初始化为0
            self.queryIdx += 1
            if self.queryIdx == NUM_QUERY_TYPES: self.queryIdx = 0    #一轮到头，重新来
        else:
            queryIdx = query-1 #convert from 1-based to 0-based      #根据query获得索引

        if self.rlqueryEval:    #false
            #set the reconciliation action
            action = ACCEPT_ALL

        numOfMentionsByThisQuery = len(self.extArticles[queryIdx]) if queryIdx>=0 else 0
        numOfUsedMentionsByQuery = self.arrayOfNumOfUsedMentionsByQuery[queryIdx] if queryIdx>=0 else 0
        if ignoreDuplicates:    #false
            nextMentionByThisQuery = None
            while not nextMentionByThisQuery and numOfUsedMentionsByQuery < numOfMentionsByThisQuery:
                extArticleIdx = self.shuffledIndxs[queryIdx][self.arrayOfNumOfUsedMentionsByQuery]
                if self.articleSim(self.srcArticleIdx, queryIdx, extArticleIdx) < 0.95:
                    nextMentionByThisQuery = self.extArticles[queryIdx][extArticleIdx]
                else:
                    numOfUsedMentionsByQuery += 1
        else:
            #get next article            
            if evalMode and DEBUG:
                detailOutFile.write("[q" + str(queryIdx) + "]" + str(numOfUsedMentionsByQuery + 1) + "/" + str(numOfMentionsByThisQuery))
            if numOfUsedMentionsByQuery < numOfMentionsByThisQuery:  #对1个查询类型来说，是否还有没step过的
                # extArticleIdx = self.shuffledIndxs[queryIdx][self.arrayOfNumOfUsedMentionsByQuery[queryIdx]]
                extArticleIdx = numOfUsedMentionsByQuery
                nextMentionByThisQuery = self.extArticles[queryIdx][extArticleIdx]
                if evalMode and DEBUG:
                    detailOutFile.write("\tnextMentionByThisQuery=" + mentionToStr(nextMentionByThisQuery) + "\n")
                # 这个代码的位置很重要，不要移动它
                self.arrayOfNumOfUsedMentionsByQuery[queryIdx] += 1
            else:
                nextMentionByThisQuery = None
                if evalMode and DEBUG:
                    detailOutFile.write("\tnextMentionByThisQuery=None\n")

        # 如果是stop，那么不用更新抽取的值，维持原来的就行了
        if action != STOP_ACTION:
            ################################################### [BEGIN] update entities and confidences ###############3
            # integrate the values into the current DB state
            # all other tags
            for slotIdx in range(NUM_SLOTS):   #对每个属性循环，i必须与action一一对应
                # if action != ACCEPT_ALL and slotIdx != action: continue #only perform update for the entity chosen by agent
                if constants.mode==constants.NER_SIA:
                    if action==IGNORE_ALL: continue

                self.bestIndex = (self.prevQueryIdx, self.prevArticleIndx) #analysis
                # accept an entity
                if self.aggregate == 'majority':
                    self.bestEntitySet[slotIdx].append((self.slotsFromPrevArticle[slotIdx], self.confsFromPrevArticle[slotIdx])) #括号内的属性值与置信度构成一个元组
                    self.slotsCurrentlyHeld[slotIdx], self.confsCurrentlyHeld[slotIdx] = self.majorityVote(self.bestEntitySet[slotIdx])
                else:
                    if constants.mode==constants.NER_SPLON:
                        self.slotsCurrentlyHeld[slotIdx] = constants.mapOfIntToActionName[action]
                        self.confsCurrentlyHeld[slotIdx] = self.srcMention["prob"]
                    else:
                        if slotIdx==0:    #下面三个分支的代码一样啊！
                            #handle shooterName -  add to list or directly replace
                            if not self.slotsCurrentlyHeld[slotIdx]:
                                self.slotsCurrentlyHeld[slotIdx] = self.slotsFromPrevArticle[slotIdx]
                                self.confsCurrentlyHeld[slotIdx] = self.confsFromPrevArticle[slotIdx]
                            elif self.aggregate == 'always' or self.confsFromPrevArticle[slotIdx] > self.confsCurrentlyHeld[slotIdx]:
                                self.slotsCurrentlyHeld[slotIdx] = self.slotsFromPrevArticle[slotIdx] #directly replace
                                # self.bestEntities[i] = self.bestEntities[i] + '|' + entities[i] #add to list
                                self.confsCurrentlyHeld[slotIdx] = self.confsFromPrevArticle[slotIdx]
                        else:
                            raise RuntimeError("unexpected branch")
                            # if not self.bestEntities[i] or self.aggregate == 'always' or confidences[i] > self.bestConfidences[i]:
                                # self.bestEntities[i] = entities[i]
                                # self.bestConfidences[i] = confidences[i]
                                # print "Changing best Entities"
                                # print "New entities", self.bestEntities
            # if DEBUG:
            #     print "entitySet:", self.bestEntitySet
            ################################################### [END] update entities and confidences ###############3

        # todo: 当nextMentionByThisQuery为空，维持上一次的状态，等于空转一次（reward-=0.001）
        if nextMentionByThisQuery and action != STOP_ACTION:
            #!!!!!! next
            assert(extArticleIdx != None)
            slotsFromNextExtArticle, confsFromNextExtArticle = [nextMentionByThisQuery['neType']], [nextMentionByThisQuery['prob']]
            if constants.flagOfCtxProbs:
                ctxProbsFromNextExtArticle = nextMentionByThisQuery['ctxProbs']
            assert(len(slotsFromNextExtArticle) == len(confsFromNextExtArticle))
        else:
            # print "No next article"
            slotsFromNextExtArticle, confsFromNextExtArticle, ctxProbsFromNextExtArticle = [""] * NUM_SLOTS, [0] * NUM_SLOTS, [0] * 16
            self.terminal = True

        #modify self.state appropriately
        # print(self.bestEntities, entities)
        if constants.mode == 'Shooter':
            matches = map(self.checkEquality, self.slotsCurrentlyHeld.values()[1:-1], slotsFromNextExtArticle[1:-1])
            matches.insert(0, self.checkEqualityShooter(self.slotsCurrentlyHeld.values(), slotsFromNextExtArticle))
            matches.append(self.checkEqualityCity(self.slotsCurrentlyHeld.values()[-1], slotsFromNextExtArticle[-1]))
        else:
            matches = map(self.checkEqualityShooter, self.slotsCurrentlyHeld.values(), slotsFromNextExtArticle)
            # entities是上一步结果（current）
            # bestEntities是新抽取结果（new）
            # map中第一个参数是函数f，后面的参数都是列表，f(p1[i], p2[i])
            # 以上调用返回匹配的属性数量

        ################################# state start ################################
        # numOfStateDim = 2 * numOfEntity + 2 * numOfEntity + 1 + 2 * numOfEntity * ctxLen
        #                 old, new confs    match, mismatch  sim  pre, sub context words
        #               = numOfEntity*2*(2+ctxLen)+1
        #       shooter   4*2*(2+3)+1=41
        #           xxx   3*2*(2+3)+1=31
        #           NER   1*2*(2+3)+1=11
        # pdb.set_trace()
        self.state = [0 for slotIdx in range(STATE_SIZE)]

        stateStr = ""
        mapOfMentionsByQueries = self.srcMention["mapOfMentionsByQueries"]
        ################################# state seg 0: newConfidences
        ################################# state seg 1: curConfidences
        idxInState = -1
        for slotIdx in range(NUM_SLOTS):
            if constants.flagOfConfDiff:
                if evalMode and DEBUG:
                    stateStr += "\tConfDiff:"
                idxInState+=1
                diff = confsFromNextExtArticle[slotIdx]-self.confsCurrentlyHeld[slotIdx]
                self.state[idxInState] = (diff/2)+0.5
                if evalMode and DEBUG:
                    stateStr+="\t["+format(diff*1.0, "0.3f")+"]\t"+format(self.state[idxInState]*1.0, "0.3f")

            if evalMode and DEBUG:
                stateStr += "\n\tCurrentHeld:"
            idxInState+=1
            self.state[idxInState] = self.confsCurrentlyHeld[slotIdx] #DB state
            if evalMode and DEBUG:
                stateStr+="\t"+format(self.state[idxInState]*1.0, "0.3f")

            idxInState+=1
            self.state[idxInState] = 1 if self.slotsCurrentlyHeld[slotIdx] == 'PER' else 0
            if evalMode and DEBUG:
                stateStr+="\t"+str(self.state[idxInState])

            idxInState+=1
            self.state[idxInState] = 1 if self.slotsCurrentlyHeld[slotIdx] == 'LOC' else 0
            if evalMode and DEBUG:
                stateStr+="\t"+str(self.state[idxInState])

            idxInState+=1
            self.state[idxInState] = 1 if self.slotsCurrentlyHeld[slotIdx] == 'ORG' else 0
            if evalMode and DEBUG:
                stateStr+="\t"+str(self.state[idxInState])

            idxInState+=1
            self.state[idxInState] = 1 if self.slotsCurrentlyHeld[slotIdx] == 'O'   else 0
            if evalMode and DEBUG:
                stateStr+="\t"+str(self.state[idxInState])

            if constants.flagOfPartTag:
                idxInState+=1
                self.state[idxInState] = 1 if self.srcMention["inPER"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if self.srcMention["inLOC"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if self.srcMention["inORG"] else 0
                if evalMode and DEBUG:
                    stateStr += "\tin: "+str(self.state[idxInState-2])+" "+str(self.state[idxInState-1])+" "+str(self.state[idxInState])
                idxInState+=1
                self.state[idxInState] = 1 if self.srcMention["containsPER"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if self.srcMention["containsLOC"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if self.srcMention["containsORG"] else 0
                if evalMode and DEBUG:
                    stateStr += "\tcontains: "+str(self.state[idxInState-2])+" "+str(self.state[idxInState-1])+" "+str(self.state[idxInState])
                idxInState+=1
                self.state[idxInState] = 1 if self.srcMention["hasOverlapWithPER"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if self.srcMention["hasOverlapWithLOC"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if self.srcMention["hasOverlapWithORG"] else 0
                if evalMode and DEBUG:
                    stateStr += "\toverlap: "+str(self.state[idxInState-2])+" "+str(self.state[idxInState-1])+" "+str(self.state[idxInState])

            # 上下文词性
            # stateStr += "\tCtxPos: "
            # idxInState += 1
            # self.state[idxInState] = 1 if self.srcMention["prevPosTag"]=="NNP" else 0
            # stateStr += "\t" + str(self.state[idxInState])
            # idxInState += 1
            # self.state[idxInState] = 1 if self.srcMention["succPosTag"]=="NNP" else 0
            # stateStr += "\t" + str(self.state[idxInState])

            if constants.flagOfQS:
                if evalMode and DEBUG:
                    stateStr+= "\tLastQuery: "
                for queryIdxForLoop in range(NUM_QUERY_TYPES):
                    queryForLoop = queryIdxForLoop + 1
                    idxInState += 1
                    self.state[idxInState] = 1 if queryForLoop == query else 0
                    if evalMode and DEBUG:
                        stateStr += "\t" + str(self.state[idxInState])

            if evalMode and DEBUG:
                stateStr += "\n\tByLastQuery:"
            idxInState+=1
            self.state[idxInState] = confsFromNextExtArticle[slotIdx]  #IMP: (original) next article state
            if evalMode and DEBUG:
                stateStr+="\t"+format(self.state[idxInState]*1.0, "0.3f")

            idxInState+=1
            self.state[idxInState] = 1 if slotsFromNextExtArticle[slotIdx] == 'PER' else 0
            if evalMode and DEBUG:
                stateStr+="\t"+str(self.state[idxInState])

            idxInState+=1
            self.state[idxInState] = 1 if slotsFromNextExtArticle[slotIdx] == 'LOC' else 0
            if evalMode and DEBUG:
                stateStr+="\t"+str(self.state[idxInState])

            idxInState+=1
            self.state[idxInState] = 1 if slotsFromNextExtArticle[slotIdx] == 'ORG' else 0
            if evalMode and DEBUG:
                stateStr+="\t"+str(self.state[idxInState])

            idxInState+=1
            self.state[idxInState] = 1 if slotsFromNextExtArticle[slotIdx] == 'O'   else 0
            if evalMode and DEBUG:
                stateStr+= "\t" + str(self.state[idxInState])

            if constants.flagOfPartTag:
                idxInState+=1
                self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["inPER"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["inLOC"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["inORG"] else 0
                if evalMode and DEBUG:
                    stateStr += "\tin: "+str(self.state[idxInState-2])+" "+str(self.state[idxInState-1])+" "+str(self.state[idxInState])
                idxInState+=1
                self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["containsPER"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["containsLOC"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["containsORG"] else 0
                if evalMode and DEBUG:
                    stateStr += "\tcontains: "+str(self.state[idxInState-2])+" "+str(self.state[idxInState-1])+" "+str(self.state[idxInState])
                idxInState+=1
                self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["hasOverlapWithPER"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["hasOverlapWithLOC"] else 0
                idxInState+=1
                self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["hasOverlapWithORG"] else 0
                if evalMode and DEBUG:
                    stateStr += "\toverlap: "+str(self.state[idxInState-2])+" "+str(self.state[idxInState-1])+" "+str(self.state[idxInState])
            # 上下文词性
            # stateStr += "\tCtxPos: "
            # idxInState += 1
            # self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["prevPosTag"]=="NNP" else 0
            # stateStr += "\t" + str(self.state[idxInState])
            # idxInState += 1
            # self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["succPosTag"]=="NNP" else 0
            # stateStr += "\t" + str(self.state[idxInState])

        ################################# state seg 2: matches
        ################################# state seg 3: mismatches
            matchScore = float(matches[slotIdx])
            idxInState+=1
            if matchScore > 0:  # 分大于零和等于零两种情况，占不同的位置，采用0/1独热编码
                self.state[idxInState] = 1
            else:
                self.state[idxInState] = 0
            if evalMode and DEBUG:
                stateStr+= "\n\tMatch      : "
            if evalMode and DEBUG:
                stateStr+= "\t" + str(self.state[idxInState])

            if constants.flagOfPos_ctx>=2:
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.prevPosTag2      : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if self.srcMention["prevPosTag2"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\text.prevPosTag2      : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["prevPosTag2"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
            if constants.flagOfPos_ctx>=1:
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.prevPosTag1      : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if self.srcMention["prevPosTag1"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\text.prevPosTag1      : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["prevPosTag1"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
            if constants.flagOfPos_se:
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.startPosTag      : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if self.srcMention["startPosTag"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\text.startPosTag      : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["startPosTag"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.endPosTag        : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if self.srcMention["endPosTag"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\text.endPosTag        : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["endPosTag"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
            if constants.flagOfPos_ctx>=1:
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.succPosTag1      : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if self.srcMention["succPosTag1"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\text.succPosTag1      : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["succPosTag1"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
            if constants.flagOfPos_ctx>=2:
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.succPosTag2      : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if self.srcMention["succPosTag2"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\text.succPosTag2      : "
                for posTag in constants.posTags:
                    idxInState += 1
                    self.state[idxInState] = 1 if nextMentionByThisQuery and nextMentionByThisQuery["succPosTag2"] == posTag else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])


            if constants.flagOfVec_ctx>=1:
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.prevVec1      : "
                for idxInWordVec in range(constants.wordVecDim):
                    idxInState += 1
                    self.state[idxInState] = self.srcMention["prevVec1"][idxInWordVec]
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\text.prevVec1      : "
                for idxInWordVec in range(constants.wordVecDim):
                    idxInState += 1
                    self.state[idxInState] = nextMentionByThisQuery["prevVec1"][idxInWordVec] if nextMentionByThisQuery else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
            if constants.flagOfVec_se:
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.startVec      : "
                for idxInWordVec in range(constants.wordVecDim):
                    idxInState += 1
                    self.state[idxInState] = self.srcMention["startVec"][idxInWordVec]
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\text.startVec      : "
                for idxInWordVec in range(constants.wordVecDim):
                    idxInState += 1
                    self.state[idxInState] = nextMentionByThisQuery["startVec"][idxInWordVec] if nextMentionByThisQuery else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.endVec        : "
                for idxInWordVec in range(constants.wordVecDim):
                    idxInState += 1
                    self.state[idxInState] = self.srcMention["endVec"][idxInWordVec]
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\text.endVec        : "
                for idxInWordVec in range(constants.wordVecDim):
                    idxInState += 1
                    self.state[idxInState] = nextMentionByThisQuery["endVec"][idxInWordVec] if nextMentionByThisQuery else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
            if constants.flagOfVec_ctx>=1:
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.succVec1      : "
                for idxInWordVec in range(constants.wordVecDim):
                    idxInState += 1
                    self.state[idxInState] = self.srcMention["succVec1"][idxInWordVec]
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])
                if evalMode and DEBUG:
                    stateStr+= "\n\text.succVec1      : "
                for idxInWordVec in range(constants.wordVecDim):
                    idxInState += 1
                    self.state[idxInState] = nextMentionByThisQuery["succVec1"][idxInWordVec] if nextMentionByThisQuery else 0
                    if evalMode and DEBUG:
                        stateStr+= " " + str(self.state[idxInState])

            if constants.flagOfGaze:
                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.jobTitle      : "
                idxInState += 1
                self.state[idxInState] = self.srcMention["prev1JobTitle"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = self.srcMention["startJobTitle"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = self.srcMention["endJobTitle"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = self.srcMention["succ1JobTitle"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])

                if evalMode and DEBUG:
                    stateStr+= "\n\text.jobTitle      : "
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["prev1JobTitle"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["startJobTitle"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["endJobTitle"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["succ1JobTitle"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])

                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.locPrefix      : "
                idxInState += 1
                self.state[idxInState] = self.srcMention["prev1LocPrefix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = self.srcMention["startLocPrefix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = self.srcMention["endLocPrefix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = self.srcMention["succ1LocPrefix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])

                if evalMode and DEBUG:
                    stateStr+= "\n\text.locPrefix      : "
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["prev1LocPrefix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["startLocPrefix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["endLocPrefix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["succ1LocPrefix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])

                if evalMode and DEBUG:
                    stateStr+= "\n\tsrc.locSuffix      : "
                idxInState += 1
                self.state[idxInState] = self.srcMention["prev1LocSuffix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = self.srcMention["startLocSuffix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = self.srcMention["endLocSuffix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = self.srcMention["succ1LocSuffix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])

                if evalMode and DEBUG:
                    stateStr+= "\n\text.locSuffix      : "
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["prev1LocSuffix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["startLocSuffix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["endLocSuffix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = 0 if not nextMentionByThisQuery else nextMentionByThisQuery["succ1LocSuffix"]
                if evalMode and DEBUG:
                    stateStr+= " " + str(self.state[idxInState])

            if constants.flagOfCompareNNP:
                # 上下文NNP，要么都是，要么都不是
                if evalMode and DEBUG:
                    stateStr+= "\t\t\t\t\tCtxPosMatch: "
                idxInState += 1
                self.state[idxInState] = \
                    1 if nextMentionByThisQuery and \
                         ((self.srcMention["prevPosTag"]=="NNP" and nextMentionByThisQuery["prevPosTag"] == "NNP") or
                                              (self.srcMention["prevPosTag"]!="NNP" and nextMentionByThisQuery["prevPosTag"] != "NNP"))\
                    else 0
                if evalMode and DEBUG:
                    stateStr+= "\t" + str(self.state[idxInState])
                idxInState += 1
                self.state[idxInState] = \
                    1 if nextMentionByThisQuery and \
                         ((self.srcMention["succPosTag"]=="NNP" and nextMentionByThisQuery["succPosTag"] == "NNP") or
                                              (self.srcMention["succPosTag"]!="NNP" and nextMentionByThisQuery["succPosTag"] != "NNP"))\
                    else 0
                if evalMode and DEBUG:
                    stateStr+= "\t" + str(self.state[idxInState])

            # 这个新查询到的指称，与初始的指称位置交叉吗？
            idxInState += 1
            if constants.flagOfOverlap:
                self.state[idxInState] = 1 if nextMentionByThisQuery and self.overlap(self.srcMention, nextMentionByThisQuery) else 0
            else:
                self.state[idxInState] = 1 if nextMentionByThisQuery and (nextMentionByThisQuery.directOverlap or nextMentionByThisQuery.indirectOverlap) else 0
            if evalMode and DEBUG:
                stateStr+= "\n\tOverlap    : "
            if evalMode and DEBUG:
                stateStr+= "\t" + str(self.state[idxInState])

            if constants.flagOfCtxProbs:
                ctxProbs = ctxProbsFromNextExtArticle
                if evalMode and DEBUG:
                    stateStr+= "\n\tctxProbs    : "
                for ctxProbIdx in range(16):
                    idxInState += 1
                    self.state[idxInState] = ctxProbs[ctxProbIdx]
                    if evalMode and DEBUG:
                        stateStr+= "\t" + str(self.state[idxInState])

            # self.state[2*NUM_ENTITIES+i] = float(matches[i])*confidences[i] if float(matches[i])>0 else -1*confidences[i]
        ################################# state seg 4: similarity
        # if nextMentionByThisQuery:
            # print self.srcArticleIdx, queryIdx, extArticleIdx
            # print COSINE_SIM[self.srcArticleIdx][queryIdx]
            # self.state[4 * NUM_SLOTS] = self.articleSim(self.srcArticleIdx, queryIdx, extArticleIdx)
        # else:
        #     self.state[4 * NUM_SLOTS] = 0
        ################################# state seg 4: 各种查询结果的剩余数量
        if evalMode and DEBUG:
            stateStr+= "\n\tAvailable  :\t"
        for queryIdxForLoop in range(NUM_QUERY_TYPES):
            numOfUsedMentionsByQueryForLoop = self.arrayOfNumOfUsedMentionsByQuery[queryIdxForLoop]
            queryNameForLoop = constants.mapOfIntToQueryName[queryIdxForLoop]
            mentionsByQueryForLoop = mapOfMentionsByQueries[queryNameForLoop]
            numOfMentionsByQueryForLoop = len(mentionsByQueryForLoop)
            idxInState+=1
            self.state[idxInState] = 1 if numOfUsedMentionsByQueryForLoop < numOfMentionsByQueryForLoop else 0
            if evalMode and DEBUG:
                stateStr+= str(self.state[idxInState]) + "\t"

        if evalMode and DEBUG:
            detailOutFile.write("STATE\t=\n" + stateStr + "\n")

        #selectively mask states
        # NUM_ENTITIES = len(int2tags)
        # self.entity = args.entity, 已经有int2tags，这个entity似乎没有意义
        if self.entity != NUM_SLOTS:
            raise RuntimeError("self.entity(" + self.entity +") != NUM_ENTITIES(" + NUM_SLOTS + ")")
            # for j in range(NUM_SLOTS):
            #     if j != self.entity:
            #         self.state[j] = 0
            #         self.state[NUM_SLOTS + j] = 0
            #         TODO: mask matches

        #add in context information
        # if nextMentionByThisQuery and CONTEXT_TYPE != 0:
        #     j = 4 * NUM_SLOTS + 1
        #     for q in range(NUM_SLOTS):
        #         if self.entity == NUM_SLOTS or self.entity == q:
                    #self.state[j:j+2*CONTEXT_LENGTH] = CONTEXT[self.srcArticleIdx][queryIdx][extArticleIdx+1][q]
                    # self.state[j:j+2*CONTEXT_LENGTH] = [0]*(2*CONTEXT_LENGTH)
                # j += 2*CONTEXT_LENGTH

        # pdb.set_trace()
        ################################# state end ################################

        #update state variables
        self.slotsFromPrevArticle = slotsFromNextExtArticle        # 属性值
        self.confsFromPrevArticle = confsFromNextExtArticle  # 置信度
        self.prevQueryIdx = queryIdx          # 查询类型
        self.prevArticleIndx = extArticleIdx  # 额外文章索引

        return

    # check if two entities are equal. Need to handle city
    def checkEquality(self, e1, e2):
        # if gold is unknown, then dont count that
        return e2!='' and (COUNT_ZERO or e2 != 'zero')  and e1.lower() == e2.lower()

    # lutm: OK
    #mode=ner，非shooter模式也会进此分支
    #其实这个分支的意思是：完全一样，就算对，否则算错
    def checkEqualityShooter(self, e1, e2):
        if e2 == '' or e2=='unknown': return 0.

        gold = set(splitBars(e2.lower()))
        pred = set(splitBars(e1.lower()))
        correct = len(gold.intersection(pred))
        prec = float(correct)/len(pred)
        rec = float(correct)/len(gold)

        if self.shooterLenientEval:
            if correct > 0:
                return 1.
            else:
                return 0.
        else:
            if prec+rec > 0:
                f1 = (2*prec*rec)/(prec+rec)
            else:
                f1 = 0.
            return f1

    # lutm: OK
    def checkEqualityCity(self, e1, e2):
        return e2!='' and e1.lower() == e2.lower()


    # lutm: OK
    def calculateReward(self, oldEntities, newEntities):
        if constants.mode == 'Shooter':
            rewards = [int(self.checkEquality(newEntities[1], self.goldEntities[1])) - int(self.checkEquality(oldEntities[1], self.goldEntities[1])),
                        int(self.checkEquality(newEntities[2], self.goldEntities[2])) - int(self.checkEquality(oldEntities[2], self.goldEntities[2]))]


            #add in shooter reward
            if self.goldEntities[0]:
                rewards.insert(0, self.checkEqualityShooter(newEntities[0], self.goldEntities[0]) \
                        - self.checkEqualityShooter(oldEntities[0], self.goldEntities[0]))
            else:
                rewards.insert(0, 0.)

            # add in city reward
            rewards.append(self.checkEqualityCity(newEntities[-1], self.goldEntities[-1]) \
                    - self.checkEqualityCity(oldEntities[-1], self.goldEntities[-1]))
        else:
            #进此分支
            rewards = []
            for i in range(len(newEntities)):
                if self.goldEntities[i] != 'unknown':
                    rewards.append(self.checkEqualityShooter(newEntities[i], self.goldEntities[i]) - self.checkEqualityShooter(oldEntities[i], self.goldEntities[i]))
                else:
                    raise RuntimeError("unexpected branch")

        if self.entity == NUM_SLOTS:
            return sum(rewards)
        else:
            raise RuntimeError("unexpected branch")
            # return rewards[self.entity]

    def overlap(self, cdSpan0, cdSpan1):
        if cdSpan0["docId"] == cdSpan1["docId"]:
            start0 = cdSpan0["start"]
            start1 = cdSpan1["start"]
            end0 = cdSpan0["end"]
            end1 = cdSpan1["end"]
            if start0 < start1:
                if end0 > start1:
                    return True
                else:
                    return False
            else:
                if start0 == start1:
                    return True
                else:
                    if start0 < end1:
                        return True
                    else:
                        return False
        else:
            return False


# lutm: OK
    def calculateStatSign(self, oldEntities, newEntities):
        if constants.mode == 'Shooter':
            rewards = [int(self.checkEquality(newEntities[1], self.goldEntities[1])) - int(self.checkEquality(oldEntities[1], self.goldEntities[1])),
                        int(self.checkEquality(newEntities[2], self.goldEntities[2])) - int(self.checkEquality(oldEntities[2], self.goldEntities[2]))]


            #add in shooter reward
            if self.goldEntities[0]:
                rewards.insert(0, self.checkEqualityShooter(newEntities[0], self.goldEntities[0]) \
                        - self.checkEqualityShooter(oldEntities[0], self.goldEntities[0]))
            else:
                rewards.insert(0, 0.)

            # add in city reward
            rewards.append(self.checkEqualityCity(newEntities[-1], self.goldEntities[-1]) \
                    - self.checkEqualityCity(oldEntities[-1], self.goldEntities[-1]))
        else:
            rewards = []
            for i in range(len(newEntities)):
                if self.goldEntities[i] != 'unknown':
                    rewards.append(self.checkEqualityShooter(newEntities[i], self.goldEntities[i]) - self.checkEqualityShooter(oldEntities[i], self.goldEntities[i]))
                else:
                    rewards.append(0.)

        return rewards
        


    # lutm: OK
    #evaluate the bestEntities retrieved so far for a single article
    #IMP: make sure the evaluate variables are properly re-initialized
    def evaluateArticle(self, predEntities, goldEntities, shooterLenientEval, shooterLastName, evalOutFile):
        # print "Evaluating article", predEntities, goldEntities

        if constants.mode == 'Shooter':
            #shooterName first: only add this if gold contains a valid shooter
            if goldEntities[0]!='':
                if shooterLastName:
                    gold = set(splitBars(goldEntities[0].lower())[-1:])
                else:
                    gold = set(splitBars(goldEntities[0].lower()))

                pred = set(splitBars(predEntities[0].lower()))
                correct = len(gold.intersection(pred))

                if shooterLenientEval:
                    CORRECT[int2slots[0]] += (1 if correct > 0 else 0)
                    GOLD[int2slots[0]] += (1 if len(gold) > 0 else 0)
                    PRED[int2slots[0]] += (1 if len(pred) > 0 else 0)
                else:
                    CORRECT[int2slots[0]] += correct
                    GOLD[int2slots[0]] += len(gold)
                    PRED[int2slots[0]] += len(pred)



            #all other tags
            for i in range(1, NUM_SLOTS):
                if COUNT_ZERO or goldEntities[i] != 'zero':
                    # gold = set(goldEntities[i].lower().split())
                    # pred = set(predEntities[i].lower().split())
                    # correct = len(gold.intersection(pred))
                    # GOLD[int2tags[i]] += len(gold)
                    # PRED[int2tags[i]] += len(pred)
                    GOLD[int2slots[i]] += 1
                    PRED[int2slots[i]] += 1
                    if predEntities[i].lower() == goldEntities[i].lower():
                        CORRECT[int2slots[i]] += 1
        else:
            #all other tags
            for i in range(NUM_SLOTS):
                if goldEntities[i] != 'unknown':

                    #old eval
                    gold = set(splitBars(goldEntities[i].lower()))
                    pred = set(splitBars(predEntities[i].lower()))
                    # if 'unknown' in pred:
                    # pred = set()
                    correct = len(gold.intersection(pred))

                    if shooterLenientEval:
                        CORRECT[int2slots[i]] += (1 if correct > 0 else 0)
                        GOLD[int2slots[i]] += (1 if len(gold) > 0 else 0)
                        PRED[int2slots[i]] += (1 if len(pred) > 0 else 0)
                    else:
                        # 这个是真正执行的代码
                        CORRECT[int2slots[i]] += correct
                        GOLD[int2slots[i]] += len(gold)
                        PRED[int2slots[i]] += len(pred)

                    # print i, pred, "###", gold, "$$$", correct                    

                    #new eval (Adam)
                    # pred = predEntities[i].lower()
                    # gold = goldEntities[i].lower()
                    # if pred in gold:
                    #     CORRECT[int2tags[i]] += 1
                    # GOLD[int2tags[i]] += 1
                    # if pred != 'unknown':
                    #     PRED[int2tags[i]] += 1


        if evalOutFile:
            goldType = list(gold)[0].upper()
            srcType = self.srcMention["neType"].upper()
            predType = list(pred)[0].upper()
            srcCorrectStr  = "T" if goldType==srcType  else "F"
            predCorrectStr = "T" if goldType==predType else "F"
            labelTrans = srcCorrectStr+"->"+predCorrectStr
            label_trans_map[labelTrans] += 1
            msg = "\n" + goldType \
                  + "\t" + srcType \
                  + "\t" + format(self.srcMention["prob"],"0.3f") \
                  + "\t" + predType \
                  + "\t" + labelTrans \
                  + "\t" + str(self.srcMention["docId"]) \
                  + "\t" + str(self.srcMention["start"]) \
                  + "\t" + str(self.srcMention["end"]) \
                  + "\t" + self.srcMention["name"] + "\n"
            evalOutFile.write(msg)


    # lutm: OK
    #TODO for EMA
    #TODO: use conf or 1 for mode calculation
    def majorityVote(self, entityList):
        if not entityList: return '',0.

        dic = collections.defaultdict(lambda:0.)
        confDic = collections.defaultdict(lambda:0.)
        cnt = collections.defaultdict(lambda:0.)
        ticker = 0
        for entity, conf in entityList:
            dic[entity] += 1
            cnt[entity] += 1
            confDic[entity] += conf
            if ticker == 0: dic[entity] += 0.1 #extra for original article to break ties
            ticker += 1

        bestEntity, bestVote = sorted(dic.items(), key=itemgetter(1), reverse=True)[0]
        return bestEntity, confDic[bestEntity]/cnt[bestEntity]


    # lutm: OK
    #take a single step in the episode
    # action和query从1开始索引
    def step(self, action, query):
        global CHANGES
        oldExtractedValue = copy.copy(self.slotsCurrentlyHeld.values())
        if evalMode and DEBUG:
            detailOutFile.write("oldValue=" + oldExtractedValue[0] + "\n")

        #update pointer to next article
        queryIdx = query-1

        self.updateState(action, query, self.ignoreDuplicates)

        newExtractedValue = self.slotsCurrentlyHeld.values()
        if evalMode and DEBUG:
            detailOutFile.write("newValue=" + newExtractedValue[0] + "\n")

        if self.delayedReward == 'True':
            reward = self.calculateReward(self.entitiesInSrcArticle, newExtractedValue)
        else:
            reward = self.calculateReward(oldExtractedValue, newExtractedValue)
        if evalMode and DEBUG:
            detailOutFile.write("reward\t=" + format(reward * 1.0, "0.3"))
            if reward!=0:
                detailOutFile.write("\tq=" + str(queryIdx))
            detailOutFile.write("\n")

        # negative per step
        reward -= PENALTY

        return self.state, reward, self.terminal


# lutm: NA
# called in evalEnd()
def plot_hist(evalconf, name):
    for i in evalconf.keys():
        plt.hist(evalconf[i], bins=[0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.title("Gaussian Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(name+"_"+str(i)+".png")
        plt.clf()


# lutm: NA
def splitDict(dict, start, end):
        return [dict[i] for i in range(start, end)]


#lutm: OK
def main(args):  # lutm: Python 中，一个变量的作用域总是由在代码中被赋值的地方所决定的。函数定义的是本地作用域，而模块定义的是全局作用域。
    global SRC_ENTITIES, EXT_ENTITIES, SRC_CONFIDENCES, EXT_CONFIDENCES, COSINE_SIM, CONTEXT  # lutm: 如果想要在函数内定义全局作用域，需要加上global修饰符。
    global TRAIN_SRC_ENTITIES, TRAIN_SRC_CONFIDENCES, TRAIN_EXT_ENTITIES, TRAIN_EXT_CONFIDENCES, TRAIN_COSINE_SIM, TRAIN_CONTEXT
    global TEST_SRC_ENTITIES, TEST_SRC_CONFIDENCES, TEST_EXT_ENTITIES, TEST_EXT_CONFIDENCES, TEST_COSINE_SIM, TEST_CONTEXT
    global evalMode
    global CORRECT, GOLD, PRED, EVALCONF, EVALCONF2
    global QUERY, ACTION, CHANGES
    global trained_model
    global CONTEXT_TYPE
    global STAT_POSITIVE, STAT_NEGATIVE
    global trainingMentionKeys, testingMentionKeys
    global trainingMentionMap, testingMentionMap
    global testingGoldPLOKeys
    global testingPredPLOKeys
    global evalRoundCnt
    global detailOutFile
    global label_trans_map
    global DEBUG

    evalRoundCnt = 0
    DEBUG = True

    print args

    #trained_model = pickle.load( open(args.modelFile, "rb" ) )

    #load cached entities (speed up)
    #train_articles, train_titles, train_goldTypes, train_downloaded_articles, TRAIN_EXT_ENTITIES, TRAIN_EXT_CONFIDENCES, TRAIN_COSINE_SIM, CONTEXT1, CONTEXT2 = pickle.load(open(args.trainEntities, "rb"))   # lutm: pickle.load：反序列化，从文件解析为对象。open是打开文件，第一个参数（name）是文件名，第二个参数"rb"是readonly & byte。
    print "loading " + resDir + "train-q.keys.json"
    trainingMentionKeys  = fileUtil.readToObject(resDir + "train-q.keys.json")
    print "trainingMentionKeys="+str(len(trainingMentionKeys))

    print "loading " + resDir + "test.keys.json"
    testingMentionKeys   = fileUtil.readToObject(resDir + "test.keys.json")
    print "testingMentionKeys="+str(len(testingMentionKeys))

    #boundaryToClassifierLabelNConfDict     = fileUtil.readToDict(resDir+"Stanford+Fudan.cvs", ['goldType', 'classifierLabel', 'conf'])
    #boundaryToClassifierLabelNConfDictWEKs = []
    #boundaryToClassifierLabelNConfDictWEKs.append(fileUtil.readToDict(resDir+"UrlPart[0,0].cvs", ['goldType', 'classifierLabel', 'conf']))
    ##boundaryToClassifierLabelNConfDictWEKs.append(fileUtil.readToDict(resDir+"UrlPart[-1,1].cvs", ['goldType', 'classifierLabel', 'conf']))
    #boundaryToClassifierLabelNConfDictWEKs.append(fileUtil.readToDict(resDir+"UrlPart[-2,2].cvs", ['goldType', 'classifierLabel', 'conf']))

    print "loading " + resDir + "train-q.map.json"
    trainingMentionMap     = fileUtil.readToObject(resDir + "train-q.map.json")
    print "loading " + resDir + "test.map.json"
    testingMentionMap     = fileUtil.readToObject(resDir + "test.map.json")

    # i=0
    # for testingMentionKey in testingMentionKeys:
    #     i+=1
    #     testingMention = testingMentionMap[testingMentionKey]
    #     goldType = testingMention["goldType"]
    #     testType = testingMention["neType"]
    #     correct = "T" if goldType==testType else "F"
    #     print format(i, '>5')+"\t"+goldType+"\t"+testType+"\t"+correct

    print "loading " + resDir + "test.gold.json"
    # 从mention表读出，天然没有O
    testingGoldPLOKeys = fileUtil.readToObject(resDir + "test.gold.json")
    print "testingGoldPLOKeys="+str(len(testingGoldPLOKeys))

    # numOfExtArticlePerQuery = 1

    train_articles = []
    train_titles = []
    train_goldTypes = []
    train_downloaded_articles = []
    TRAIN_SRC_ENTITIES    = []
    TRAIN_SRC_CONFIDENCES = []
    TRAIN_EXT_ENTITIES    = []
    TRAIN_EXT_CONFIDENCES = []
    TRAIN_COSINE_SIM  = []
    CONTEXT1          = []
    CONTEXT2          = []

    # testFile = open(args.evalOutFile + "[test].txt", 'w')
    for boundaryIdx in range(len(trainingMentionKeys)):
        boundaryKey = trainingMentionKeys[boundaryIdx]
        mention = trainingMentionMap[boundaryKey]

        # print "train: " + "\t " + mentionToStr(mention)
        # testFile.write("test: " + "\t " + mentionToStr(mention)+"\n")

        train_articles.append(mention)
        train_titles.append(boundaryKey)
        train_goldTypes.append([mention["goldType"]])

        classifierLabels = [mention["neType"]]
        classifierConfs = [mention["prob"]]
        TRAIN_SRC_ENTITIES.append(classifierLabels)
        TRAIN_SRC_CONFIDENCES.append(classifierConfs)

        train_downloaded_articles.append([])
        TRAIN_EXT_ENTITIES.append([])
        TRAIN_EXT_CONFIDENCES.append([])
        TRAIN_COSINE_SIM.append([])

        mapOfMentionsByQueries = mention["mapOfMentionsByQueries"]
        
        for queryIdx in range(NUM_QUERY_TYPES):
            queryType = constants.mapOfIntToQueryName[queryIdx]
            mentionsByQuery = mapOfMentionsByQueries[queryType]
            train_downloaded_articles[boundaryIdx].append([])
            TRAIN_EXT_ENTITIES[boundaryIdx].append([])
            TRAIN_EXT_CONFIDENCES[boundaryIdx].append([])
            TRAIN_COSINE_SIM[boundaryIdx].append([])

            for dldIdx in range(len(mentionsByQuery)):
                mentionByQuery = mentionsByQuery[dldIdx]
                train_downloaded_articles[boundaryIdx][queryIdx].append(mentionByQuery)
                TRAIN_EXT_ENTITIES[boundaryIdx][queryIdx].append(mentionByQuery['neType'])
                TRAIN_EXT_CONFIDENCES[boundaryIdx][queryIdx].append(mentionByQuery['prob'])
                # TRAIN_COSINE_SIM[boundaryIdx][queryIdx].append(1.)

    if args.contextType == 1:
        TRAIN_CONTEXT = CONTEXT1
    else:
        TRAIN_CONTEXT = CONTEXT2

    CONTEXT_TYPE = args.contextType

    
    #test_articles, test_titles, test_goldTypes, test_downloaded_articles, TEST_EXT_ENTITIES, TEST_EXT_CONFIDENCES, TEST_COSINE_SIM, CONTEXT1, CONTEXT2 = pickle.load(open(args.testEntities, "rb"))
    test_articles = []
    test_titles = []
    test_goldTypes = []
    test_downloaded_articles = []
    TEST_SRC_ENTITIES = []
    TEST_SRC_CONFIDENCES = []
    TEST_EXT_ENTITIES = []
    TEST_EXT_CONFIDENCES = []
    TEST_COSINE_SIM = []
    CONTEXT1 = []
    CONTEXT2 = []

    for boundaryIdx in range(len(testingMentionKeys)):
        boundaryKey = testingMentionKeys[boundaryIdx]
        mention = testingMentionMap[boundaryKey]
        # print "test: " + "\t " + mentionToStr(mention)
        # testFile.write("test: " + "\t " + mentionToStr(mention)+"\n")

        test_articles.append(mention)
        test_titles.append(boundaryKey)
        test_goldTypes.append([mention["goldType"]])

        classifierLabels = [mention["neType"]]
        classifierConfs = [mention["prob"]]
        TEST_SRC_ENTITIES.append(classifierLabels)
        TEST_SRC_CONFIDENCES.append(classifierConfs)

        test_downloaded_articles.append([])
        TEST_EXT_ENTITIES.append([])
        TEST_EXT_CONFIDENCES.append([])
        TEST_COSINE_SIM.append([])

        mapOfMentionsByQueries = mention["mapOfMentionsByQueries"]

        for queryIdx in range(NUM_QUERY_TYPES):
            queryType = constants.mapOfIntToQueryName[queryIdx]
            mentionsByQuery = mapOfMentionsByQueries[queryType]
            test_downloaded_articles[boundaryIdx].append([])
            TEST_EXT_ENTITIES[boundaryIdx].append([])
            TEST_EXT_CONFIDENCES[boundaryIdx].append([])
            TEST_COSINE_SIM[boundaryIdx].append([])

            for dldIdx in range(len(mentionsByQuery)):
                mentionByQuery = mentionsByQuery[dldIdx]
                test_downloaded_articles[boundaryIdx][queryIdx].append(mentionByQuery)
                TEST_EXT_ENTITIES[boundaryIdx][queryIdx].append(mentionByQuery['neType'])
                TEST_EXT_CONFIDENCES[boundaryIdx][queryIdx].append(mentionByQuery['prob'])
                # TEST_COSINE_SIM[boundaryIdx][queryIdx].append(1.)

    if args.contextType == 1:
        TEST_CONTEXT = CONTEXT1
    else:
        TEST_CONTEXT = CONTEXT2

    print "train_articles="+str(len(train_articles))
    print "test_articles="+str(len(test_articles))


    #starting assignments
    if not args.baselineEval and not args.thresholdEval and not args.confEval:      # lutm: 都是false，所以进第一个分支
        SRC_ENTITIES = TRAIN_SRC_ENTITIES
        SRC_CONFIDENCES = TRAIN_SRC_CONFIDENCES
        EXT_ENTITIES = TRAIN_EXT_ENTITIES
        EXT_CONFIDENCES = TRAIN_EXT_CONFIDENCES
        COSINE_SIM = TRAIN_COSINE_SIM
        CONTEXT = TRAIN_CONTEXT
        articles, titles, goldTypes, downloaded_articles = train_articles, train_titles, train_goldTypes, train_downloaded_articles
    else:
        SRC_ENTITIES = TEST_SRC_ENTITIES
        SRC_CONFIDENCES = TEST_SRC_CONFIDENCES
        EXT_ENTITIES = TEST_EXT_ENTITIES
        EXT_CONFIDENCES = TEST_EXT_CONFIDENCES
        COSINE_SIM = TEST_COSINE_SIM
        CONTEXT = TEST_CONTEXT
        articles, titles, goldTypes, downloaded_articles = test_articles, test_titles, test_goldTypes, test_downloaded_articles

    if args.baselineEval:        
        raise RuntimeError("unexpected branch")
        # args.baselineEval=false, 暂时不管这个方法
        # baselineEval(articles, goldTypes, args)
        # noinspection PyUnreachableCode
        return
    elif args.thresholdEval:
        raise RuntimeError("unexpected branch")
        # args.thresholdEval=false, 暂时不管这个方法
        # thresholdEval(articles, downloaded_articles, goldTypes, args)
        # noinspection PyUnreachableCode
        return
    elif args.confEval:
        raise RuntimeError("unexpected branch")
        # args.confEval=false, 暂时不管这个方法
        # confEval(articles, downloaded_articles, goldTypes, args)
        # noinspection PyUnreachableCode
        return
    elif args.classifierEval:
        raise RuntimeError("unexpected branch")
        # noinspection PyUnreachableCode
        print args.trainEntities
        print args.testEntities


        m = "TEST"
        split_index = 292
        if  m == "DEV":
            CLS_TEST_ENTITIES = splitDict(TRAIN_EXT_ENTITIES, split_index, len(TRAIN_EXT_ENTITIES))
            CLS_TEST_CONFIDENCES = splitDict(TRAIN_EXT_CONFIDENCES, split_index, len(TRAIN_EXT_ENTITIES))
            CLS_TEST_COSINE_SIM = splitDict(TRAIN_COSINE_SIM, split_index, len(TRAIN_EXT_ENTITIES))
            CLS_TEST_CONTEXT = splitDict(TRAIN_CONTEXT, split_index, len(TRAIN_EXT_ENTITIES))
            CLS_test_goldTypes = splitDict(train_goldTypes, split_index, len(TRAIN_EXT_ENTITIES))
                     
            CLS_TRAIN_ENTITIES = splitDict(TRAIN_EXT_ENTITIES, 0, split_index)
            CLS_TRAIN_CONFIDENCES = splitDict(TRAIN_EXT_CONFIDENCES, 0, split_index)
            CLS_TRAIN_COSINE_SIM = splitDict(TRAIN_COSINE_SIM, 0, split_index )
            CLS_TRAIN_CONTEXT= splitDict(TRAIN_CONTEXT, 0, split_index )

            CLS_train_goldTypes   = splitDict(train_goldTypes, 0, split_index )
        elif m == "TEST":
            CLS_TRAIN_ENTITIES =TRAIN_EXT_ENTITIES 
            CLS_TRAIN_CONFIDENCES =TRAIN_EXT_CONFIDENCES 
            CLS_TRAIN_COSINE_SIM =TRAIN_COSINE_SIM 
            CLS_TRAIN_CONTEXT =TRAIN_CONTEXT 
                     
            CLS_TEST_ENTITIES = TEST_EXT_ENTITIES 
            CLS_TEST_CONFIDENCES = TEST_EXT_CONFIDENCES 
            CLS_TEST_COSINE_SIM = TEST_COSINE_SIM 
            CLS_TEST_CONTEXT=     TEST_CONTEXT

       
            CLS_train_goldTypes = train_goldTypes
            CLS_test_goldTypes   = test_goldTypes

        baseline = Classifier(CLS_TRAIN_ENTITIES, CLS_TRAIN_CONFIDENCES, CLS_TRAIN_COSINE_SIM, CLS_TRAIN_CONTEXT,\
                 CLS_TEST_ENTITIES, CLS_TEST_CONFIDENCES, CLS_TEST_COSINE_SIM, CLS_TEST_CONTEXT)
        
        baseline.trainAndEval(CLS_train_goldTypes, CLS_test_goldTypes, args.entity, COUNT_ZERO)
        return

    indx = 0
    articleNum = 0
    savedArticleNum = 0

    outFile = open(outDir+"run.out", 'w', 0) #unbuffered
    outFile.write("dataset="+constants.dataset+"\n")
    outFile.write("classifierModel="+constants.classifierModel+"\n")
    outFile.write("outDir="+outDir+"\n")
    outFile.write(str(args)+"\n")

    outFile2 = open(outDir+"run.out"+'.2', 'w', 0) #for analysis
    outFile2.write(str(args)+"\n")

    detailOutFile = None
    # if args.evalOutFile != '':
    #    evalOutFile = open(args.evalOutFile, 'w')

    # pdb.set_trace()

    #server setup
    # port = args.port
    port = constants.port
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    print "Started server on port", port

    #for analysis
    stepCnt = 0

    # server loop
    while True:                                     # lutm: 这个设计很有意思哦，可以参考；本地网络，应该速度很快吧
        #  Wait for next request from client
        message = socket.recv()                     # lutm: 接收到的情况包括：newGame, evalStart, evalEnd, step
        # print "Received request: ", message

        if message == "newGame":
            # srcArticleIdx = articleNum % 10 #for test
            indx = articleNum % len(articles)   # articles可能是train_articles，也可能是test_articles
            #if DEBUG: print "INDX:", indx
            articleNum += 1 # 一直加1，没有取余
            originalArticle = articles[indx] #since article has words and tags

            if evalMode and DEBUG:
                detailOutFile.write("\n")
                detailOutFile.write("***************************************************************\n")
                detailOutFile.write("newGame: " + format(indx + 1, ">4") + "\t" + format(articleNum, ">5") + "\t\t " + mentionToStr(originalArticle) + "\n")
                detailOutFile.write("***************************************************************\n")

            #IMP: make sure downloaded_articles is of form <srcArticleIdx, listNum>
            # extArticles = [[q.split(' ')[:WORD_LIMIT] for q in sublist] for sublist in downloaded_articles[srcArticleIdx]]
            extArticles = downloaded_articles[indx]
            goldEntities = goldTypes[indx]
            env = Environment(originalArticle, extArticles, goldEntities, indx, args, evalMode)
            newstate, reward, terminal = env.state, 0, 'false'

        elif message == "evalStart":
            label_trans_map = collections.defaultdict(lambda: 0.)
            evalRoundCnt += 1
            if evalRoundCnt%10==1:
                DEBUG = True
            else:
                DEBUG = True

            if args.evalOutFile != '':
                detailOutFile = open(detailDir
                                     + format(evalRoundCnt, "03d")
                                     +".txt", 'w')

            testingPredPLOKeys = []
            CORRECT = collections.defaultdict(lambda:0.)  # lutm:? 没有输入，输出是0.。这是啥意思呢？
            GOLD = collections.defaultdict(lambda:0.)
            PRED = collections.defaultdict(lambda:0.)
            QUERY = collections.defaultdict(lambda:0.)
            ACTION = collections.defaultdict(lambda:0.)
            CHANGES = 0
            evalMode = True
            savedArticleNum = articleNum  # 这是一个临时的缓存
            articleNum = 0                # 开始评估，清零
            stepCnt = 0
            STAT_POSITIVE, STAT_NEGATIVE = [0 for i in range(NUM_SLOTS)], [0 for i in range(NUM_SLOTS)]

            SRC_ENTITIES = TEST_SRC_ENTITIES            # lutm: 把这些关键变量切换为测试数据
            EXT_ENTITIES = TEST_EXT_ENTITIES            
            SRC_CONFIDENCES = TEST_SRC_CONFIDENCES
            EXT_CONFIDENCES = TEST_EXT_CONFIDENCES
            COSINE_SIM = TEST_COSINE_SIM
            CONTEXT = TEST_CONTEXT
            articles, titles, goldTypes, downloaded_articles = test_articles, test_titles, test_goldTypes, test_downloaded_articles

            # print "##### Evaluation Started ######"

        elif message == "evalEnd":
            temp = format(evalRoundCnt+1, ">4")+"\t"+format(articleNum, ">4")+" ------------\nEvaluation Stats: (Accuracy, Precision, Recall, F1):"
            print temp
            outFile.write(temp+"\n")
            numOfCorrectPLOs = len(set(testingPredPLOKeys).intersection(set(testingGoldPLOKeys)))
            numOfPredPLOs = len(testingPredPLOKeys)
            numOfGoldPLOs = len(testingGoldPLOKeys)
            for tag in int2slots:
                accuracy = CORRECT[tag]/PRED[tag]
                prec = (numOfCorrectPLOs*1.0) / numOfPredPLOs
                rec = (numOfCorrectPLOs*1.0) / numOfGoldPLOs
                f1 = (2*prec*rec)/(prec+rec)
                print tag, accuracy, prec, rec, f1, "########[CorrectPLOOs", CORRECT[tag], "TotalPLOOs", PRED[tag], GOLD[tag], "] CorrectPLOs", numOfCorrectPLOs, "PredPLOs", numOfPredPLOs, "GoldPLOs", len(testingGoldPLOKeys)
                outFile.write('\t'.join([str(tag), str(accuracy), str(prec), str(rec), str(f1)])+'\n')
            print "StepCnt (total, average):", stepCnt, float(stepCnt)/len(articles)
            outFile.write("StepCnt (total, average): " + str(stepCnt)+ ' ' + str(float(stepCnt)/len(articles)) + '\n')
            labelTrans = "" \
                  +   "T->T: " + format(label_trans_map["T->T"], ">4") \
                  + "\tF->F: " + format(label_trans_map["F->F"], ">4") \
                  + "\tT->F: " + format(label_trans_map["T->F"], ">4") \
                  + "\tF->T: " + format(label_trans_map["F->T"], ">4")
            print labelTrans
            outFile.write(labelTrans + '\n')

            qsum = sum(QUERY.values())
            asum = sum(ACTION.values())
            outFile2.write("------------\nQsum: " + str(qsum) +  " Asum: " +  str(asum)+'\n')
            for k, val in QUERY.items():
                outFile2.write("Query " + str(k) + ' ' + str(val/qsum)+'\n')
            for k, val in ACTION.items():
                outFile2.write("Action " + str(k) + ' ' + str(val/asum)+'\n')
            outFile2.write("CHANGES: "+str(CHANGES)+ ' ' + str(float(CHANGES)/len(articles))+"\n")
            outFile2.write("STAT_POSITIVE, STAT_NEGATIVE "+str(STAT_POSITIVE) + ', ' +str(STAT_NEGATIVE)+'\n')

            #for analysis
            # pdb.set_trace()

            evalMode = False            
            articleNum = savedArticleNum    # 评估结束，恢复评估前的值

            SRC_ENTITIES = TRAIN_SRC_ENTITIES            # lutm: 把这些关键变量切换为训练数据
            EXT_ENTITIES = TRAIN_EXT_ENTITIES            
            SRC_CONFIDENCES = TRAIN_SRC_CONFIDENCES
            EXT_CONFIDENCES = TRAIN_EXT_CONFIDENCES
            COSINE_SIM = TRAIN_COSINE_SIM
            CONTEXT = TRAIN_CONTEXT
            articles, titles, goldTypes, downloaded_articles = train_articles, train_titles, train_goldTypes, train_downloaded_articles
            # print "##### Evaluation Ended ######"


            if args.oracle:
                plot_hist(EVALCONF, "conf1")
                plot_hist(EVALCONF2, "conf2")

            #save the extracted entities
            if args.saveEntities:
                pickle.dump([TRAIN_EXT_ENTITIES, TRAIN_EXT_CONFIDENCES, TRAIN_COSINE_SIM], open("train2.entities", "wb"))
                pickle.dump([TEST_EXT_ENTITIES, TEST_EXT_CONFIDENCES, TEST_COSINE_SIM], open("test2.entities", "wb"))
                return

        else:
            # message is "step"
            action, query = [int(q) for q in message.split()]
            queryIdx = query - 1
            if evalMode and DEBUG:
                actionName = constants.mapOfIntToActionName[action]
                detailOutFile.write("\naction\t=" + format(action, ">3") + "\t" + actionName + "\n")
                queryName = constants.mapOfIntToQueryName[queryIdx]
                detailOutFile.write("query\t=" + format(queryIdx, ">3") + "\t" + queryName + "\n")

            if evalMode:
                ACTION[action] += 1   # lutm: 看样子是个计数器
                QUERY[query] += 1

            # if evalMode and DEBUG:
            #     print "State:"
            #     print newstate[:4]
            #     print newstate[4:8]
            #     print newstate[8:]
            #     print "Entities:", env.slotsFromPrevArticle
            #     print "Action:", action, query

            newstate, reward, terminal = env.step(action, query)

            terminal = 'true' if terminal else 'false'
            if evalMode and DEBUG:
                detailOutFile.write("terminal=" + str(terminal) + "\n")

            #remove reward unless terminal
            if args.delayedReward == 'True' and terminal == 'false':
                reward = 0

            # if evalMode and DEBUG and reward != 0:
            #     print "Reward:", reward
            #     pdb.set_trace()

        if message != "evalStart" and message != "evalEnd":
            #do article eval if terminal
            # evalMode=True，表示当前在evalStart和evalEnd之间，所以需要评估当前文章
            # articles是测试集中的所有文章，articleNum是已经评估的文章，这个意思是：评估一遍所有的文章
            # terminal是env.step(action, query)返回的，当新下载文章用完了，或者动作是stop，就是terminal=True
            if evalMode and articleNum <= len(articles) and terminal == 'true':
                if args.oracle:
                    raise RuntimeError("unexpected branch")
                    # env.oracleEvaluate(env.goldEntities, ENTITIES[env.srcArticleIdx], CONFIDENCES[env.srcArticleIdx])
                else:
                    # 一篇文章经过多次精化、查询结束了，才进入评估
                    env.evaluateArticle(env.slotsCurrentlyHeld.values(), env.goldEntities, args.shooterLenientEval, args.shooterLastName, detailOutFile)
                    predNeType = env.slotsCurrentlyHeld[0]
                    if predNeType != 'O':
                        testingPredPLOKeys.append(env.srcMention["boundaryKey"] + " " + predNeType)

                stepCnt += sum(env.arrayOfNumOfUsedMentionsByQuery)

                #stat sign
                vals = env.calculateStatSign(env.entitiesInSrcArticle, env.slotsCurrentlyHeld.values())
                for i, val in enumerate(vals):
                    if val > 0:
                        STAT_POSITIVE[i] += val
                    else:
                        STAT_NEGATIVE[i] -= val

                #for analysis
                for entityNum in [0,1,2,3]:
                    if ANALYSIS and evalMode and env.slotsCurrentlyHeld.values()[entityNum].lower() != env.entitiesInSrcArticle[entityNum].lower() and reward > 0:
                        CHANGES += 1
                        try:
                            print "ENTITY:", entityNum
                            print "Entities:", 'best', env.slotsCurrentlyHeld.values()[entityNum], 'orig', env.entitiesInSrcArticle[entityNum], 'gold', env.goldEntities[entityNum]
                            print ' '.join(originalArticle)
                            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                            print ' '.join(extArticles[env.bestIndex[0]][env.bestIndex[1]])
                            print "----------------------------"
                        except:
                            pass

            #send message (IMP: only for newGame or step messages)
            outMsg = 'state, reward, terminal = ' + str(newstate) + ',' + str(reward)+','+terminal  # lutm: teminal是三项拼接，去看agent如何解析，才能理解
            socket.send(outMsg.replace('[', '{').replace(']', '}'))
        else:
            socket.send("done")                         # lutm: 给agent发


def mentionToStr(originalArticle):
    # if not originalArticle["succText"]:
    #     print originalArticle["boundaryKey"]
    return originalArticle["boundaryKey"] \
           + "\tGold: " + originalArticle["goldType"] \
           + "\tPred: " + originalArticle["neType"] \
           + "\t" + format(originalArticle["prob"], "0.3") \
           + "\t" + originalArticle["prevText"].replace("\n", "[\\n]") + "" \
           + " [" + originalArticle["prevPosTag1"] + "]" \
           + " [" + originalArticle["startPosTag"] + "]" \
           + " " + originalArticle["name"] \
           + " [" + originalArticle["endPosTag"] + "]" \
           + " [" + originalArticle["succPosTag1"] + "]" \
           + " " + originalArticle["succText"].replace("\n", "[\\n]") + ""

           # + " [" + originalArticle["prevPosTag2"] + "]" \
           # + " [" + originalArticle["succPosTag2"] + "]" \
    #lutm: OK
if __name__ == '__main__':                              # lutm: 当运行这个文件时，这里是入口
    env = None                                          # lutm: None相当于Java的null
    newstate, reward, terminal = None, None, None       # lutm：这样赋值也不错

    argparser = argparse.ArgumentParser(sys.argv[0])    # lutm: 有一个文件叫argparse.pyi，此文件里有一个类叫ArgumentParser，第一个参数是prog。0程序文件绝对路径；1及以后，外部参数。
    argparser.add_argument("--port",
        type = int,
        default = 5050,
        help = "port for server")
    argparser.add_argument("--trainFile",
        type = str,
        help = "training File")
    argparser.add_argument("--testFile",
        type = str,
        default = "",
        help = "Testing File")
    argparser.add_argument("--outFile",
        type = str,
        help = "Output File")

    argparser.add_argument("--evalOutFile",
        default = "",
        type = str,
        help = "Output File for predictions")

    argparser.add_argument("--modelFile",
        type = str,
        help = "Model File")

    argparser.add_argument("--shooterLenientEval",
        type = bool,
        default = False,
        help = "Evaluate shooter leniently by counting any match as right")

    argparser.add_argument("--shooterLastName",
        type = bool,
        default = False,
        help = "Evaluate shooter using only last name")

    argparser.add_argument("--oracle",
        type = bool,
        default = False,
        help = "Evaluate using oracle")

    argparser.add_argument("--ignoreDuplicates",
        type = bool,
        default = False,
        help = "Ignore duplicate articles in downloaded ones.")

    argparser.add_argument("--baselineEval",
        type = bool,
        default = False,
        help = "Evaluate baseline performance")

    argparser.add_argument("--classifierEval",
        type = bool,
        default = False,
        help = "Evaluate performance using a simple maxent classifier")

    argparser.add_argument("--thresholdEval",
        type = bool,
        default = False,
        help = "Use tf-idf similarity threshold to select articles to extract from")

    argparser.add_argument("--threshold",
        type = float,
        default = 0.8,
        help = "threshold value for Aggregation baseline above")

    argparser.add_argument("--confEval",
        type = bool,
        default = False,
        help = "Evaluate with best conf ")

    argparser.add_argument("--rlbasicEval",
        type = bool,
        default = False,
        help = "Evaluate with RL agent that takes only reconciliation decisions.")

    argparser.add_argument("--rlqueryEval",
        type = bool,
        default = False,
        help = "Evaluate with RL agent that takes only query decisions.")

    argparser.add_argument("--shuffleArticles",
        type = bool,
        default = False,
        help = "Shuffle the order of new articles presented to agent")

    argparser.add_argument("--entity",
                           type = int,
                           default = NUM_SLOTS,
                           help = "Entity num. 4 means all.")

    argparser.add_argument("--aggregate",
        type = str,
        default = 'always',
        help = "Options: always, conf, majority")

    argparser.add_argument("--delayedReward",
        type = str,
        default = 'False',
        help = "delay reward to end")

    argparser.add_argument("--trainEntities",
        type = str,
        default = '',
        help = "Pickle file with extracted train entities")

    argparser.add_argument("--testEntities",
        type = str,
        default = '',
        help = "Pickle file with extracted test entities")

    argparser.add_argument("--numEntityLists",
        type = int,
        default = 1,
        help = "number of different query lists to consider")

    argparser.add_argument("--contextType",
        type = int,
        default = 1,
        help = "Type of context to consider (1 = counts, 2 = tfidf, 0 = none)")    


    argparser.add_argument("--saveEntities",
        type = bool,
        default = False,
        help = "save extracted entities to file")


    args = argparser.parse_args()                       # lutm: 根据以上若干argparser.add_argument的类型、默认值、说明，解析出参数列表，其中每个元素应该已经有了类型

    main(args)                                          # lutm: 调用main函数

#sample
#python server.py --port 7000 --trainEntities consolidated/train+context.5.p --testEntities consolidated/dev+test+context.5.p --outFile outputs/tmp2.out --modelFile trained_model2.p --entity 4 --aggregate always --shooterLenientEval True --delayedReward False --contextType 2


