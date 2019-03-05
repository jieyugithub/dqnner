# -- coding: utf-8 --

import collections
import json
import os

def returnNone():
    return None


def returnEmptyStr():
    return ''


def readToStr(path):
    with open(path, "r") as f:
        text = f.read()
        text = text.replace('\xef\xbb\xbf', '', 1)
    #print(text)
    return text


def readToObject(path):
    str = readToStr(path)
    obj = json.loads(str)
    return obj


def readToStrList(path):
    fileHandler = open(path, 'r')
    lines = fileHandler.readlines()
    i = 0
    for line in lines:
        if(i==0):
            line = line.replace('\xef\xbb\xbf', '', 1)
        line = line.strip()
        lines[i] = line
        i += 1
    return lines


def readToStrListList(path):
    lines = readToStrList(path)
    strListList = []
    i = 0
    for line in lines:
        strList = line.split('\t')
        strListList.append(strList)
        i += 1
    return strListList


# 第一列是key，后面是value
def readToDict(path, fieldNames):
    # dict = collections.defaultdict(returnNone, {})
    dict = {}
    strListList = readToStrListList(path)
    i=0
    for strList in strListList:
        key = strList[0]
        if key == '':
            continue
        if len(strList) == 2:
            value = strList[1].strip()
        else:
            value = strList[1:]
            if fieldNames:
                values = value
                value = {}
                for i in range(len(fieldNames)):
                    value[fieldNames[i]] = values[i]
        dict[key] = value
        i = i + 1
    return dict


def mkdir(path):
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)


def test():
    dict = readToDict("D:\\Research\\NER-RL\\Code\\lutm\\res\\test.dic")
    print 'ltm', dict['ltm']
    print 'ltm1', dict['ltm1']


#mkdir("/mnt/hgfs/VMWareShare/CONLLTest_R/LogisticRegression/1/2/3")