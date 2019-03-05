# -- coding: utf-8 --

import lutm.fileUtil

class PseudoClassifier:
    def __init__(self, resDir, name):
        self.name = name
        self.dict = lutm.fileUtil.readToDict(resDir+name + ".cvs")


    def classify(self, boundary):
        return self.dict[boundary]



