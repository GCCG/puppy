# Class GroupTrans
# This class is responsible for transferring between groups
import numpy as np
import sys
import os
from . import  parameters
from . import group_type
from . import group

class GroupTrans:
    def __init__(self, groupList, phaseNum):
        self._groupNumDict = {}
        self._type2GroupsDict = {}
        self._groupList = groupList
        self._groupTransDicts = []
        groupTypes = []
        for g in groupList:
            if g.getTypeName() not in groupTypes:
                groupTypes.append(g.getTypeName())
        for i in range(phaseNum):
            tmpDict = {}
            for k in groupTypes:
                tmpN2N = {}
                for e in groupTypes:
                    tmpN2N[e] = 0
                    if k==e:
                        tmpN2N[e] = 1
                tmpDict[k] = tmpN2N
            self._groupTransDicts.append(tmpDict)

    def randGroup(self, currentGroup, phase):
        transDict = self._groupTransDicts[phase][currentGroup.getTypeName()]
        tmpRand = np.random.random()
        prob = 0.0
        for typeK in list(transDict.keys()):
            if prob <= tmpRand < prob+transDict[typeK]:
                tmpGroups = self._type2GroupsDict[typeK]
                return tmpGroups[np.random.randint(0, len(tmpGroups))]
            else:
                prob = prob + transDict[typeK]
        return currentGroup

    def setGroupNum(self, groupTypeName, groupNum):
        if groupNum > 0:
            tmpGroups = []
            tmpList = []
            for g in self._groupList:
                if g.getTypeName() == groupTypeName:
                    tmpList.append(g)
            print("In " + os.path.split(__file__)[-1] + ", Groups of type %s:" % groupTypeName)
            for g in tmpList:
                print(g.getKey())
            if groupNum > len(tmpList):
                print("groupNum is bigger than number of given type of group")
            groupNum = min([groupNum, len(tmpList)])
            tmpIndices = np.random.randint(0, len(tmpList), groupNum)
            for index  in tmpIndices:
                tmpGroups.append(tmpList[index])
            self._groupNumDict[groupTypeName] = groupNum
            self._type2GroupsDict[groupTypeName] = tmpGroups
        else:
            sys.exit("We need groupNum to be positive.")

    def resetGroupList(self, groupList):
        if len(groupList) < len(self._groupList):
            print("The given groupList is smaller than current one.")
        self._groupList = groupList
        for typeK in list(self._type2GroupsDict.keys):
            self.setGroupNum(typeK, self._groupNumDict[typeK])

    def setGroupTransProb(self, phase, currentTypeName, probDict):
        if sum(list(probDict.values())) != 1:
            sys.exit("The sum of probList should be 1.")
        for k in list(probDict.keys()):
            self._groupTransDicts[phase][currentTypeName][k] = probDict[k]
        
    def getGroupTypes(self):
        return list(self._type2GroupsDict.keys())

    def replaceGroup(self, group):
        groupTypeName = group.getTypeName()
        flag = False
        # If group is already in self._type2GroupsDict, then finish.
        for i in range(len(self._type2GroupsDict[groupTypeName])):
            if group == self._type2GroupsDict[groupTypeName][i]:
                return
        # if not, then randomly select  a group in self._type2GroupsDict[groupTypeName] to replace.
        self._type2GroupsDict[groupTypeName][np.random.randint(0, len(self._type2GroupsDict[groupTypeName]))] = group

    def getGroups(self, groupTypeName):
        return list(self._type2GroupsDict[groupTypeName])

    def addPhase(self):
        pass


def createAGroupTrans(groupList):
    phaseNum = 3
    gTrans = GroupTrans(groupList, phaseNum)

    gTrans.setGroupNum(parameters.CODE_GROUP_TYPE_BUSINESS, 3)
    gTrans.setGroupNum(parameters.CODE_GROUP_TYPE_COMMMUNITY, 1)
    gTrans.setGroupNum(parameters.CODE_GROUP_TYPE_COMPANY, 2)

    grouTypeNameList = gTrans.getGroupTypes()
    bussinessDict = {}
    bussinessDict[parameters.CODE_GROUP_TYPE_BUSINESS] = 0.8
    bussinessDict[parameters.CODE_GROUP_TYPE_COMMMUNITY] = 0.1
    bussinessDict[parameters.CODE_GROUP_TYPE_COMPANY] = 0.1
    gTrans.setGroupTransProb(0, parameters.CODE_GROUP_TYPE_BUSINESS, bussinessDict)

    communityDict = {}
    communityDict[parameters.CODE_GROUP_TYPE_COMMMUNITY] = 0.15
    communityDict[parameters.CODE_GROUP_TYPE_COMPANY] = 0.05
    communityDict[parameters.CODE_GROUP_TYPE_BUSINESS] = 0.8
    gTrans.setGroupTransProb(0, parameters.CODE_GROUP_TYPE_COMMMUNITY, communityDict)

    companyDict = {}
    companyDict[parameters.CODE_GROUP_TYPE_COMPANY] = 0.05
    companyDict[parameters.CODE_GROUP_TYPE_BUSINESS] = 0.9
    companyDict[parameters.CODE_GROUP_TYPE_COMMMUNITY] = 0.05
    gTrans.setGroupTransProb(0, parameters.CODE_GROUP_TYPE_COMPANY, companyDict)

    return gTrans


if __name__ == "__main__":
    groupList = []
    gtBusiness = group_type.GroupType(4, parameters.CODE_GROUP_TYPE_BUSINESS)
    gtCompany = group_type.GroupType(6, parameters.CODE_GROUP_TYPE_COMPANY)
    gtCommunity = group_type.GroupType(10, parameters.CODE_GROUP_TYPE_COMMMUNITY)

    numDict = {parameters.CODE_GROUP_TYPE_BUSINESS:5, parameters.CODE_GROUP_TYPE_COMMMUNITY:7, parameters.CODE_GROUP_TYPE_COMPANY:8}
    for key in list(numDict.keys()):
        for i in range(numDict[key]):
            groupList.append(group.Group(key, {}))
    
    gTrans = createAGroupTrans(groupList)

    print("Randomly select groups:")

    for i in range(10):
        initGroup = groupList[np.random.randint(0, len(groupList))]
        print("init group is: "+ initGroup.getKey())
        for j in range(20):
            initGroup = gTrans.randGroup(initGroup, 0)
            print("Trans %d, end group is: %s" % (j, initGroup.getKey()))

    print("Before replace:")
    groups = gTrans.getGroups(groupList[10].getTypeName())
    for g in groups:
        print(g.getKey())
    print("Use %s to replace:" % (groupList[10].getKey()))
    gTrans.replaceGroup(groupList[10])
    groups = gTrans.getGroups(groupList[10].getTypeName())
    for g in groups:
        print(g.getKey())


                

