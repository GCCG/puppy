# Class GroupType
import numpy as np
from . import  parameters
from . import task_type

class GroupType:
    def __init__(self, defaultServerNum, typeName,):
        self._defaultServerNum = defaultServerNum
        self._typeName = typeName
        self._bandwidthList = []
        self._comCapacityList = []
        self._taskGenInfoDict = {}
        self._lengthList = []

    def generateLinkBan(self):
        # Randomly select a value from self._bandwidthList
        return self._bandwidthList[np.random.randint(0, len(self._bandwidthList))]

    def generateLinkLen(self):
        return self._lengthList[np.random.randint(0, len(self._lengthList))]

    def generateServerRsc(self):
        # Randomly select a value from self._comCapacityList
        return self._comCapacityList[np.random.randint(0, len(self._comCapacityList))]

    def generateServerNum(self):
        return self._defaultServerNum #+ np.random.randint(0, 5)

    def expandLengthList(self, len):
        if type(len) != int:
            Exception("Wrong type of len, we need an integer value.")
        elif len <=0:
            Exception("The value of len should be positive.")
        else:
            self._lengthList.append(len)

    def expandBandwidthList(self, ban):
        if type(ban) != int:
            Exception("Wrong type of ban, we need an integer value.")
        elif ban <=0:
            Exception("The value of ban should be positive.")
        else:
            self._bandwidthList.append(ban)

    def expandComCapacityList(self, rsc):
        if type(rsc) != int:
            Exception("Wrong type of rsc, we need an integer value.")
        elif rsc<=0:
            Exception("The value of rsc should be positive.")
        else:
            self._comCapacityList.append(rsc)

    def addTaskGenInfo(self, taskTypeName, mean, variance):
        if mean <0 or variance <0:
            Exception("The value of mean or variance should not be negative.")
        else:
            self._taskGenInfoDict[taskTypeName] = [mean, variance]

    def getTaskGenInfo(self, taskTypeName):
        return self._taskGenInfoDict[taskTypeName]

    def getTaskGenInfoDict(self):
        # Return one copy of self._taskGenInfoDict
        return dict(self._taskGenInfoDict)

    def getGroupTypeName(self):
        return self._typeName

    def createGroup(self, serverNum):
        pass

def createAGroupType(groupTypeName=None):
    if groupTypeName == None:
        groupTypeList = [parameters.CODE_GROUP_TYPE_BUSINESS, \
        parameters.CODE_GROUP_TYPE_COMMMUNITY, parameters.CODE_GROUP_TYPE_COMPANY]
        groupTypeName = groupTypeList[np.random.randint(0, len(groupTypeList))]
    gt = GroupType(4, groupTypeName)

    gt.addTaskGenInfo(parameters.CODE_TASK_TYPE_VA, 3, 4)
    gt.addTaskGenInfo(parameters.CODE_TASK_TYPE_IoT, 5, 2)
    gt.addTaskGenInfo(parameters.CODE_TASK_TYPE_VR, 4, 3)

    gt.expandBandwidthList(10)
    gt.expandBandwidthList(5)
    gt.expandBandwidthList(15)

    gt.expandComCapacityList(5)
    gt.expandComCapacityList(3)
    gt.expandComCapacityList(1)

    gt.expandLengthList(7)
    gt.expandLengthList(9)
    gt.expandLengthList(8)
    return gt
    

if __name__ == "__main__":
    gt  = createAGroupType()
    nameList = [parameters.CODE_TASK_TYPE_IoT, parameters.CODE_TASK_TYPE_VA, parameters.CODE_TASK_TYPE_VR]
    for name in nameList:
        print("Task type %s generation info is: %s" % (name, gt.getTaskGenInfo(name)))

    print("Task generation information dictionary:\n", gt.getTaskGenInfoDict())

    
    for i in range(5):
        print("Random server num is: %d" % gt.generateServerNum())
        print("Random server comCapacity is: %d" % gt.generateServerRsc())
        print("Random link bandwidth is: %d" % gt.generateLinkBan())
        print("Random link length is: %d." % gt.generateLinkLen())
