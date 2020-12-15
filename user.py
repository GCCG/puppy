# Class User
import numpy as np
from . import parameters
from . import task
from . import group_trans

class User:
    """ Each day is splited into three part,
    day, night, midnight, each part has its lenght. """
    dayTimeLen = 4
    nightTimeLen = 2
    midnightTimeLen = 2
    userNum = 0
    def __init__(self, initServer, initGroup, userTypeName, groupTrans):
        self._userID = User.userNum
        User.userNum = User.userNum + 1
        self._currentServer = initServer
        self._currentGroup = initGroup
        self._userTypeName = userTypeName
        self._groupTrans = groupTrans
        self._groupTrans.replaceGroup(self._currentGroup)
    
    def move2Group(self, time):
        period = User.dayTimeLen + User.nightTimeLen + User.midnightTimeLen
        if time % period < 4:
            self._currentGroup = self._groupTrans.randGroup(self._currentGroup, phase=0)
        elif time % period < 6:
            self._currentGroup = self._groupTrans.randGroup(self._currentGroup, phase=1)
        else:
            self._currentGroup = self._groupTrans.randGroup(self._currentGroup, phase=2)
        
        return self._currentGroup

    def move2Server(self):
        serverList = self._currentGroup.getServerList()
        self._currentServer = serverList[np.random.randint(0, len(serverList))]
        return self._currentServer

    def getCurrentGroup(self):
        return self._currentGroup

    def getCurrentServer(self):
        return self._currentServer

    def getKey(self):
        return "user-" + str(self._userID)
    
    def getTypeName(self):
        return self._userTypeName


    # def __randonmGroupTypeName(self, phase):
    #     dictValues = list(self._groupTypeTransDicts[phase][self._currentGroup.getTypeName()].values())
    #     dictKeys = list(self._groupTypeTransDicts[phase][self._currentGroup.getTypeName()].keys())
    #     tmp = np.random.random()
    #     prob = 0
    #     for i in range(len(dictValues)):
    #         if prob <= tmp < prob+dictValues[i]:
    #             return dictValues[i]
        
    # def __randomGroup(self, groupTypeName):
    #     if groupTypeName == parameters.CODE_GROUP_TYPE_COMMMUNITY:
    #         randGroup = self._homeGroupList[np.random.randint(0, len(self._homeGroupList))]
    #     elif groupTypeName == parameters.CODE_GROUP_TYPE_COMPANY:
    #         randGroup = self._workGroupList[np.random.randint(0, len(self._workGroupList))]
    #     else:
    #         randGroup = self._entGroupList[np.random.randint(0, len(self._entGroupList))]

    #     return randGroup

    # def generateTasks(self):
    #     taskList = []
    #     taskGenInfoDict = self._currentGroup.getTaskGenInfoDict()
    #     for typek in list(taskGenInfoDict.keys()):
    #         for i in range(np.random.poisson(taskGenInfoDict[typek][0])):
    #             tmpTask = task.Task(typek, self._currentServer, )


if __name__ == "__main__":
    pass





        