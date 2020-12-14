# Class TaskGenerator
import numpy as np
from . import net_ap
from . import net_link
from . import task
from . import user
from . import user_type
from . import parameters

class TaskGenerator:
    def __init__(self):
        self._taskNum = 0
        self._periodLength = 0
        # Maps user type name to UserType object
        self._userTypeDict = {}
        # Maps task type name to TaskType object
        self._taskTypeDict = {}
        self._userList = []
        self._taskList = []

    def generateTasks(self, time):
        tmpList = []
        # For each user in self._userList
        for u in self._userList:
        # get group from user,
            userGroup = u.getCurrentGroup()
            # get task generation information from group,
            taskGenInfoDict = userGroup.getTaskGenInfoDict()
            # for each task type
            for typeK in list(taskGenInfoDict.keys()):
                taskGenNum = np.random.poisson(taskGenInfoDict[typeK][0])
                for i in range(taskGenNum):
                    tmpList.append(self._taskTypeDict[typeK].createTask(u.getCurrentServer, time))
        # use its task generation information to generate tasks
        self._taskList.extend(tmpList)
        return tmpList
    
    def userMove(self, time):
        for u in self._userList:
            u.move2Group(time)
            u.move2Server()

    def generateUsers(self, initGroup, userNum, userType):
        for i in range(userNum):
            self._generateUser(initGroup, userType)

    def addUserType(self, typeName, groupTrans):
        self._userTypeDict[typeName] = user_type.UserType(typeName, groupTrans)
         
    def _generateUser(self, initGroup, userType):
        tmpUserType = self._userTypeDict[userType]
        tmpUser = tmpUserType.createUser(initGroup)
        self._userList.append(tmpUser)
        return tmpUser
