# Class TaskGenerator
import numpy as np
import copy
from . import net_ap
from . import net_link
from . import task
from . import user
from . import user_type
from . import parameters

from . import group
from . import group_trans
from . import group_type

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
        self._taskNum = self._taskNum + len(tmpList)
        return tmpList
    
    def userMove(self, time):
        for u in self._userList:
            # print("%s is moving, current_group:%s" % (u.getKey(), u.getCurrentGroup().getKey()))
            u.move2Group(time)
            u.move2Server()
            # print("%s moving finished, current_group:%s" % (u.getKey(), u.getCurrentGroup().getKey()))

    def generateUsers(self, initGroup, userNum, userTypeName):
        for i in range(userNum):
            self._generateUser(initGroup, userTypeName)

    def addUserType(self, typeName, groupTrans):
        self._userTypeDict[typeName] = user_type.UserType(typeName, groupTrans)
    
    def getUsers(self):
        return list(self._userList)
         
    def _generateUser(self, initGroup, userTypeName):
        tmpUserType = self._userTypeDict[userTypeName]
        tmpUser = tmpUserType.createUser(initGroup)
        self._userList.append(tmpUser)
        return tmpUser


if __name__ == "__main__":
    tg = TaskGenerator()
    gtBusiness = group_type.createAGroupType(parameters.CODE_GROUP_TYPE_BUSINESS)
    gtCompany = group_type.createAGroupType(parameters.CODE_GROUP_TYPE_COMPANY)
    gtCommunity = group_type.createAGroupType(parameters.CODE_GROUP_TYPE_COMMMUNITY)
    groupList = []
    for i in range(10):
        groupList.append(group.createAGroup(gtBusiness))
        groupList.append(group.createAGroup(gtCommunity))
        groupList.append(group.createAGroup(gtCompany))
    gTrans = group_trans.createAGroupTrans(groupList)
    # ut1 = user_type.UserType(parameters.CODE_USER_TYPE_OTAKU, gTrans)
    # ut2 = user_type.UserType(parameters.CODE_USER_TYPE_RICH_MAN, gTrans)
    # ut3 = user_type.UserType(parameters.CODE_USER_TYPE_SALARY_MAN, gTrans)

    tg.addUserType(parameters.CODE_USER_TYPE_OTAKU, gTrans)
    tg.addUserType(parameters.CODE_USER_TYPE_SALARY_MAN, gTrans)
    tg.addUserType(parameters.CODE_USER_TYPE_RICH_MAN, gTrans)
    for g in groupList:
        tg.generateUsers(g, 2, parameters.CODE_USER_TYPE_RICH_MAN)
        tg.generateUsers(g, 2, parameters.CODE_USER_TYPE_OTAKU)
        tg.generateUsers(g, 2, parameters.CODE_USER_TYPE_SALARY_MAN)
    before = copy.deepcopy(tg.getUsers())
    tg.userMove(time=0)
    after = tg.getUsers()
    for i in range(len(before)):
        print("Before trans, %s, current_group:%s , current_server:%s" % \
            (before[i].getKey(),before[i].getCurrentGroup().getKey(), before[i].getCurrentServer().getKey()))
        print("After- trans, %s, current_group:%s , current_server:%s" % \
            (after[i].getKey(),after[i].getCurrentGroup().getKey(), after[i].getCurrentServer().getKey()))


