# Class UserType
import numpy as np
import sys
from . import parameters
from . import group
from . import group_type
from . import user
from . import group_trans

class UserType:
    def __init__(self, typeName, groupTrans):
        self._typeName = typeName
        if type(groupTrans) == group_trans.GroupTrans:
            self._groupTrans = groupTrans
        else:
            sys.exit("The type of groupTrans should be GroupTrans.")
    
    def createUser(self, initGroup):
        serverList = initGroup.getServerList()
        return user.User(serverList[np.random.randint(0, len(serverList))],
        initGroup, self._typeName, self._groupTrans)

    def getUserTypeName(self):
        return self._typeName

if __name__ == "__main__":
    gtBusiness = group_type.createAGroupType(parameters.CODE_GROUP_TYPE_BUSINESS)
    gtCompany = group_type.createAGroupType(parameters.CODE_GROUP_TYPE_COMPANY)
    gtCommunity = group_type.createAGroupType(parameters.CODE_GROUP_TYPE_COMMMUNITY)
    groupList = []
    for i in range(10):
        groupList.append(group.createAGroup(gtBusiness))
        groupList.append(group.createAGroup(gtCommunity))
        groupList.append(group.createAGroup(gtCompany))
    gTrans = group_trans.createAGroupTrans(groupList)
    ut = UserType(parameters.CODE_USER_TYPE_OTAKU, gTrans)
    userList = []
    for i in range(200):
        userList.append(ut.createUser(groupList[np.random.randint(0, len(groupList))]))

    for u in userList:
        print("Before trans, User %s, group:%s, server:%s, server_group:%s user_type:%s" % (u.getKey(), \
            u.getCurrentGroup().getKey(), u.getCurrentServer().getKey(), u.getCurrentServer().getGroup().getKey(), u.getTypeName()))
        u.move2Group(time=0)
        u.move2Server()
        print("After- trans, User %s, group:%s, server:%s, server_group:%s user_type:%s" % (u.getKey(), \
            u.getCurrentGroup().getKey(), u.getCurrentServer().getKey(), u.getCurrentServer().getGroup().getKey(), u.getTypeName()))
