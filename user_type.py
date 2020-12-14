# Class UserType
import numpy as np
import sys
from . import parameters
from . import group
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
        initGroup, self._groupTrans)

    def getUserTypeName(self):
        return self._typeName

