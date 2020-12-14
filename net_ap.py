# Class NetAP

import sys
from . import group
from . import parameters
from . import group_type

class NetAP:
    def __init__(self, apType, apID, rscAmount, serverID=-1, group=None):
        if apType != parameters.CODE_SERVER and apType != parameters.CODE_SWITCH:
            print(apType)
            sys.exit("---In "+__file__+"\nNetAP apType is not switch or server")
        self._apType = apType
        self._apID = apID
        self._rscAmount = rscAmount
        self._serverID = serverID
        self._group = group
    
    def isServer(self):
        if self._apType == parameters.CODE_SERVER:
            return True
        elif self._apType == parameters.CODE_SWITCH:
            return False
        else:
            print("Something is wrong with your NetAP class")
            
    
    def getType(self):
        return self._apType

    def getRscAmount(self):
        return self._rscAmount

    def getID(self):
        return self._serverID

    def getGroup(self):
        return self._group

    def setGroup(self, APGroup):
        if type(APGroup) != group.Group:
            sys.exit("In netap.py setGroup, type of APGroup is \
            %s, not Group", type(APGroup))
        else:
            self._group = APGroup
    
    def getKey(self):
        return 'server-'+str(self._serverID)

def createANetAP(group, apID, serverID, groupType):
    return NetAP(parameters.CODE_SERVER, apID, groupType.generateServerRsc(), serverID, group)


def testFunc():
    print("You have successfully invoked netap.testFunc!")

if __name__=="__main__":
    # group.testFunc()
    gt = group_type.createAGroupType()
    gp = group.Group(gt.getGroupTypeName(), gt.getTaskGenInfoDict())
    server = createANetAP(gp, 1, 1, gt)
    print("Created server is of tye: %s" % (server.getType()))
    print("Created server has key: %s" % (server.getKey()))
    print("Created server has rsc amount: %d" % (server.getRscAmount()))