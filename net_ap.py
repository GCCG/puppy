# Class NetAP

import sys
from . import group
from . import parameters
from . import group_type

class NetAP:
    generatedApNUM = 0
    generatedServerNUM = 0
    generatedSwitchNum = 0
    def __init__(self, apType, rscAmount, group=None):
        if apType != parameters.CODE_SERVER and apType != parameters.CODE_SWITCH:
            print(apType)
            sys.exit("---In "+__file__+"\nNetAP apType is not switch or server")
        self._apType = apType
        self._apID = NetAP.generatedApNUM
        NetAP.generatedApNUM = 1 + NetAP.generatedApNUM
        self._rscAmount = rscAmount
        if apType==parameters.CODE_SERVER:
            self._serverID = NetAP.generatedServerNUM
            NetAP.generatedServerNUM = 1 + NetAP.generatedServerNUM
            self._switchID = -1
        else:
            self._serverID = -1
            self._switchID = NetAP.generatedSwitchNum
            NetAP.generatedSwitchNum = 1 + NetAP.generatedSwitchNum
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
        return self._apID

    def getGroup(self):
        return self._group

    def setGroup(self, APGroup):
        if type(APGroup) != group.Group:
            sys.exit("In netap.py setGroup, type of APGroup is \
            %s, not Group", type(APGroup))
        else:
            self._group = APGroup
    
    def getKey(self):
        if self._apType == parameters.CODE_SERVER:
            return self._apType + '-' + str(self._serverID)
        elif self._apType == parameters.CODE_SWITCH:
            return self._apType + "-" + str(self._switchID)

def createANetAP(group, groupType):
    return NetAP(parameters.CODE_SERVER, groupType.generateServerRsc(), group)


def testFunc():
    print("You have successfully invoked netap.testFunc!")

if __name__=="__main__":
    # group.testFunc()
    gt = group_type.createAGroupType()
    gp = group.Group(gt.getGroupTypeName(), gt.getTaskGenInfoDict())
    server = createANetAP(gp, gt)
    print("Created server is of type: %s" % (server.getType()))
    print("Created server has key: %s" % (server.getKey()))
    print("Created server has rsc amount: %d" % (server.getRscAmount()))