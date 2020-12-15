# Class Group manages information of group in
# network and  relationship between server and group
import sys
from . import net_ap
from . import group_type
from . import parameters


class Group:
    groupNum = 0
    def __init__(self, groupTypeName, taskGenInfoDict):
        """ 
        groupTypeName should be a string, taskGenInfoDict should be a dictionary.
        """
        self._groupID = Group.groupNum
        Group.groupNum = Group.groupNum + 1
        self._serverList = []
        if type(groupTypeName) != str:
            sys.exit("Parameter 'groupTypeName' should be a string.")
        self._groupTypeName = groupTypeName
        self._taskGenInfoDict = taskGenInfoDict

    def addServer(self, server):
        if type(server) != net_ap.NetAP:
            sys.exit("Parameter 'server' should be a NetAP object.")
        else:
            self._serverList.append(server)
    
    def getTaskGenInfoDict(self):
        return dict(self._taskGenInfoDict)

    def addTaskGenInfo(self, taskTypeName, mean, variance):
        if mean <0 or variance <0:
            sys.exit("The value of mean or variance should not be negative.")
        else:
            self._taskGenInfoDict[taskTypeName] = [mean, variance]

    def deleteTaskGenInfo(self, taskTypeName):
        del self._taskGenInfoDict[taskTypeName]
    
    def getServerList(self):
        return list(self._serverList)
    
    def getTypeName(self):
        return self._groupTypeName

    def getID(self):
        return self._groupID
    
    def getKey(self):
        return 'group-' + self._groupTypeName + '-' + str(self._groupID)

    def __eq__(self, other):
        if type(other) == Group:
            if self._groupID == other.getID():
                return True
            else:
                False
        else:
            return False

def createAGroup(grouType=None):
    if grouType ==None:
        grouType = group_type.createAGroupType()
    gp = Group(grouType.getGroupTypeName(), grouType.getTaskGenInfoDict())
    for i in range(grouType.generateServerNum()):
        tmpServer = net_ap.NetAP(parameters.CODE_SERVER, grouType.generateServerRsc(), gp)
        gp.addServer(tmpServer)
    return gp

def testFunc():
    print("You have successfully inovked group.testFun!")

if __name__ == "__main__":
    gp = createAGroup()
    print("Group's type name is: %s." %(gp.getTypeName()))
    print("Gropu's taskGenInfoDict is: ", gp.getTaskGenInfoDict())
    for s in gp.getServerList():
        print("The server in this group has a key %s, its group type name is:%s" % (s.getKey(), s.getGroup().getTypeName()))

