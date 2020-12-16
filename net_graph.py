# Class NetGraph
import sys
from . import net_ap
from . import net_link
from . import net_path
from . import parameters
from . import group_type
from . import group

INF = 1000000 # Exact value of infinity.
class NetGraph:
    def __init__(self):
        self._serverNum = 0
        self._APNum = 0
        self._linkNum = 0
        self._APList = []
        self._linkMetrix = []
        self._pathMatrix = []

    # ID generation methods
    # Because we don't need to delete sth at the current stage,
    # so we can generate serverID incrementally.
    def _generateServerID(self):
        return self._serverNum + 1
    
    def _generateApID(self):
        return self._APNum + 1

    def _generateLinkID(self):
        return self._linkNum + 1

    def _floydShortestPath(self):
        pathMatrix = []
        pathLen = []
        # Initialize pathLen metrix
        for i in range(self._APNum):
            pathes = []
            rows = []
            for j in range(self._APNum):
                pathes.append(net_path.NetPath())
                if self._linkMetrix[i][j] == None:
                    rows.append(INF)
                elif type(self._linkMetrix[i][j]) == net_link.NetLink:
                    rows.append(self._linkMetrix[i][j].getLinkLength())
                    pathes[j].addLink(self._linkMetrix[i][j])
                else:
                    sys.exit("Wrong type of link, expect type NetLink,but get %s", 
                    type(self._linkMetrix[i][j]))
                if i==j:
                    rows[j] = 0
            pathMatrix.append(pathes)
            pathLen.append(rows)
        print("PathMatrix has been created")
        for c in range(self._APNum):
            for a in range(self._APNum):
                for b in range(self._APNum):
                    if pathLen[a][c]+pathLen[c][b] < pathLen[a][b]:
                        pathLen[a][b] = pathLen[a][c] + pathLen[c][b]
                        # print("---Concatenating: %d-%d-%d" % (a+1, c+1, b+1))
                        pathMatrix[a][b] = self.__concatenate(pathMatrix[a][c], pathMatrix[c][b])
        print("PathLenMatrix:\n", pathLen)
        self._pathMatrix = pathMatrix

    def getShortestPath(self, start, end):
        startIndex = self.__getAPIndex(start)
        endIndex = self.__getAPIndex(end)
        
        return self._pathMatrix[startIndex][endIndex]

    def _dijkstraShortestPath(self):
        pass

    def getServerList(self):
        serverList = []
        for ap in self._APList:
            if ap.isServer():
                serverList.append(ap)
        return serverList
    
    def getLinkList(self):
        linkList = []
        for i in range(self._APNum):
            for j in range(self._APNum):
                if self._linkMetrix[i][j] != None:
                    linkList.append(self._linkMetrix[i][j])
        return linkList

    def addSwitch(self):
        switch = net_ap.NetAP(parameters.CODE_SWITCH, 0,)
        # Add new switch into APList
        self._APList.append(switch)
        # Expand linkMetrix to accommodate links refers to created switch
        for e in self._linkMetrix:
            e.append(None)
        self._APNum = self._APNum + 1 # Update APNum
        tmpRow = []
        for i in range(self._APNum):
            tmpRow.append(None)
        self._linkMetrix.append(tmpRow)

        return switch

    def addLink(self, headAP, tailAP, ban, len, type):
        # Create a link
        tmpLink = net_link.NetLink(len, headAP, tailAP, ban, type)
        # Add it into linkMetrix
        # print("Index:", self.__getAPIndex(headAP)+1, '  ',self.__getAPIndex(tailAP)+1)
        self._linkMetrix[self.__getAPIndex(headAP)][self.__getAPIndex(tailAP)] = tmpLink
        return tmpLink

    def addServer(self, rscAmount):
        # Create a new server and add it into APlist
        tmpServer = net_ap.NetAP(parameters.CODE_SERVER, rscAmount)
        self._APList.append(tmpServer)
        # Expand linkMertix
        for e in self._linkMetrix:
            e.append(None)
        self._APNum = self._APNum + 1
        self._serverNum = self._serverNum + 1
        tmpRow = []
        for i in range(self._APNum):
            tmpRow.append(None)
        self._linkMetrix.append(tmpRow)

        return tmpServer

    def __concatenate(self, path1, path2):
        # print("Entered in concatenate.")
        tmpPath = net_path.NetPath()
        path1.reset()
        path2.reset()
        for i in range(path1.getLinkNum()):
            tmpLink = path1.nextLink()
            # print("link lenght is:", tmpLink.getLinkLength())
            tmpPath.addLink(tmpLink)
        for j in range(path2.getLinkNum()):
            tmpLink = path2.nextLink()
            # print("link lenght is:", tmpLink.getLinkLength())
            tmpPath.addLink(tmpLink)
        
        return tmpPath

    def __getAPIndex(self, ap):
        index = 0
        for i in range(self._APNum):
            if self._APList[i].getID() == ap.getID():
                return i
        sys.exit("No such AP in this net graph!")


class TreeNetGraph(NetGraph):
    def __init__(self):
        NetGraph.__init__(self,)
        self._layerNum = 0
        self._groupList = []
        self._groupTypeDict = {}

        # Initialize rootSwitch
        self._rootSwitch = self.addSwitch()
    
    def genGroup(self, fatherSwitch, len, ban, groupTypeName, serverNum=None):
        type = self._groupTypeDict[groupTypeName]
        # First, Create a Group.
        serverGroup = group.Group(groupTypeName, type.getTaskGenInfoDict())
        # Then create a switch and its backhaul links for this group.
        switch = self.addSwitch()
        switch.setGroup(serverGroup)
        self.addLink(switch, fatherSwitch, ban, len, parameters.CODE_BACKHAUL_LINK)
        self.addLink(fatherSwitch, switch, ban, len, parameters.CODE_BACKHAUL_LINK)
        if serverNum == None:
            serverNum = type.generateServerNum()
        # After that, create servers and their links iteratively.
        for i in range(serverNum):
            # Create a server
            # tmpServer = net_ap.NetAP(parameters.CODE_SERVER, self._generateApID(),
            # type.generateServerRsc(),self._generateServerID(), serverGroup)
            tmpServer = self.addServer(type.generateServerRsc())
            serverGroup.addServer(tmpServer)

            bandwidth = type.generateLinkBan()
            length = type.generateLinkLen()
            # backLink = net_link.NetLink(self._generateLinkID(), length,
            # switch, tmpServer, bandwidth, parameters.CODE_NORMAL_LINK)
            # forLink = net_link.NetLink(self._generateLinkID(), length,
            # tmpServer, switch, bandwidth, parameters.CODE_NORMAL_LINK)
            self.addLink(tmpServer, switch, bandwidth, length, parameters.CODE_NORMAL_LINK)
            self.addLink(switch, tmpServer, bandwidth, length, parameters.CODE_NORMAL_LINK)
        self._groupList.append(serverGroup)

    def addGroup(self, groupTypeName):
        type = self._groupTypeDict[groupTypeName]
        tmpGroup = group.Group(groupTypeName, type.getTaskGenInfoDict())
        self._groupList.append(tmpGroup)
        return tmpGroup
    
    def getRootSwitch(self):
         return self._rootSwitch
        
    def getGroupList(self):
        # Return a copy of groupList
        return list(self._groupList)

    def getGroupTypeNameList(self):
        tmp = list(self._groupTypeDict.keys())
        print("In TreeNetGraph, group types are:",tmp)
        return tmp
            
def createANetGraph():
    ng = NetGraph()

    v1 = ng.addServer(1)
    v2 = ng.addServer(2)
    v3 = ng.addServer(3)
    v4 = ng.addServer(4)
    v5 = ng.addServer(5)

    l12 = ng.addLink(v1, v2, 5, 2, parameters.CODE_NORMAL_LINK)
    l21 = ng.addLink(v2, v1, 7, 2, parameters.CODE_NORMAL_LINK)

    l13 = ng.addLink(v1, v3, 1, 8, parameters.CODE_NORMAL_LINK)
    l31 = ng.addLink(v3, v1, 1, 8, parameters.CODE_NORMAL_LINK)

    l24 = ng.addLink(v2, v4, 4, 3, parameters.CODE_NORMAL_LINK)
    l42 = ng.addLink(v4, v2, 8, 3, parameters.CODE_NORMAL_LINK)

    l34 = ng.addLink(v3, v4, 3, 1, parameters.CODE_NORMAL_LINK)
    l43 = ng.addLink(v4, v3, 9, 1, parameters.CODE_NORMAL_LINK)

    l35 = ng.addLink(v3, v5, 2, 6, parameters.CODE_NORMAL_LINK)
    l53 = ng.addLink(v5, v3, 10, 6, parameters.CODE_NORMAL_LINK)

    ng._floydShortestPath()
    return  ng

if __name__ == "__main__":
    ng = createANetGraph()
    tmpServerList = ng.getServerList()
    for s in tmpServerList:
        print("Server has key %s" % (s.getKey()))

    listLen = len(tmpServerList)
    for a in range(listLen):
        for b in range(listLen):
            tmpPath = ng.getShortestPath(tmpServerList[a], tmpServerList[b])
            print("---Path from v%d to v%d has linkLenList:" % (a+1, b+1))
            print(tmpPath.getLinkLengthList())
            print("Path from v%d to v%d has lenght: %d\n" % (a+1, b+1,tmpPath.getPathLength()))

        

        
        

