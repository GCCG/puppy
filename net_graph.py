# Class NetGraph
import sys
import numpy as np
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
    # Because we don't need to delete sth at the current bound,
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
        # print("PathMatrix has been created")
        for c in range(self._APNum):
            for a in range(self._APNum):
                for b in range(self._APNum):
                    if pathLen[a][c]+pathLen[c][b] < pathLen[a][b]:
                        pathLen[a][b] = pathLen[a][c] + pathLen[c][b]
                        # print("---Concatenating: %d-%d-%d" % (a+1, c+1, b+1))
                        pathMatrix[a][b] = self.__concatenate(pathMatrix[a][c], pathMatrix[c][b])
        # print("PathLenMatrix:\n", pathLen)
        self._pathMatrix = pathMatrix

    def getShortestPath(self, start, end):
        # print("In net_graph, path from %s to %s" % (start.getKey(), end.getKey()))
        startIndex = self.getAPIndex(start)
        endIndex = self.getAPIndex(end)
        # print("Their indices are: %d, %d." % (startIndex, endIndex))
        
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
        # print("Index:", self.getAPIndex(headAP)+1, '  ',self.getAPIndex(tailAP)+1)
        self._linkMetrix[self.getAPIndex(headAP)][self.getAPIndex(tailAP)] = tmpLink
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

    def getAPIndex(self, ap):
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
            tmpServer.setGroup(serverGroup)
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

        return serverGroup

    def addGroup(self, groupTypeName):
        type = self._groupTypeDict[groupTypeName]
        tmpGroup = group.Group(groupTypeName, type.getTaskGenInfoDict())
        self._groupList.append(tmpGroup)
        return tmpGroup
    
    def addGroupType(self, groupType):
        # if type(groupType) != group_type.GroupType:
        #     sys.exit("In net_graph, provided groupType should be a GroupType object. now it is of type %s." % type(groupType))
        self._groupTypeDict[groupType.getGroupTypeName()] = groupType
        print("In net_graph, Add new group Type %s." % (groupType.getGroupTypeName()))

    
    def getRootSwitch(self):
         return self._rootSwitch
        
    def getGroupList(self):
        # Return a copy of groupList
        return list(self._groupList)

    def getGroupTypeNameList(self):
        tmp = list(self._groupTypeDict.keys())
        print("In TreeNetGraph, group types are:",tmp)
        return tmp

    def print_info(self):
        print("-----Tree net graph information:")
        print("Servers:")
        s_list = self.getServerList()
        for i in range(len(s_list)):
            print("%s, resource amount is %f, load info is:" % (s_list[i].getKey(), s_list[i].getRscAmount()))
            info_dict = s_list[i].getGroup().getTaskGenInfoDict()
            # print(info_dict)
            for key in info_dict.keys():
                # print(key)
                print("---task type %s, mean:%f, variance:%f." % (key, info_dict[key][0], info_dict[key][1]))
        print("Links:")
        l_list = self.getLinkList()
        for j in range(len(l_list)):
            print("%s, from  %s to %s, bandwidth %f, length %f." % (l_list[j].getKey(), l_list[j].getHeadAP().getKey(),\
                l_list[j].getTailAP().getKey(), l_list[j].getBandwidth(), l_list[j].getLinkLength()))
        # print("Pathes for servers:")
        print("--------------------------------\n")

    def getLinksInGroup(self, group):
        s_list = group.getServerList()
        l_list = self.getLinkList()

        tmp_list = []
        for l in l_list:
            for s in s_list:
                if l.getHeadAP().getKey() ==  s.getKey() or l.getTailAP().getKey() == s.getKey():
                    tmp_list.append(l)
        return tmp_list

    def getServersInGroup(self, group):
        for g in self._groupList:
            # print(g.getKey(), group.getKey())
            if g.getKey() == group.getKey():
                return group.getServerList()
        print("No such group in this graph.")
        sys.exit()

    def getSwitchInGroup(self, group):
        for s in self._APList:
            try:
                # print(s.getGroup().getKey())
                # print(group.getKey())
                if s.isServer() != True and s.getGroup().getKey() == group.getKey():
                    return s
            except:
                continue
        print("%s has no switch" % group.getKey())
    
    def get_backhaul_links(self):
        l_list = self.getLinkList()
        tmp = []
        for l in l_list:
            if l.getHeadAP().isServer() or l.getTailAP().isServer():
                continue
            else:
                tmp.append(l)
        for l in tmp:
            print("%s, from %s, to %s" % (l.getKey(), l.getHeadAP().getKey(), l.getTailAP().getKey()))
        return tmp

    def reset_link_ban(self, start_ap, end_ap, ban):
        if ban > 0:
            self._linkMetrix[self.getAPIndex(start_ap)][self.getAPIndex(end_ap)].resetBandwidth(ban)
        else:
            raise ValueError("Bandwith should be a positive value.")

    # def check_load(self):
    #     # First check if computation rsc is enough

    # 一切包含着其他对象的引用的对象，都无法在其内部转化成json形式，因为
    # 这些引用信息是外部的，不在内部。
    # def jsonfy(self):
    #     info_dict = {}
    #     info_dict['server_num'] = self._serverNum
    #     info_dict['ap_num'] = self._APNum
    #     info_dict['link_num'] = self._linkNum
    #     # 先AP
    #     ap_info = []
    #     for ap in self._APList:
    #         ap_info.append(ap.jsonfy())
    #     # 再link
    #     link_info = []
    #     for i in range(self._APNum):
    #         for j in range(self._APNum):
    #             pass
    #     # 再group
        
    #     group_info = []
    #     for g in self._groupList:
    #         pass


            
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

def createATreeGraph():

    # Control system model here.
    np.random.seed(1)

    tng = TreeNetGraph()
    gtBusiness = group_type.GroupType(defaultServerNum=2, typeName=parameters.CODE_GROUP_TYPE_BUSINESS)
    gtBusiness.addTaskGenInfo(parameters.CODE_TASK_TYPE_IoT, 2, 5)
    gtBusiness.addTaskGenInfo(parameters.CODE_TASK_TYPE_VA, 2, 5)
    gtBusiness.addTaskGenInfo(parameters.CODE_TASK_TYPE_VR, 2, 8)
    gtBusiness.expandBandwidthList(15)
    gtBusiness.expandBandwidthList(15)
    gtBusiness.expandBandwidthList(15)
    gtBusiness.expandBandwidthList(15)
    gtBusiness.expandLengthList(12)
    gtBusiness.expandLengthList(15)
    gtBusiness.expandLengthList(3)
    gtBusiness.expandLengthList(20)
    gtBusiness.expandComCapacityList(4)
    gtBusiness.expandComCapacityList(8)
    gtBusiness.expandComCapacityList(2)

    gtCommunity = group_type.GroupType(defaultServerNum=2, typeName=parameters.CODE_GROUP_TYPE_COMMMUNITY)
    gtCommunity.addTaskGenInfo(parameters.CODE_TASK_TYPE_IoT, 2,5)
    gtCommunity.addTaskGenInfo(parameters.CODE_TASK_TYPE_VA, 2, 5)
    gtCommunity.addTaskGenInfo(parameters.CODE_TASK_TYPE_VR, 2, 8)
    gtCommunity.expandBandwidthList(15)
    gtCommunity.expandBandwidthList(15)
    gtCommunity.expandBandwidthList(15)
    gtCommunity.expandBandwidthList(15)
    gtCommunity.expandLengthList(5)
    gtCommunity.expandLengthList(10)
    gtCommunity.expandLengthList(12)
    gtCommunity.expandLengthList(20)
    gtCommunity.expandComCapacityList(5)
    gtCommunity.expandComCapacityList(1)
    gtCommunity.expandComCapacityList(2)

    gtCompany = group_type.GroupType(defaultServerNum=2, typeName=parameters.CODE_GROUP_TYPE_COMPANY)
    gtCompany.addTaskGenInfo(parameters.CODE_TASK_TYPE_IoT, 4,5)
    gtCompany.addTaskGenInfo(parameters.CODE_TASK_TYPE_VA, 2, 5)
    gtCompany.addTaskGenInfo(parameters.CODE_TASK_TYPE_VR, 2, 8)
    gtCompany.expandBandwidthList(15)
    gtCompany.expandBandwidthList(15)
    gtCompany.expandBandwidthList(15)
    gtCompany.expandBandwidthList(25)
    gtCompany.expandLengthList(12)
    gtCompany.expandLengthList(15)
    gtCompany.expandLengthList(3)
    gtCompany.expandLengthList(20)
    gtCompany.expandComCapacityList(5)
    gtCompany.expandComCapacityList(2)
    gtCompany.expandComCapacityList(2)

    tng.addGroupType(gtBusiness)
    tng.addGroupType(gtCommunity)
    tng.addGroupType(gtCompany)

    rootSwitch = tng.getRootSwitch()
    tng.genGroup(rootSwitch, 20, 20, parameters.CODE_GROUP_TYPE_BUSINESS)
    tng.genGroup(rootSwitch, 20, 20, parameters.CODE_GROUP_TYPE_COMMMUNITY)
    # tng.genGroup(rootSwitch, 35, 35, parameters.CODE_GROUP_TYPE_COMPANY)
    # tng.genGroup(rootSwitch, 20, 20, parameters.CODE_GROUP_TYPE_BUSINESS)
    # tng.genGroup(rootSwitch, 35, 35, parameters.CODE_GROUP_TYPE_COMPANY)
    tng._floydShortestPath()

    print()

    
    return tng



if __name__ == "__main__":
    print("------Create normal net graph:")
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
    
    print("------Create tree net graph")
    tng = createATreeGraph()
    tmpServerList = tng.getServerList()
    tmpGroupList = tng.getGroupList()
    for g in tmpGroupList:
        print("Group %s has servers:" %(g.getTypeName()))
        for s in g.getServerList():
            print(s.getKey())
    print("In tree net graph, servers are:")
    for s in tmpServerList:
        print(s.getKey())

    serverList_1 = tmpGroupList[0].getServerList()
    path01 = tng.getShortestPath(serverList_1[0], serverList_1[1])
    print("From %s to %s, path has links:" % (serverList_1[0].getKey(), serverList_1[1].getKey()))
    print(path01.getLinkLengthList())
    serverList_2 = tmpGroupList[2].getServerList()
    path02 = tng.getShortestPath(serverList_1[0], serverList_2[0])
    print("From %s to %s, path has links:" % (serverList_1[0].getKey(), serverList_2[0].getKey()))
    print(path02.getLinkLengthList())
    linkList = path02.getLinkList()
    for l in linkList:
        print("%s, head:%s, tail:%s, length:%d" %(l.getKey(), l.getHeadAP().getKey(), l.getTailAP().getKey(), l.getLinkLength()))

    rootSwitch = tng.getRootSwitch()
    path03 = tng.getShortestPath(rootSwitch, serverList_1[0])
    print("From %s to %s, path has links:" % (rootSwitch.getKey(), serverList_1[0].getKey()))
    print(path03.getLinkLengthList())
    for l in path03.getLinkList():
        print("%s, head:%s, tail:%s, length:%d" %(l.getKey(), l.getHeadAP().getKey(), l.getTailAP().getKey(), l.getLinkLength()))



        

        
        

