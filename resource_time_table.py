# Class ResourceTimeTable
import sys
from . import slot
from . import parameters
from . import net_ap
from . import net_link

from . import net_graph
from . import task

class ResourceTimeTable:
    def __init__(self, slotSize, slotNum, currentTime):
        self._slotSize = slotSize
        self._rscNum = 0
        self._slotNum = slotNum
        # rsc2SlotRowDict, its key is generated by link's or server's getKey() method, its value is a list of Slot objects
        self._rsc2SlotRowDict = {}
        # slotStartTime is the time of the first slot in slot row.
        self._slotStartTime = 0
        # The end of the slot row, is the furthest allocated slot index or the current slot index. 
        self._rsc2SloRowEnds = {}
        self._currentTime = currentTime
        self._taskAlcLinkInfoDict = {}
        self._taskAlcSerInfoDict = {}

    def addLinkRsc(self, link):
        tmpRow = []
        print("In addLinkRsc, bandwidth:%d" % (link.getBandwidth()))
        for i in range(self._slotNum):
            tmpRow.append(slot.Slot(parameters.CODE_RSC_LINK, link.getBandwidth()))
        self._rscNum = self._rscNum + 1
        self._rsc2SlotRowDict[link.getKey()] = tmpRow
        self._rsc2SloRowEnds[link.getKey()] = self._currentTime - self._slotStartTime

    def addServerRsc(self, server):
        tmpRow = []
        for i in range(self._slotNum):
            tmpRow.append(slot.Slot(parameters.CODE_RSC_SERVER, server.getRscAmount()))
        self._rsc2SlotRowDict[server.getKey()] = tmpRow
        self._rsc2SloRowEnds[server.getKey()] = self._currentTime - self._slotStartTime

    def allocateLinkSlot(self, task, time, path):
        if time < self._currentTime:
            sys.exit("Argument 'time' is smaller than current time.")
        index, slotBan = self.__searchAvailableLinkSlot(time - self._slotStartTime, path)
        
        # Try to insert the index of link slot for this task,
        # if failed, then creat a index list for this task
        try:
            self._taskAlcLinkInfoDict[task.getKey()].append(index)
        except KeyError:
            self._taskAlcLinkInfoDict[task.getKey()] = [index]
        
        linkLenList = path.getLinkLengthList()
        path.reset()
        # Allocate slots of links in path for this task.
        tmpIndex = index
        for i in range(len(linkLenList)):
            linkKey = path.nextLink().getKey()
            # Add task into corresponding link slot.
            self._rsc2SlotRowDict[linkKey][tmpIndex].addTask(task, slotBan)
            # Update slot row end
            self.__updateSlotRowEnd(linkKey, tmpIndex + self._slotStartTime)

            tmpIndex = tmpIndex + linkLenList[i]
        # Return time of allocated slot and allocated bandwidth.
        startTime = index + self._slotStartTime
        endTime = tmpIndex + self._slotStartTime
        return startTime, endTime, slotBan

    def allocateServerSlot(self, task, time, server):
        if time < self._currentTime:
            sys.exit("Argument 'time' is smaller than current time.")
        index  = self.__searchAvailableServerSlot(time - self._slotStartTime, server)

        # Try to insert the index of server slot for this task,
        # if failed, then creat a index list for this task.
        try:
            self._taskAlcSerInfoDict[task.getKey()].append(index)
        except KeyError:
            self._taskAlcSerInfoDict[task.getKey()] = [index]

        self._rsc2SlotRowDict[server.getKey()][index].addTask(task, 1)
        # Return time of allocated slot
        startTime = index + self._slotStartTime
        endTime = startTime + 1
        return startTime, endTime       

    def freeRscFromCurrentTime(self):
        currentIndex = self._currentTime - self._slotStartTime
        for k in list(self._rsc2SlotRowDict.keys()):
            for i in range(self._rsc2SloRowEnds[k] - currentIndex):
                self._rsc2SlotRowDict[k][currentIndex + i].free()

    def recallRsc(self, time, task, path):
        linkLenList = path.getLinkLengthList()

        # Free link slot resource.
        for index in self._taskAlcLinkInfoDict[task.getKey()]:
            tmpIndex = index
            path.reset()
            for len in linkLenList:
                if index + self._slotStartTime < time:
                    print("History should be reserved.")
                    continue
                if self._rsc2SlotRowDict[path.nextLink().getKey][tmpIndex].deleteTask(task) != True:
                    sys.exit("No such task in this link slot")
                tmpIndex = tmpIndex + len

        # Free server slot resource.
        for index in self._taskAlcSerInfoDict[task.getKey()]:
            if self._rsc2SlotRowDict[task.getDispatchedServer().getKey()][index].deleteTask(task) != True:
                sys.exit("No such task in this server slot")


        
    def updateTime(self):
        self._currentTime = self._currentTime + 1
        for k in list(self._rsc2SloRowEnds.keys()):
            self.__updateSlotRowEnd(k, self._currentTime)
    
    def __rscKey(self, rsc):
        if type(rsc) == net_ap.NetAP:
            return 'server-' + str(rsc.getID())
        elif type(rsc) == net_link.NetLink:
            return 'link-' + str(rsc.getID())

    def __checkPathBandwidth(self, index, path):
        distList = path.getLinkLengthList()
        path.reset()
        minBan = self._rsc2SlotRowDict[path.nextLink().getKey()][index].remainedRsc()
        print("In resource_time_table.py, checkPathBandwidth:")
        print(minBan)
        for i in range(len(distList)-1):
            tmpLink = path.nextLink()
            print(tmpLink.getKey())
            tmpBan = self._rsc2SlotRowDict[tmpLink.getKey()][index+distList[i]].remainedRsc()
            print(tmpBan)
            index = index + distList[i]
            if tmpBan < minBan:
                minBan = tmpBan
        return minBan
    
    def __searchAvailableLinkSlot(self, index, path):
        slotBan = 0
        while slotBan <= 0:
            slotBan = self.__checkPathBandwidth(index, path)
            index = index + 1
        return index-1, slotBan
    
    def __searchAvailableServerSlot(self, index, server):
        serverSlots = self._rsc2SlotRowDict[self.__rscKey(server)]
        while serverSlots[index].remainedRsc() <=0:
            index = index + 1
        return index

    def __updateSlotRowEnd(self, rscKey, time):
        if time - self._slotStartTime > self._rsc2SloRowEnds[rscKey]:
            self._rsc2SloRowEnds[rscKey] = time - self._slotStartTime

if __name__ == "__main__":
    ng = net_graph.createANetGraph()
    rtt = ResourceTimeTable(1, 100, 0)
    linkList = ng.getLinkList()
    serverList = ng.getServerList()
    for l in linkList:
        rtt.addLinkRsc(l)
    for s in serverList:
        rtt.addServerRsc(s)
    path1 = ng.getShortestPath(serverList[0], serverList[4])
    
    tmpTask = task.Task(parameters.CODE_TASK_TYPE_IoT, serverList[0], 20, 0, 5,4)
    tmpTask.setDispatchedServer(serverList[4])
    print("start_ap:%s, end_ap:%s." % (serverList[0].getKey(), serverList[4].getKey()))
    startTime, endTime, ban = rtt.allocateLinkSlot(tmpTask, time=0, path=path1)
    print("Data transmission, start_time:%d, end_time:%d, bandwidth:%d" % (startTime, endTime, ban))
    startTime, endTime = rtt.allocateServerSlot(tmpTask, endTime, tmpTask.getDispatchedServer())
    print("Computation, start_time:%d, end_time:%d." % (startTime, endTime))