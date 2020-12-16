# Class Scheduler
import copy
import sys
from . import task_ex_info
from . import resource_time_table
from . import net_graph
from . import parameters


class Scheduler:
    def __init__(self, timeTableLen, currentTime):
        self._currentTime = currentTime
        self._taskNum = 0
        self._rscTimeTable = resource_time_table.ResourceTimeTable(1, timeTableLen, currentTime)
        self._taskExInfoDict = {}
        self._currentACT = parameters.NUM_FLOAT_INFINITY
        self._currentDS = 0

    def schedule(self, newTask, time):
        """ task provided here should be assigned with a server. """
        sys.exit("You should implement function schedule in subclass")
        pass

    def schedulePlan(self, newTask, time):
        sys.exit("You should implement function schedulePlan in subclass")
        pass

    def getDR(self):
        return self._currentDS/self._taskNum

    def getNetGraph(self):
        sys.exit("You should implement function getNetGraph in subclass")
        pass

    def getResult(self):
        return self._currentACT, self._currentDS


class DedasScheduler(Scheduler):
    def __init__(self, netGraph, timeTableLen, currentTime):
        Scheduler.__init__(self, timeTableLen, currentTime)

        # Initialize netGraph and rscTimeTable.
        self._netGraph = netGraph
        for s in self._netGraph.getServerList():
            self._rscTimeTable.addServerRsc(s)
        for l in self._netGraph.getLinkList():
            self._rscTimeTable.addLinkRsc(l)

        # Each state of taskQueue corresponds to a state of rscTimeTable.
        # So once state of taskQueue is fixed, the state of rscTimeTable is fixed
        self._taskQueue = []

    def schedulePlan(self, newTask, time):
        bestACT = self._currentACT
        bestIndex = -1
        for i in range(len(self._taskQueue)+1):
            tmpTaskQueue = copy.deepcopy(self._taskQueue)
            tmpTaskExInfoDict = copy.deepcopy(self._taskExInfoDict)
            tmpRscTimeTable = copy.deepcopy(self._rscTimeTable)
            ACT, DS = self.__dedasInsert(i, newTask, time, tmpTaskQueue, tmpTaskExInfoDict, tmpRscTimeTable)
            if DS == (self._currentDS + 1) and ACT < bestACT:
                bestIndex = i
                bestACT = ACT
        if bestIndex > 0:
            return bestACT, (self._currentDS + 1), True
        else:
            for i in range(len(self._taskQueue)):
                tmpTaskQueue = copy.deepcopy(self._taskQueue)
                tmpTaskExInfoDict = copy.deepcopy(self._taskExInfoDict)
                tmpRscTimeTable = copy.deepcopy(self._rscTimeTable)
                ACT, DS = self.__dedasReplace(i, newTask, time, time, tmpTaskQueue, tmpTaskExInfoDict, tmpRscTimeTable)
                if DS == self._currentDS and ACT < bestACT:
                    bestIndex = i
                    bestACT = ACT
            if bestIndex > 0:
                return bestACT, self._currentDS, True
            else:
                return self._currentACT, self._currentDS, False

    def schedule(self, newTask, time):
        bestACT = self._currentACT
        bestIndex = -1
        bestQueue = None
        bestExInfoDict = None
        bestTimeTable = None
        for i in range(len(self._taskQueue)+1):
            tmpTaskQueue = copy.deepcopy(self._taskQueue)
            tmpTaskExInfoDict = copy.deepcopy(self._taskExInfoDict)
            tmpRscTimeTable = copy.deepcopy(self._rscTimeTable)
            ACT, DS = self.__dedasInsert(i, newTask, time, tmpTaskQueue, tmpTaskExInfoDict, tmpRscTimeTable)
            if DS == (self._currentDS + 1) and ACT < bestACT:
                bestIndex = i
                bestACT = ACT
                bestQueue = tmpTaskQueue
                bestExInfoDict = tmpTaskExInfoDict
                bestTimeTable = tmpRscTimeTable
        if bestIndex > 0:
            self._currentACT = bestACT
            self._currentDS = self._currentDS + 1
            self._taskQueue = bestQueue
            self._taskExInfoDict = bestExInfoDict
            self._rscTimeTable = bestTimeTable
        else:
            for i in range(len(self._taskQueue)):
                tmpTaskQueue = copy.deepcopy(self._taskQueue)
                tmpTaskExInfoDict = copy.deepcopy(self._taskExInfoDict)
                tmpRscTimeTable = copy.deepcopy(self._rscTimeTable)
                ACT, DS = self.__dedasReplace(i, newTask, time, tmpTaskQueue, tmpTaskExInfoDict, tmpRscTimeTable)
                if DS == self._currentDS and ACT < bestACT:
                    bestIndex = i
                    bestACT = ACT
                    bestQueue = tmpTaskQueue
                    bestExInfoDict = tmpTaskExInfoDict
                    bestTimeTable = tmpRscTimeTable
            if bestIndex > 0:
                self._currentACT = bestACT
                self._taskQueue = bestQueue
                self._taskExInfoDict = bestExInfoDict
                self._rscTimeTable = bestTimeTable
        return self._currentACT, self._currentDS

    def update(self, currentTime):
        """ 
        Each after the scheduling, we should use this funciton to 
        update scheduler state to fix scheduling result of current time slot.
         """
        if self._currentTime == currentTime - 1:
            self._currentTime = currentTime
        else:
            print("Something is wrong with your dispatching loop.")
            return

        disposeList = []
        for i in len(self._taskQueue):
             self._taskExInfoDict[self._taskQueue[i].getKey()].exInfoUpdate(currentTime)
             if self._taskExInfoDict[self._taskQueue[i].getKey()].comIsFinished():
                 tmpExInfo = self._taskExInfoDict[self._taskQueue[i].getKey()]
                 if tmpExInfo.getExpectedComTime() != tmpExInfo.getComplettionTime():
                     print("Something is wrong with your task execution information.")
                 disposeList.append(i)
        # Dispose taskExInfoDict and taskQueue
        for index in disposeList:
            del self._taskExInfoDict[self._taskQueue[index].getKey()]
            del self._taskQueue[index]
        # CTList = []
        # for t in self._taskQueue:
        #     CTList.append(self._taskExInfoDict[t.getKey()].getExpectedComTime())
        # self._currentACT = sum(CTList)/len(self._taskQueue)

    def __isJoint(self, taskA, taskB):
        """ 
        Check if two tasks has joint link or server. 
        """
        if taskA.getDispatchedServer().getKey() == taskB.getDispatchedServer(). getKey():
            return True
        else:
            pathA = self._netGraph.getShortestPath(taskA.getAccessPoint(), taskA.getDispatchedServer())
            pathB = self._netGraph.getShortestPath(taskB.getAccessPoint(), taskB.getDispatchedServer())
            for la in pathA.getLinkList():
                for lb in pathB.getLinkList():
                    if la.getKey() == lb.getKey():
                        return True
            return False

    def __dedasInsert(self, index, newTask, time, tmpTaskQueue, tmpTaskExInfoDict, tmpRscTimeTable):
        """ 
        If insertion is failed, then return schedule result before insertion.
         """
        # Recall resources allocated for tasks
        tmpPath = None
        for i in range(len(self._taskQueue) - index):
            tmpTask = self._taskQueue[i]
            tmpPath = self._netGraph.getShortestPath(tmpTask.getAccessPoint(), tmpTask.getDispatchedServer())
            tmpTaskExInfoDict[tmpTask.getKey()].cancelScheduleFromNow(self._currentTime)
            tmpRscTimeTable.recallRsc(time, tmpTask, tmpPath)
        
        tmpTaskQueue.insert(index, newTask)
        tmpTaskExInfoDict[newTask.getKey()] = task_ex_info.TaskExInfo(newTask, \
            sum(self._netGraph.getShortestPath(newTask.getAccessPoint(), newTask.getDispatchedServer()).getLinkList()))
        CTList = []
        for t in tmpTaskQueue[:index]:
            CTList.append(tmpTaskExInfoDict[t.getKey()].getExpectedComTime())
        tmpDS = index
        # Reallocate resources for task which position is not smaller than index.
        for i in range(len(tmpTaskQueue) - index):
            tmpTask = tmpTaskQueue[index + i]
            tmpPath = self._netGraph.getShortestPath(tmpTask.getAccessPoint(), tmpTask.getDispatchedServer())
            # First, allocate bandwidth resource and record it
            remData = tmpTask.getDataSize()
            remComTime = tmpTask.getComputeTime()
            startTime = time - 1
            endTime = 0
            # tmpTaskExInfoDict[newTask.getKey] = task_ex_info.TaskExInfo(newTask, sum(tmpPath.getLinkList()))
            while remData > 0:
                startTime, endTime, ban = tmpRscTimeTable.allocateLinkSlot(tmpTask, startTime + 1, tmpPath)
                remData = remData - ban
                tmpTaskExInfoDict[tmpTask.getKey()].addTransInfo(ban, startTime)
            # Seconde, allocate computation resource and record it
            while remComTime > 0:
                startTime, endTime = tmpRscTimeTable.allocateServerSlot(tmpTask,endTime, tmpTask.getDispatchedServer())
                remComTime = remComTime - 1
                tmpTaskExInfoDict[tmpTask.getKey()].addComInfo(startTime)
            CTList.append(endTime - time)
            tmpTaskExInfoDict[tmpTask.getKey()].setExpectedComTime(endTime - time)
            if tmpTaskExInfoDict[tmpTask.getKey()].deadlineIsSatisfied():
                tmpDS = tmpDS + 1
            else:
                return self._currentACT, self._currentDS
        return sum(CTList)/len(tmpTaskQueue), tmpDS
    
    def __dedasReplace(self, index, newTask, time, tmpTaskQueue, tmpTaskExInfoDict, tmpRscTimeTable):
        """ 
        If replace is failed, then return schedule result before replace.
         """
        tmpTask = tmpTaskQueue[index]
        tmpTaskQueue[index] = newTask
        tmpPath = self._netGraph.getShortestPath(tmpTask.getAccessPoint(), tmpTask.getDispatchedServer())
        tmpTaskExInfo = tmpTaskExInfoDict[tmpTask.getKey()]

        if not self.__isJoint(newTask, tmpTask):
            return self._currentACT, self._currentDS
        
        if tmpTaskExInfo.getRemDataSize() >= newTask.getDataSize() and \
            tmpTaskExInfo.getRemComTime() >= newTask.getComputeTime():
            # Recall remained resources of replaced task and wipe record about it.
            tmpRscTimeTable.recallRsc(time, tmpTask, tmpPath)
            tmpTaskExInfo.cancelScheduleFromNow(time)
            # Get info of new task
            newPath = self._netGraph.getShortestPath(newTask.getAccessPoint(), newTask.getDispatchedServer())
            remData = newTask.getDataSize()
            remComTime = newTask.getComputeTime()
            startTime = time-1
            endTime = 0
            # First, allocate bandwidth resource and record it.
            while remData > 0:
                startTime, endTime, ban = tmpRscTimeTable.allocateLinkSlot(newTask, startTime + 1, newPath)
                remData = remData - ban
                tmpTaskExInfoDict[newTask.getKey()].addTransInfo(ban, startTime)
            # Then allocate computation resource and record it
            while remComTime > 0:
                startTime, endTime = tmpRscTimeTable.allocateServerSlot(newTask, endTime, tmpTask.getDispatchedServer())
                remComTime = remComTime - 1
                tmpTaskExInfoDict[newTask.getKey()].addComInfo(startTime)
            tmpTaskExInfoDict[newTask.getKey()].setExpectedComTime(endTime - time)
            if not tmpTaskExInfoDict[newTask.getKey()].deadlineIsSatisfied():
                # return self._currentACT, self._currentDS
                sys.exit("Something is wrong with your function __dedasReplace.")
            else:
                CTList = []
                for t in tmpTaskQueue:
                    CTList.append(tmpTaskExInfoDict[t.getKey()].getExpectedComTime())
                return sum(CTList)/len(tmpTaskQueue), self._currentDS
        else:
            return self._currentACT, self._currentDS

if __name__ == "__main__":
    pass
        