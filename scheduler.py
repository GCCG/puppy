# Class Scheduler
import copy
import sys
import numpy as np
import xlrd
import xlwt
from . import task_ex_info
from . import resource_time_table
from . import net_graph
from . import parameters

from . import task_generator
from . import task_type
from . import task
from . import user_type


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
        bestACT = parameters.NUM_FLOAT_INFINITY
        bestIndex = -1
        print("Length of taskQueue:%d." % (len(self._taskQueue)))
        for i in range(len(self._taskQueue)+1):
            tmpTaskQueue = copy.deepcopy(self._taskQueue)
            tmpTaskExInfoDict = copy.deepcopy(self._taskExInfoDict)
            tmpRscTimeTable = copy.deepcopy(self._rscTimeTable)
            ACT, DS = self.__dedasInsert(i, newTask, time, tmpTaskQueue, tmpTaskExInfoDict, tmpRscTimeTable)
            if DS == (self._currentDS + 1) and ACT < bestACT:
                bestIndex = i
                bestACT = ACT
        print("bestIndex: %d" % bestIndex)
        if bestIndex >= 0:
            return bestACT, (self._currentDS + 1), True
        else:
            for i in range(len(self._taskQueue)):
                tmpTaskQueue = copy.deepcopy(self._taskQueue)
                tmpTaskExInfoDict = copy.deepcopy(self._taskExInfoDict)
                tmpRscTimeTable = copy.deepcopy(self._rscTimeTable)
                ACT, DS = self.__dedasReplace(i, newTask, time, tmpTaskQueue, tmpTaskExInfoDict, tmpRscTimeTable)
                if DS == self._currentDS and ACT < bestACT:
                    bestIndex = i
                    bestACT = ACT
            if bestIndex >= 0:
                return bestACT, self._currentDS, True
            else:
                return self._currentACT, self._currentDS, False

    def schedule(self, newTask, time):
        # Set some local variables.
        bestACT = parameters.NUM_FLOAT_INFINITY
        bestIndex = -1
        bestQueue = None
        bestExInfoDict = None
        bestTimeTable = None
        # Try to insert this task into taskQueue
        for i in range(len(self._taskQueue)+1):
            # Copy those info relate to scheduling
            tmpTaskQueue = copy.deepcopy(self._taskQueue)
            tmpTaskExInfoDict = copy.deepcopy(self._taskExInfoDict)
            tmpRscTimeTable = copy.deepcopy(self._rscTimeTable)
            # Insert newTask into tmpTaskQueue
            ACT, DS = self.__dedasInsert(i, newTask, time, tmpTaskQueue, tmpTaskExInfoDict, tmpRscTimeTable)
            # tmpRscTimeTable.printStatus()
            # If this insertion result is better, then record its result
            if DS == (self._currentDS + 1) and ACT < bestACT:
                bestIndex = i
                # Record insertion result.
                bestACT = ACT
                bestQueue = tmpTaskQueue
                bestExInfoDict = tmpTaskExInfoDict
                bestTimeTable = tmpRscTimeTable
        # If insertion works, then set its best result as sheduling result
        if bestIndex >= 0:
            self._currentACT = bestACT
            self._currentDS = self._currentDS + 1
            self._taskQueue = bestQueue
            self._taskExInfoDict = bestExInfoDict
            self._rscTimeTable = bestTimeTable
            # print("Schedule finished:")
            # self._rscTimeTable.printStatus()
        # If insertion doesn't work, then try to replace 
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
            if bestIndex >= 0:
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
        # Recall resources allocated for tasks from current time
        print("---------------------------In scheduler, inserting %s at position %d." % (newTask.getKey(), index))
        tmpPath = None
        for i in range(len(tmpTaskQueue)):
            tmpTask = tmpTaskQueue[i]
            tmpPath = self._netGraph.getShortestPath(tmpTask.getAccessPoint(), tmpTask.getDispatchedServer())
            tmpTaskExInfoDict[tmpTask.getKey()].cancelScheduleFromNow(self._currentTime)
            tmpRscTimeTable.recallRsc(self._currentTime, tmpTask, tmpPath)
            # tmpRscTimeTable.recallRsc(time, tmpTask, tmpPath)
        
        tmpTaskQueue.insert(index, newTask)
        ts = ""
        for t in tmpTaskQueue:
            ts = ts + "." + t.getKey()
        print("In scheduler, tmpTaskQueue is: %s" % (ts))
        # Create a task execution info for new task.
        try:
            tmpTaskExInfoDict[newTask.getKey()]
        except KeyError:
            print("In scheduler, create taskExInfo for %s." % (newTask.getKey()))
            tmpTaskExInfoDict[newTask.getKey()] = task_ex_info.TaskExInfo(newTask, \
            self._netGraph.getShortestPath(newTask.getAccessPoint(), newTask.getDispatchedServer()).getPathLength())
        CTList = []
        # for t in tmpTaskQueue[0:index]:
        #     CTList.append(tmpTaskExInfoDict[t.getKey()].getExpectedComTime())

        tmpDS = 0
        # Reallocate resources for all tasks.
        for i in range(len(tmpTaskQueue)):
            tmpTask = tmpTaskQueue[i]
            print("In scheudler, allocate resources for %s........................." %(tmpTask.getKey()))
            tmpPath = self._netGraph.getShortestPath(tmpTask.getAccessPoint(), tmpTask.getDispatchedServer())
            # First, set starTime, endTime and  remaind rsc which need to allocate.
            remData = tmpTaskExInfoDict[tmpTask.getKey()].getRemDataSize()
            remComTime = tmpTaskExInfoDict[tmpTask.getKey()].getRemComTime()
            # Data trans starts at time, but in link rsc allocating loop,
            # (startTime + 1) is passed as parameter, so subtract 1.
            startTime = time - 1
            endTime = 0
            # Allocate bandwidth resource and record it
            while remData > 0:
                startTime, endTime, ban = tmpRscTimeTable.allocateLinkSlot(tmpTask, startTime + 1, tmpPath)
                if ban == 0:
                    break
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
        print("In scheduler, insertion ACT:%d, DS:%s." % (sum(CTList)/len(tmpTaskQueue), tmpDS))
        taskCTDict = {}
        for i in range(len(CTList)):
            taskCTDict[tmpTaskQueue[i].getKey()] = CTList[i]
        print("In schedule, tasks' completion time is:",taskCTDict)
        tmpRscTimeTable.printStatus()
        return sum(CTList)/len(tmpTaskQueue), tmpDS
    
    def __dedasReplace(self, index, newTask, time, tmpTaskQueue, tmpTaskExInfoDict, tmpRscTimeTable):
        """ 
        If replace is failed, then return schedule result before replace.
         """
        print("--------------In scheduler, replacing some task with %s." % (newTask.getKey()))
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
                if ban == 0:
                    break
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
    def printRTTStatus(self):
        self._rscTimeTable.printStatus()

if __name__ == "__main__":
    ng = net_graph.createANetGraph()
    ds = DedasScheduler(ng, 40, 0)
    taskList = []
    serverList = ng.getServerList()
    taskTypeNameList = [parameters.CODE_TASK_TYPE_IoT, parameters.CODE_TASK_TYPE_VA, parameters.CODE_TASK_TYPE_VR]
    for i in range(len(serverList)):
        for name in taskTypeNameList:
            tmpTask = task.Task(name, serverList[i], 40, 0, 5, 10)
            tmpTask.setDispatchedServer(serverList[np.random.randint(0, len(serverList))])
            taskList.append(tmpTask)
    print("task num is:%d" % (len(taskList)))

    for t in taskList[0:4]:
        #ACT, DS, taskIsSatisfied = ds.schedulePlan(t, 0)
        #print("Plan: ACT:%d, DS:%d, taskIsSatisfied:%s" % (ACT, DS, taskIsSatisfied))
        ACT, DS = ds.schedule(t, 0)
        print("Schedule: ACT:%d, DS:%d." % (ACT, DS))
        ds.printRTTStatus()
    
        

            




    



        