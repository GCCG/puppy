# Task class, for modeling tasks in experiment
import sys
import numpy as np
from . import net_ap
from . import group
from . import group_type
from . import parameters

class Task:
    createdTaskNum = 0
    def __init__(self, taskTypeName, accessPoint, deadline, birthTime, dataSize, computeSize):
        self._taskID = Task.createdTaskNum
        Task.createdTaskNum = Task.createdTaskNum + 1
        self._taskTypeName = taskTypeName
        self._accessPoint = accessPoint 
        self._deadline = deadline 
        self._birthTime = birthTime
        self._dataSize = dataSize
        # self._computeTime = 0
        self._computaSize = computeSize

        self._dispatchedServer = None # If None, then it means

    def getComputeTime(self):
        return self._computaSize / self._dispatchedServer.getRscAmount()

    def setDispatchedServer(self, server):
        self._dispatchedServer = server
        # self._computeTime = self._computaSize / server.getRscAmount()

    def getDispatchedServer(self):
        if self._dispatchedServer == None:
            sys.exit("No server is dispatched to this task")
        else:
            return self._dispatchedServer
    
    def getTaskTypeName(self):
        return self._taskTypeName
    
    def getAccessPoint(self):
        return self._accessPoint

    def getDeadline(self):
        return self._deadline

    def getBirthTime(self):
        return self._birthTime

    def getDataSize(self):
        return self._dataSize

    def __eq__(self, other):
        return self._taskID == other.getID()
    
    def getID(self):
        return self._taskID

    def getKey(self):
        return "task-"+str(self._taskID)

def createTasks():
    gp = group.createAGroup()
    taskList = []
    print("What's wrong?")
    for s in gp.getServerList():
        print(s.getKey())
        for i in range(10):
            taskList.append(Task(parameters.CODE_TASK_TYPE_IoT, s, 20,5, 10, 10))
    return taskList

if __name__=="__main__":
    # gt =  group_type.createAGroupType()
    gp = group.createAGroup()
    taskList = []
    # print("What's wrong?")
    for s in gp.getServerList():
        print(s.getKey())
        for i in range(10):
            taskList.append(Task(parameters.CODE_TASK_TYPE_IoT, s, 20,5, 10, 10))
    for t in taskList:
        t.setDispatchedServer(gp.getServerList()[np.random.randint(0,3)])
        print("Task %s is generated in %s, data_size:%d, deadline:%d, type_name:%s, compute_time:%d." \
            % (t.getKey(), t.getAccessPoint().getKey(), t.getDataSize(), t.getDeadline(), \
            t.getTaskTypeName(), t.getComputeTime()))
    