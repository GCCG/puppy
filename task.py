# Task class, for modeling tasks in experiment
import sys

class Task:
    createdTaskNum = 0
    def __init__(self, taskType, accessPoint, deadline, birthTime, dataSize, computeSize):
        self._taskID = Task.createdTaskNum
        Task.createdTaskNum = Task.createdTaskNum + 1
        self._taskType = taskType
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
    
    def getTaskType(self):
        return self._taskType
    
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

if __name__=="__main__":
    pass