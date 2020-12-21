# Class TaskExInfo. When resource is allocated to corresponding task,
# we should update it execution information in a TaskExInfo object.
import sys
import json
import math

from . import task
from . import group
from . import net_graph
from . import parameters


class TaskExInfo:
    def __init__(self, task, pathLen):
        self._arrivalTime = task.getBirthTime()
        self._taskDeadline = task.getDeadline()
        self._taskID = task.getID()
        self._pathLen = pathLen
        self.transWaitTimeLen = 0
        self.transTimeLen = 0
        self.comWaitTimeLen = 0
        self.comTimeLen = math.ceil(task.getComputeTime())

        self.remData = task.getDataSize()
        self.remComTime = task.getComputeTime()
        
        self._transStartTime = -1
        self._transEndTime = -1
        self._comStartTime = -1
        self._comEndTime = -1

        self._transTimeList = []
        self._transBanList = []
        self._comTimeList = []
        self._expectCompletionTime = -1
        if self._pathLen == 0:
            print("In task_ex_info, %s has 0 path length, startAP:%s, endAP:%s." % \
                (task.getKey(), task.getAccessPoint().getKey(), task.getDispatchedServer().getKey()))
            self.transWaitTimeLen = 0
            self._transStartTime = task.getBirthTime()
            self._transEndTime = task.getBirthTime()
            self._remData = 0
            self.transTimeLen = 0
    
    

    
    def getCompletionTime(self):
        return self.transTimeLen + self.transWaitTimeLen + self._pathLen*2 + self.comWaitTimeLen + self.comTimeLen

    def setExpectedComTime(self, comEndTime):
        self._expectCompletionTime = comEndTime - self._arrivalTime

    def getExpectedComTime(self):
        return self._expectCompletionTime + self._pathLen

    def getRemDataSize(self):
        return self.remData

    def getRemComTime(self):
        return self.remComTime

    def addTransInfo(self, ban, startTime):
        if len(self._transTimeList) > 0 and startTime <= self._transTimeList[-1]:
            sys.exit("In task_ex_info, new added trans time should be bigger than latest one.")
        if len(self._transTimeList) > 0 and self._transTimeList[len(self._transTimeList)-1] >= startTime:
            sys.exit("In task_ex_info, allocated time for transmition is smaller than the latest one.")
        if ban <= 0:
            sys.exit("In task_ex_info, allocated bandwidth should be positive.")
        self._transTimeList.append(startTime)
        self._transBanList.append(ban)
        pass
        # if self._transStartTime == -1:
        #     self._transStartTime = startTime
        #     self._transEndTime = startTime

        # self.transWaitTimeLen = self.transWaitTimeLen + (startTime - self._transEndTime)
        # self.transTimeLen = self.transTimeLen + 1
        # self._transEndTime = startTime + 1
        # if self.remComTime - ban <= 0:
        #     # print("Task data transmission finished.")
        #     self.remData = 0
        #     return True
        # else:
        #     self.remData = self.remData - ban
        #     return False

    def addComInfo(self, startTime):
        if len(self._comTimeList) > 0 and startTime <= self._comTimeList[-1]:
            sys.exit("In task_ex_info, com start time should be bigger than latest one.")
        if len(self._comTimeList) > 0 and startTime <= self._comTimeList[len(self._comTimeList)-1]:
            sys.exit("In task_ex_info, allocated time for computation is smaller than the latest one.")
        else:
            self._comTimeList.append(startTime)
        pass
        # if self._comStartTime == -1:
        #     self._comStartTime = startTime
        #     self._comEndTime = startTime
        # self.comWaitTimeLen = self.comWaitTimeLen + (startTime - self._comEndTime)
        # self._comEndTime = startTime + timeLen
        # if self.remComTime - timeLen <= 0:
        #     self.remComTime = 0
        #     return True
        # else:
        #     self.remComTime = self.remComTime - timeLen
            # return False
    
    def transIsFinished(self):
        return self.remData == 0
    
    def comIsFinished(self):
        return self.remComTime == 0

    def deadlineIsSatisfied(self):
        if self._expectCompletionTime == -1:
            return False
        return self._expectCompletionTime <= self._taskDeadline

    def cancelScheduleFromNow(self, currentTime):
        if self.remComTime == 0 and currentTime >= self._comEndTime:
            sys.exit("In task_ex_info, task has been finished at time %d, nothing to cancel." % (self._comEndTime))
        for i in range(len(self._transTimeList)):
            if self._transTimeList[i] >= currentTime:
                self._transBanList = self._transBanList[0:i]
                self._transTimeList = self._transTimeList[0:i]
                break
                # del self._transTimeList[i]
                # del self._transBanList[i]
        for j in range(len(self._comTimeList)):
            if self._comTimeList[j] >= currentTime:
                self._comTimeList = self._comTimeList[0:j]
                break
        print("In task_ex_info, task-%d's execution info cancelation from time %d finished." % (self._taskID, currentTime))
        print("transTimeList:",self._transTimeList)
        print("comTimeList:",self._comTimeList)

    def exInfoUpdate(self, time):
        # print("remComTime:%d, comEndTime:%d" % (self.remComTime, self._comEndTime))
        print("Before update task-%d' ex info, info is:" % (self._taskID))
        print(self.__dict__)
        if self.remComTime == 0 and self._comEndTime < time:
            print("In task_ex_info, something is wrong with your program, task has finished at time %d, current time is %d" %(self._comEndTime, time))
        for i in range(len(self._transTimeList)):
            if self._transTimeList[i] == time:
                if self._transStartTime == -1:
                    self._transStartTime = time
                    self._transEndTime = time + self._pathLen + 1
                    self.transWaitTimeLen =  time - self._arrivalTime
                else:
                    self.transWaitTimeLen = self.transWaitTimeLen + time - (self._transEndTime - self._pathLen)
                self.transTimeLen = self.transTimeLen + 1
                self._transEndTime = time + self._pathLen + 1
                if self.remData < self._transBanList[i]:
                    self.remData = 0
                    print("Data transmission of task %d is finished at time %d." % (self._taskID, time))
                    return False
                else:
                    self.remData = self.remData - self._transBanList[i]
                    return False
            elif self._transTimeList[i] > time:
                break
        for i in range(len(self._comTimeList)):
            if self._comTimeList[i] == time:
                if self._comStartTime == -1:
                    # print("setting")
                    if time < self._transEndTime:
                        sys.exit("Something is wrong with your program, task computation should start after time %d, now is %d" % (self._transEndTime, time))
                    self._comStartTime = time
                    self._comEndTime = time + 1
                    self.comWaitTimeLen = self._comStartTime - self._transEndTime #- 1
                else:
                    self.comWaitTimeLen = self.comWaitTimeLen + time - self._comEndTime
                
                if self.remComTime >= 1:
                    self._comEndTime = time + 1
                    self.remComTime = self.remComTime - 1
                    return False
                elif self.remComTime < 1:
                    self._comEndTime = time + 1
                    self.remComTime = 0
                    #print("Computation of task %d finished at time %d., now at time %d" % (self._taskID, self._comEndTime, time))
                    print("Computation of task %d finished at time %d., now at time %d" % (self._taskID, self._comEndTime, time))
                    return True

    def getSchedule(self):
        pass

    def jsonfy(self):
        return json.dumps(self.__dict__, indent=4)



if __name__ == "__main__":
    ng = net_graph.createANetGraph()
    serverList = ng.getServerList()
    print("------------Test base functions:")
    tmpTask = task.Task(parameters.CODE_TASK_TYPE_IoT, serverList[0], 20, 0, 5, 10)
    tmpTask.setDispatchedServer(serverList[4])
    tmpPath = ng.getShortestPath(tmpTask.getAccessPoint(), tmpTask.getDispatchedServer())

    tei = TaskExInfo(tmpTask, 3)
    tei.addTransInfo(2, 3)
    tei.addTransInfo(1, 4)
    tei.addTransInfo(3, 5)
    # tei.addComInfo(8)
    tei.addComInfo(9)
    tei.addComInfo(10)
    tei.addComInfo(11)
    for i in range(15):
        print("Iteration %d:" % (i))
        tei.exInfoUpdate(i)
    # tei.cancelScheduleFromNow(16)
    print("------------Test cancel function:")
    tmpTask = task.Task(parameters.CODE_TASK_TYPE_IoT, serverList[0], 20, 0, 5, 10)
    tmpTask.setDispatchedServer(serverList[4])
    tmpPath = ng.getShortestPath(tmpTask.getAccessPoint(), tmpTask.getDispatchedServer())

    tei = TaskExInfo(tmpTask, 3)
    tei.addTransInfo(2, 3)
    tei.addTransInfo(1, 4)
    tei.addTransInfo(3, 5)
    # tei.addComInfo(8)
    tei.addComInfo(9)
    tei.addComInfo(10)
    tei.addComInfo(11)
    for i in range(20):
        print("Iteration %d:" % (i))
        if i ==10:
            tei.cancelScheduleFromNow(i)
            tei.addComInfo(13)
            tei.addComInfo(14)
        tei.exInfoUpdate(i)
    print(tei.jsonfy())



