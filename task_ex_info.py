# Class TaskExInfo. When resource is allocated to corresponding task,
# we should update it execution information in a TaskExInfo object.
import sys


class TaskExInfo:
    def __init__(self, task, pathLen):
        self._arrivalTime = task.getBirthTime()
        self._taskDeadline = task.getDeadline()
        self._taskID = task.getID()
        self._pathLen = pathLen
        self.transWaitTimeLen = 0
        self.transTimeLen = 0
        self.comWaitTimeLen = 0
        self.comTimeLen = task.getComputeTime()

        self.remData = task.getDataSize()
        self.remComTime = task.getComputeTime()
        
        self._transStartTime = -1
        self._transEndTime = -1
        self._comStartTime = -1
        self._comEndTime = -1

        self._transTimeList = []
        self._transBanList = []
        self._comTimeList = []
        self._expectCompletionTime = 0
    
    def getCompletionTime(self):
        return self.transTimeLen + self.transWaitTimeLen + self._pathLen*2 + self.comWaitTimeLen + self.comTimeLen

    def setExpectedComTime(self, comTime):
        self._expectCompletionTime = comTime

    def getExpectedComTime(self):
        return self._expectCompletionTime

    def getRemDataSize(self):
        return self.remData

    def getRemComTime(self):
        return self.remComTime

    def addTransInfo(self, ban, startTime):
        if self._transTimeList[len(self._transTimeList)-1] >= startTime:
            sys.exit("Allocated time for transmition is smaller than the latest one.")
        if ban <= 0:
            sys.exit("Allocated bandwidth should be positive.")
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
        if startTime <= self._comTimeList[len(self._comTimeList)-1]:
            sys.exit("Allocated time for computation is smaller than the latest one.")
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
        return self._expectCompletionTime <= self._taskDeadline

    def cancelScheduleFromNow(self, currentTime):
        for i in range(len(self._transTimeList)):
            if self._transTimeList[i] >= currentTime:
                del self._transTimeList[i]
                del self._transBanList[i]
        for j in range(len(self._comTimeList)):
            if self._comTimeList[j] >= currentTime:
                del self._comTimeList[j]

    def exInfoUpdate(self, time):
        for i in range(len(self._transTimeList)):
            if self._transTimeList[i] == time:
                if self._transStartTime == -1:
                    self._transStartTime = time
                    self._transEndTime = time
                self.transWaitTimeLen = self.transWaitTimeLen + time - self._transEndTime
                self.transTimeLen = self.transTimeLen + 1
                self._transEndTime = time + 1
                if self.remData < self._transBanList[i]:
                    self.remData = 0
                    print("Data transmission of task %d is finished.", self._taskID)
                    return False
                else:
                    self.remData = self.remData - self._transBanList[i]
                    return False
            elif self._transTimeList[i] > time:
                break
        for i in range(len(self._comTimeList)):
            if self._comTimeList[i] > time:
                break
            elif self._comTimeList[i] == time:
                if self._comStartTime == -1:
                    self._comStartTime = time
                    self._comEndTime = time
                    self.comWaitTimeLen = self._comStartTime - self._transEndTime - self._pathLen
                self.comWaitTimeLen = self.comWaitTimeLen + time - self._comEndTime
                self._comEndTime = time + 1
                if self.remComTime < 1:
                    self._comEndTime = 0
                    print("Computation of task %d is finished.", self._taskID)
                    return True
                else:
                    self.remComTime = self.remComTime - 1
                    return False


