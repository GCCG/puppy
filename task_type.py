# Class TaskType
from . import task
from . import parameters


class TaskType:
    def __init__(self, defaultDataSize, defaultComputeSize, defaultTimeLimit, typeName):
        self._defaultDataSize = defaultDataSize
        self._defaultComputeSize = defaultComputeSize
        self._typeName = typeName
        self._defaultTimeLimit = defaultTimeLimit

    
    def createTask(self, accessPoint, birthTime, timeLimit=None):
        if timeLimit == None:
            timeLimit = self._defaultTimeLimit
        return task.Task(self._typeName, accessPoint, birthTime+timeLimit,
        self._defaultDataSize, self._defaultComputeSize)

    def getTaskTypeName(self):
        return self._typeName

if __name__=='__main__':
    tp = TaskType(20, 15, 30, parameters.CODE_TASK_TYPE_VA)
