# Class TaskType
import sys

from . import task
from . import parameters
from . import group
from . import net_ap


class TaskType:
    def __init__(self, defaultDataSize, defaultComputeSize, defaultTimeLimit, typeName):
        self._defaultDataSize = defaultDataSize
        self._defaultComputeSize = defaultComputeSize
        self._typeName = typeName
        self._defaultTimeLimit = defaultTimeLimit

    
    def createTask(self, accessPoint, birthTime, timeLimit=None):
        if type(accessPoint) != net_ap.NetAP:
            sys.exit("In task_type, accessPoint should be an NetAP object, but it is of type %s" % (type(accessPoint)))
        if timeLimit == None:
            timeLimit = self._defaultTimeLimit
        return task.Task(self._typeName, accessPoint, birthTime+timeLimit, birthTime,
        self._defaultDataSize, self._defaultComputeSize)

    def getTaskTypeName(self):
        return self._typeName
    
    def getDefaultDataSize(self):
        return self._defaultDataSize

    def getDefaultComputeSize(self):
        return self._defaultComputeSize

if __name__=='__main__':
    tp = TaskType(20, 15, 30, parameters.CODE_TASK_TYPE_VA)
    gp = group.createAGroup()

    serverList = gp.getServerList()
    taskList = []
    for s in serverList:
        for i in range(10):
            taskList.append(tp.createTask(s, i))
    for t in taskList:
        t.setDispatchedServer(serverList[-1])
        print("Task %s is generated in %s, data_size:%d, deadline:%d, type_name:%s, compute_time:%d." \
            % (t.getKey(), t.getAccessPoint().getKey(), t.getDataSize(), t.getDeadline(), \
            t.getTaskTypeName(), t.getComputeTime()))

