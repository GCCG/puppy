# Class Slot
import sys
from . import parameters
from . import task
from . import group

class Slot:
    def __init__(self, rscType, rscAmount):
        if rscType != parameters.CODE_RSC_LINK and rscType != parameters.CODE_RSC_SERVER:
            sys.exit("rscTpye should be %s or %s." % (parameters.CODE_RSC_LINK, parameters.CODE_RSC_SERVER))
        self._rscType = rscType
        self._remRscAmount = rscAmount
        self._rscAmount = rscAmount
        self._tasks = []
        self._occupiedRsc = []

    def remainedRsc(self):
        return self._remRscAmount

    def addTask(self, task, occupiedRsc):
        if occupiedRsc <=0:
            sys.exit("In slot.py, The value of occupiedRsc should be positive.")
        elif self._rscType == parameters.CODE_RSC_SERVER:
            self._remRscAmount = 0
            self._tasks.append(task)
            self._occupiedRsc.append(self._rscAmount)
        elif self._remRscAmount - occupiedRsc >= 0:
            self._remRscAmount = self._remRscAmount - occupiedRsc
            self._tasks.append(task)
            self._occupiedRsc.append(occupiedRsc)
        else:
            sys.exit("In slot.py, occupiedRsc is more than remained")

    def deleteTask(self, dumpedTask):
        for i in range(len(self._tasks)):
            if self._tasks[i] == dumpedTask:
                del self._tasks[i]
                self._remRscAmount = self._remRscAmount + self._occupiedRsc[i]
                del self._occupiedRsc[i]
                return True
        return False
    
    def free(self):
        self._remRscAmount = self._rscAmount
        self._tasks = []
        self._occupiedRsc = []

    def getTasks(self):
        return list(self._tasks)
    
    def getStatus(self):
        status = ""
        for t in self._tasks:
            status = status + "." + t.getKey()
        return status

if __name__ == "__main__":
    st = Slot(parameters.CODE_RSC_LINK, 10)
    taskList = []
    gp = group.createAGroup()
    serverList = gp.getServerList()
    for i in range(10):
        taskList.append(task.Task(parameters.CODE_TASK_TYPE_IoT, serverList[0], 20+i, i, 5, 10))
    print("-------Adding phase:")
    for t in taskList:
        st.addTask(t, 1)
        print("In slot, task list:")
        for tt in st.getTasks():
            print(tt.getKey())
        print("Remained rsc is: %d" % (st.remainedRsc()))
    print("-------Deleting phase:")
    for t in taskList:
        st.deleteTask(t)
        print("In slot, task list:")
        for tt in st.getTasks():
            print(tt.getKey())
        print("Remained rsc is: %d" % (st.remainedRsc()))
    


    
