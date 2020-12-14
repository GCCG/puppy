# Class Slot
from . import parameters

class Slot:
    def __init__(self, rscType, rscAmount):
        self._rscType = rscType
        self._remRscAmount = rscAmount
        self._rscAmount = rscAmount
        self._tasks = []
        self._occupiedRsc = []

    def remainedRsc(self):
        return self._rscAmount

    def addTask(self, task, occupiedRsc):
        if occupiedRsc <=0:
            Exception("The value of occupiedRsc should be positive.")
        elif self._rscType == parameters.CODE_RSC_SERVER:
            self._remRscAmount = 0
            self._tasks.append(task)
            self._occupiedRsc.append(self._rscAmount)
        else:
            self._remRscAmount = self._remRscAmount - occupiedRsc
            self._tasks.append(task)
            self._occupiedRsc.append(occupiedRsc)

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
    
