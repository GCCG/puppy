# For reconstructing Dedas's code
import sys
from . import net_ap
from . import task_generator
from . import scheduler
from . import parameters

class Dispatcher:
    def __init__(self, currentTime, taskGenerator):
        self._currentTime = currentTime
        # self._availableServerList = []
        self._scheduler = None
        # self._taskGenerator = taskGenerator

    def dispatch(self, newTask):
        pass

    def update(self, time):
        if time < self._currentTime:
            sys.exit("Something is wrong with your program, time is smaller than dispatcher's current time.")
        if time == self._currentTime:
            print("Time is equal to dispatcher's current time, no need to update.")
        else:
            self._currentTime = time
            self._scheduler.update(time)

    def _getAvailableServerList(self):
        return self._scheduler.getNetGraph().getServerList()

    def getCurrentACT(self):
        pass

    def getCurrentDS(self):
        pass


class DedasDispatcher(Dispatcher):
    def __init__(self, currentTime, currentScheduler, taskGenerator):
        Dispatcher.__init__(self, currentTime, taskGenerator)
        self._scheduler = currentScheduler

    def dispatch(self, newTask):
        targetServer = None
        serverList = self._getAvailableServerList()
        bestACT, bestDS = self._scheduler.getResult()
        for ser in serverList:
            newTask.setDispatchedServer(ser)
            ACT, DS = self._scheduler.schedulePlan(newTask, self._currentTime)
            if DS > bestDS or (DS == bestACT and ACT < bestACT):
                targetServer = ser
                bestACT = ACT
                bestDS = DS
        newTask.setDispatchedServer(targetServer)
        ACT, DS = self._scheduler.schedule(newTask, self._currentTime)
        if ACT != bestACT or DS != bestDS:
            sys.exit("Something is wrong with your scheduler.")


