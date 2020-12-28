# For reconstructing Dedas's code
import sys
from . import net_ap
from . import task_generator
from . import scheduler
from . import parameters
from . import net_graph
from . import group_trans

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
    def __init__(self, currentTime, currentScheduler):
        Dispatcher.__init__(self, currentTime)
        self._scheduler = currentScheduler

    def dispatch(self, newTask):
        targetServer = None
        serverList = self._getAvailableServerList()
        bestACT = self._scheduler.getCurrentACT()
        bestDS = self._scheduler.getCurrentDS()
        for ser in serverList:
            newTask.setDispatchedServer(ser)
            ACT, DS = self._scheduler.schedulePlan(newTask, self._currentTime)
            if DS > bestDS or (DS == bestDS and ACT < bestACT):
                targetServer = ser
                bestACT = ACT
                bestDS = DS
        newTask.setDispatchedServer(targetServer)
        ACT, DS = self._scheduler.schedule(newTask, self._currentTime)
        if ACT != bestACT or DS != bestDS:
            sys.exit("Something is wrong with your scheduler.")

if __name__ == "__main__":
    ng = net_graph.createATreeGraph()
    ds = scheduler.DedasScheduler(ng, 50, 0)
    dd = DedasDispatcher(0,ds)

    tg = task_generator.TaskGenerator()
    gTrans = group_trans.createAGroupTrans(ng.getGroupList())
    tg.addUserType(parameters.CODE_USER_TYPE_OTAKU, gTrans)
    tg.addUserType(parameters.CODE_USER_TYPE_RICH_MAN, gTrans)
    tg.addUserType(parameters.CODE_USER_TYPE_SALARY_MAN, gTrans)

    for g in ng.getGroupList():
        tg.generateUsers(g, 3, parameters.CODE_USER_TYPE_OTAKU)
        tg.generateUsers(g, 2, parameters.CODE_USER_TYPE_SALARY_MAN)
        tg.generateUsers(g, 2, parameters.CODE_USER_TYPE_RICH_MAN)
    
    taskList = tg.generateTasks(0)
    for t  in taskList:
        print("%s, access_point:%s, type_name:" %(t.getKey(), t.getAccessPoint().getKey(), ))
