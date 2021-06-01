import numpy as np
import json
import sys
from .parameters import *
from .net_graph import NetGraph, TreeNetGraph
from .task_type import TaskType
from .group_type import GroupType

class ModelManager:
    """ 对ModelManager中所有内容的更改，必须使用
    ModelManager的方法进行，否则无法被history记录，也就无法重现
    模型 """
    def __init__(self, seed=0, history_file=None):
        # 模型字典
        self._tree_net_graphs = []

        # 任务的数据量、计算量信息，是一个字典
        self._task_types = []

        # 使用域类型的信息，来为TreeNetGraph创建域
        self._group_types = []
        self._backhaul_ban_ratios = {}

        # 设置随机种子，方便重现模型
        self._seed = seed
        np.random.seed(seed)

        # 以记录操作历史的方式，来记录模型
        # 通过重现操作历史来重新构造模型
        self._history = []
        if history_file == None:
            op_info = {}
            op_info['operation'] = 'init'
            op_info['args'] = {'seed': seed}
            self._history.append(op_info)
        elif type(history_file) == str:
            with open(history_file, 'r') as f:
                history = json.load(f)
            self._restore_from_history(history)
        else:
            raise ValueError("history should be a list")
        
    def add_group_type(self, type_name, backhaul_ban_ratio, default_server_num, rsc_list, ban_list, len_list, task_gen_info_dict):
        for gt in self._group_types:
            if gt.getGroupTypeName() == type_name:
                print("Group type with name %s already exists." % type_name)
                sys.exit()
        group_type = GroupType(default_server_num, type_name)
        for info in task_gen_info_dict:
            group_type.addTaskGenInfo(info['task_type_name'], info['mean'], variance=info['variance'])
        for rsc in rsc_list:
            group_type.expandComCapacityList(rsc)
        for ban in ban_list:
            group_type.expandBandwidthList(ban)
        for length in len_list:
            group_type.expandLengthList(length)

        self._group_types.append(group_type)

        self._backhaul_ban_ratios[type_name] = backhaul_ban_ratio
        op_info = {}
        op_info['operation'] = 'add_group_type'
        op_info['args'] = {'type_name':type_name, 'backhaul_ban_ratio':backhaul_ban_ratio, 'default_server_num':default_server_num,\
            'rsc_list':rsc_list, 'ban_list':ban_list, 'len_list':len_list, 'task_gen_info_dict':task_gen_info_dict}
        self._history.append(op_info)
        
    def add_task_type(self, type_name, data_size_mean, data_size_variance, com_size_mean, com_size_variance):
        for t in self._task_types:
            if t.getTaskTypeName() ==type_name:
                print("Task type with name %s already exist." % type_name)
        self._task_types.append(TaskType(data_size_mean, com_size_mean, 0, type_name, data_size_variance, com_size_variance))

        op_info ={}
        op_info['operation'] = 'add_task_type'
        op_info['args'] = {'type_name':type_name, 'data_size_mean':data_size_mean, 'data_size_variance':data_size_variance,\
            'com_size_mean':com_size_mean, 'com_size_variance':com_size_variance}
        self._history.append(op_info)

    def get_task_type(self, type_name):
        for t in self._task_types:
            if t.getTaskTypeName() == type_name:
                return t
        print("No task type with name %s" % type_name)
    
    def get_task_type_dict(self):
        tmp_dict = {}
        for t in self._task_types:
            tmp = {}
            tmp['data_size_mean'] = t.getDefaultDataSize()
            tmp['com_size_mean'] = t.getDefaultComputeSize()
            tmp['data_size_variance'] = t.getDataSizeVariance()
            tmp['com_size_variance'] = t.getComSizeVariance()
            tmp_dict[t.getTaskTypeName()] = tmp
        return tmp_dict


    def check_load(self):
        # 统计来自任务和系统的两方面信息，来判断是否负载过重。
        # 计算负载，就以总计算量是否超过总计算能力来算；
        # 带宽负载，就以所有负载超过自身计算能力的域是否有足够的域间带宽将任务卸载出去
        pass

    def gen_tree_net_graph(self, name, group_configs):
        tng = TreeNetGraph()
        for gt in self._group_types:
            tng.addGroupType(gt)
        root_switch = tng.getRootSwitch()
        for gc in group_configs:
            gt = self._get_group_type_by_name(gc['group_type_name'])
            if type(gt) == GroupType:
                if gc['server_num'] > 0:
                    server_num = gc['server_num']
                else:
                    server_num = gt.generateServerNum()
                if gc['backhaul_ban_ratio'] > 0:
                    backhaul_ban_ratio = gc['backhaul_ban_ratio']
                else:
                    backhaul_ban_ratio = self._backhaul_ban_ratios[gc['group_type_name']]
                ban_list = gt.getBandwidthList()
                
                group = tng.genGroup(root_switch, 0, 0, gc['group_type_name'], gc['server_num'] )
                l_list = tng.getLinksInGroup(group)
                total_ban = 0
                for l in l_list:
                    total_ban = total_ban + l.getBandwidth()
                backhaul_ban = int( backhaul_ban_ratio * total_ban/2)
                switch = tng.getSwitchInGroup(group)
                tng.reset_link_ban(root_switch, switch, backhaul_ban)
                tng.reset_link_ban(switch, root_switch, backhaul_ban)

        tng._floydShortestPath()
        tmp = {}
        tmp['name'] = name
        tmp['tng'] = tng
        self._tree_net_graphs.append(tmp)

        s_list = tng.getServerList()
        rsc_total = 0
        load_total = 0
        for s in  s_list:
            rsc_total = rsc_total + s.getRscAmount()
        


        op_info = {}
        op_info['operation'] = 'gen_tree_net_graph'
        op_info['args'] = {'name':name, 'group_configs':group_configs}
        self._history.append(op_info)
        return tng

    def _get_group_type_by_name(self, type_name):
        for gt in self._group_types:
            if gt.getGroupTypeName() == type_name:
                return gt
        return None

    def get_tree_net_graph(self,name):
        print(self._tree_net_graphs)
        for tng in self._tree_net_graphs:
            if tng['name'] == name:
                return tng['tng']
        raise ValueError("No tree net graph with name %s" % name)

    def get_tree_net_graphs(self):
        return self._tree_net_graphs

    def _restore_from_history(self, history):
        print("Restoring.....")
        for op_info in history:
            operation = op_info['operation']
            args = op_info['args']
            if operation == 'add_group_type':
                self.add_group_type(args['type_name'], args['backhaul_ban_ratio'], args['default_server_num'],\
                    args['rsc_list'], args['ban_list'], args['len_list'],args['task_gen_info_dict'])
            elif operation == 'add_task_type':
                self.add_task_type(args['type_name'], args['data_size_mean'], args['data_size_variance'], \
                    args['com_size_mean'], args['com_size_variance'])
            elif operation == 'gen_tree_net_graph':
                self.gen_tree_net_graph(args['name'], args['group_configs'])
            elif operation == 'init':
                self._seed = args['seed']
                np.random.seed(self._seed)

    def store(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self._history, f, sort_keys=True, indent=4, separators=(',', ': '))
    def get_task_type_list(self):
        return self._task_types

            
if __name__ == "__main__":
    model_manager = ModelManager(0)
    model_manager.add_task_type(type_name=CODE_TASK_TYPE_VA, data_size_mean=1, data_size_variance=1, com_size_mean=1, com_size_variance=1)
    model_manager.add_task_type(type_name=CODE_TASK_TYPE_VR, data_size_mean=1, data_size_variance=1, com_size_mean=1, com_size_variance=1)
    model_manager.add_task_type(type_name=CODE_TASK_TYPE_IoT, data_size_mean=1, data_size_variance=1, com_size_mean=1, com_size_variance=1)

    task_gen_info_dict = [{'task_type_name':CODE_TASK_TYPE_VA, 'mean':2, 'variance':5},\
        {'task_type_name':CODE_TASK_TYPE_VR, 'mean':2, 'variance':8},\
        {'task_type_name':CODE_TASK_TYPE_IoT, 'mean':2, 'variance':5}]
    model_manager.add_group_type(type_name=CODE_GROUP_TYPE_BUSINESS, backhaul_ban_ratio=1/3, default_server_num=3,\
        rsc_list=[8,5,4,8], ban_list=[15, 20], len_list=[13, 15, 20], task_gen_info_dict=task_gen_info_dict)


    task_gen_info_dict = [{'task_type_name':CODE_TASK_TYPE_VA, 'mean':1, 'variance':5},\
        {'task_type_name':CODE_TASK_TYPE_VR, 'mean':1, 'variance':8},\
        {'task_type_name':CODE_TASK_TYPE_IoT, 'mean':1, 'variance':5}]
    model_manager.add_group_type(type_name=CODE_GROUP_TYPE_COMMMUNITY, backhaul_ban_ratio=1/3, default_server_num=3,\
        rsc_list=[8,5,4,8], ban_list=[15, 20], len_list=[13, 15, 20], task_gen_info_dict=task_gen_info_dict)


    task_gen_info_dict = [{'task_type_name':CODE_TASK_TYPE_VA, 'mean':2, 'variance':5},\
        {'task_type_name':CODE_TASK_TYPE_VR, 'mean':2, 'variance':8},\
        {'task_type_name':CODE_TASK_TYPE_IoT, 'mean':2, 'variance':5}]
    model_manager.add_group_type(type_name=CODE_GROUP_TYPE_COMPANY, backhaul_ban_ratio=1/3, default_server_num=3,\
        rsc_list=[8,5,4,8], ban_list=[15, 20], len_list=[13, 15, 20], task_gen_info_dict=task_gen_info_dict)

    model_manager.gen_tree_net_graph('4-servers', [{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':2, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMMMUNITY, 'server_num':2, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('6-servers', [{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':4, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMMMUNITY, 'server_num':2, 'backhaul_ban_ratio':1/2}])

    tng_1 = model_manager.get_tree_net_graph('4-servers')
    print("4-servers:")
    tng_1.print_info()
    tng_2 = model_manager.get_tree_net_graph('6-servers')
    print("6-servers:")
    tng_2.print_info()

    model_manager.store('./puppy/config/test_config.json')

    print("----------Restore model manager")
    model_manager_2 = ModelManager(history_file='./puppy/config/test_config.json')

    tngs = model_manager_2.get_tree_net_graphs()
    for tng in tngs:
        print("tng %s, info is:" % tng['name'])
        tng['tng'].print_info()



