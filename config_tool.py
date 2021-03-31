import numpy as np
import json
from .parameters import *
from .net_graph import NetGraph

class ModelManager:
    def __init__(self):
        self._tree_net_graph = None
        self._task_types = []
        self._seed = 0
        self._group_num = 0
        self._group_load_mean = 0
        self._group_load_sigma = 0
        self._backhaul_ban = 0
        self._ban_ratio = 0


        pass
    
    def add_group_type(self, ):
        pass

    def add_group_type(self, type_name, backhaul_ban, ban_ratio, default_server_num, rsc_mean, rsc_sigma):
        pass

    def set_group_list(self):
        pass


    def config_system(self):
        system_info_dict = {}
        pass

    def config_task(self):
        pass

    def check_load(self):
        # 统计来自任务和系统的两方面信息，来判断是否负载过重。
        # 计算负载，就以总计算量是否超过总计算能力来算；
        # 带宽负载，就以所有负载超过自身计算能力的域是否有足够的域间带宽将任务卸载出去
        pass

    def gen_tree_net(self, seed):
        np.random.seed(0)
        group_list = []
        group_type_list = []
        tng = NetGraph()
        for g in group_list:
            tng.add
            





