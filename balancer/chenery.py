""" 用来进行对比实验的代码 """

from ..net_graph import NetGraph, createATreeGraph, TreeNetGraph
from ..net_ap import NetAP
from ..net_link import NetLink
from ..parameters import *
import numpy as np
from scipy.optimize import minimize
import sys
from matplotlib import pyplot as plt
import json
from .wolf import check_constraints, test_normal_iteration, test_partial_iteration, compare_numerical_method_performance
import math
from ..config_tool import ModelManager
import time
from .labrador import test_hierarchy
from .multi_task import test_multi
# import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon


def create_model_manager(seed=0):
    """ model_manager = ModelManager()
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
        rsc_list=[2,5,4,3], ban_list=[15, 20], len_list=[13, 15, 20], task_gen_info_dict=task_gen_info_dict)


    task_gen_info_dict = [{'task_type_name':CODE_TASK_TYPE_VA, 'mean':2, 'variance':5},\
        {'task_type_name':CODE_TASK_TYPE_VR, 'mean':2, 'variance':8},\
        {'task_type_name':CODE_TASK_TYPE_IoT, 'mean':2, 'variance':5}]
    model_manager.add_group_type(type_name=CODE_GROUP_TYPE_COMPANY, backhaul_ban_ratio=1/3, default_server_num=3,\
        rsc_list=[4,2,1,8], ban_list=[20, 20], len_list=[13, 15, 20], task_gen_info_dict=task_gen_info_dict)

    model_manager.gen_tree_net_graph('2_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':2, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('3_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':3, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('4_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':4, 'backhaul_ban_ratio':1/2}])
    
    model_manager.gen_tree_net_graph('5_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':5, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('6_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':6, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('7_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':7, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('8_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':8, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('9_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':9, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('10_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':10, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('11_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':11, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('12_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':12, 'backhaul_ban_ratio':1/2}]) """
    np.random.seed(seed)
    model_manager = ModelManager()
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
        rsc_list=[2,5,4,3], ban_list=[15, 20], len_list=[13, 15, 20], task_gen_info_dict=task_gen_info_dict)


    task_gen_info_dict = [{'task_type_name':CODE_TASK_TYPE_VA, 'mean':2, 'variance':5},\
        {'task_type_name':CODE_TASK_TYPE_VR, 'mean':2, 'variance':8},\
        {'task_type_name':CODE_TASK_TYPE_IoT, 'mean':2, 'variance':5}]
    model_manager.add_group_type(type_name=CODE_GROUP_TYPE_COMPANY, backhaul_ban_ratio=1/3, default_server_num=3,\
        rsc_list=[4,2,1,8], ban_list=[20, 20], len_list=[13, 15, 20], task_gen_info_dict=task_gen_info_dict)

    model_manager.gen_tree_net_graph('2_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':2, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('3_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':2, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':1, 'backhaul_ban_ratio':1}])

    model_manager.gen_tree_net_graph('4_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':2, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':2, 'backhaul_ban_ratio':1/2}])
    
    model_manager.gen_tree_net_graph('5_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':3, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':2, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('6_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':3, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':3, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('7_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':4, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':3, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('8_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':4, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':4, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('9_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':5, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':4, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('10_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':5, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':5, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('11_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':6, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':5, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('12_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':6, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':6, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('13_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':6, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':7, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('14_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':7, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':7, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('15_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':7, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':8, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('16_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':8, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':8, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('17_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':8, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':9, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('18_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':9, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':9, 'backhaul_ban_ratio':1/2}])

    model_manager.gen_tree_net_graph('19_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':10, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':9, 'backhaul_ban_ratio':1/2}])
    
    model_manager.gen_tree_net_graph('20_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':10, 'backhaul_ban_ratio':1/2}, \
        {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':10, 'backhaul_ban_ratio':1/2}])

    for tng in model_manager.get_tree_net_graphs():
        print("name: %s, info:" % tng['name'])
        tng['tng'].print_info()
    
    return model_manager

def create_model_manager_v2(backhaul_ban_ratio, load_para, seed, data_size_dict, com_size_dict):
    np.random.seed(seed)
    model_manager = ModelManager()
    
    model_manager.add_task_type(type_name=CODE_TASK_TYPE_VR, data_size_mean=data_size_dict[CODE_TASK_TYPE_VR]*load_para, data_size_variance=1, com_size_mean=com_size_dict[CODE_TASK_TYPE_VR]*load_para, com_size_variance=1)
    model_manager.add_task_type(type_name=CODE_TASK_TYPE_VA, data_size_mean=data_size_dict[CODE_TASK_TYPE_VA]*load_para, data_size_variance=1, com_size_mean=com_size_dict[CODE_TASK_TYPE_VA]*load_para, com_size_variance=1)
    model_manager.add_task_type(type_name=CODE_TASK_TYPE_IoT, data_size_mean=data_size_dict[CODE_TASK_TYPE_IoT]*load_para, data_size_variance=1, com_size_mean=com_size_dict[CODE_TASK_TYPE_IoT]*load_para, com_size_variance=1)

    task_gen_info_dict = [{'task_type_name':CODE_TASK_TYPE_VA, 'mean':0, 'variance':5},\
        {'task_type_name':CODE_TASK_TYPE_VR, 'mean':0, 'variance':8},\
        {'task_type_name':CODE_TASK_TYPE_IoT, 'mean':0, 'variance':5}]
    model_manager.add_group_type(type_name=CODE_GROUP_TYPE_BUSINESS, backhaul_ban_ratio=1/3, default_server_num=3,\
        rsc_list=[8,8,8,8], ban_list=[20, 20], len_list=[13, 15, 20], task_gen_info_dict=task_gen_info_dict)


    task_gen_info_dict = [{'task_type_name':CODE_TASK_TYPE_VA, 'mean':3, 'variance':5},\
        {'task_type_name':CODE_TASK_TYPE_VR, 'mean':3, 'variance':5},\
        {'task_type_name':CODE_TASK_TYPE_IoT, 'mean':3, 'variance':5}]
    model_manager.add_group_type(type_name=CODE_GROUP_TYPE_COMMMUNITY, backhaul_ban_ratio=1/3, default_server_num=3,\
        rsc_list=[8,3,3,3], ban_list=[20, 20], len_list=[13, 15, 20], task_gen_info_dict=task_gen_info_dict)


    task_gen_info_dict = [{'task_type_name':CODE_TASK_TYPE_VA, 'mean':2, 'variance':5},\
        {'task_type_name':CODE_TASK_TYPE_VR, 'mean':2, 'variance':8},\
        {'task_type_name':CODE_TASK_TYPE_IoT, 'mean':2, 'variance':5}]
    model_manager.add_group_type(type_name=CODE_GROUP_TYPE_COMPANY, backhaul_ban_ratio=1/3, default_server_num=3,\
        rsc_list=[4,2,1,8], ban_list=[20, 20], len_list=[13, 15, 20], task_gen_info_dict=task_gen_info_dict)

    # model_manager.gen_tree_net_graph('2_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':2, 'backhaul_ban_ratio':backhaul_ban_ratio}])

    # model_manager.gen_tree_net_graph('3_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':2, 'backhaul_ban_ratio':backhaul_ban_ratio}, \
    #     {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':1, 'backhaul_ban_ratio':1}])

    model_manager.gen_tree_net_graph('4_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_BUSINESS, 'server_num':2, 'backhaul_ban_ratio':backhaul_ban_ratio}, \
        {'group_type_name':CODE_GROUP_TYPE_COMMMUNITY, 'server_num':2, 'backhaul_ban_ratio':backhaul_ban_ratio}])
    
    # model_manager.gen_tree_net_graph('5_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':3, 'backhaul_ban_ratio':backhaul_ban_ratio}, \
    #     {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':2, 'backhaul_ban_ratio':backhaul_ban_ratio}])

    # model_manager.gen_tree_net_graph('6_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_BUSINESS, 'server_num':3, 'backhaul_ban_ratio':backhaul_ban_ratio}, \
    #     {'group_type_name':CODE_GROUP_TYPE_COMMMUNITY, 'server_num':3, 'backhaul_ban_ratio':backhaul_ban_ratio}])

    # model_manager.gen_tree_net_graph('7_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':4, 'backhaul_ban_ratio':backhaul_ban_ratio}, \
    #     {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':3, 'backhaul_ban_ratio':backhaul_ban_ratio}])

    # model_manager.gen_tree_net_graph('8_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_BUSINESS, 'server_num':4, 'backhaul_ban_ratio':backhaul_ban_ratio}, \
    #     {'group_type_name':CODE_GROUP_TYPE_COMMMUNITY, 'server_num':4, 'backhaul_ban_ratio':backhaul_ban_ratio}])

    # model_manager.gen_tree_net_graph('9_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':5, 'backhaul_ban_ratio':backhaul_ban_ratio}, \
    #     {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':4, 'backhaul_ban_ratio':backhaul_ban_ratio}])

    # model_manager.gen_tree_net_graph('10_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_BUSINESS, 'server_num':5, 'backhaul_ban_ratio':backhaul_ban_ratio}, \
    #     {'group_type_name':CODE_GROUP_TYPE_COMMMUNITY, 'server_num':5, 'backhaul_ban_ratio':backhaul_ban_ratio}])

    # model_manager.gen_tree_net_graph('11_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':6, 'backhaul_ban_ratio':backhaul_ban_ratio}, \
    #     {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':5, 'backhaul_ban_ratio':backhaul_ban_ratio}])

    # model_manager.gen_tree_net_graph('12_servers', group_configs=[{'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':6, 'backhaul_ban_ratio':backhaul_ban_ratio}, \
    #     {'group_type_name':CODE_GROUP_TYPE_COMPANY, 'server_num':6, 'backhaul_ban_ratio':backhaul_ban_ratio}])

    # for tng in model_manager.get_tree_net_graphs():
    #     print("name: %s, info:" % tng['name'])
    #     tng['tng'].print_info()
    task_type_dict = model_manager.get_task_type_dict()
    BETA_H = []
    BETA_D = []
    TASK_TYPE_LIST = []
    for k in task_type_dict.keys():
        TASK_TYPE_LIST.append(k)
        BETA_D.append(task_type_dict[k]['data_size_mean'])
        BETA_H.append(task_type_dict[k]['com_size_mean'])
    print(BETA_D,BETA_H, TASK_TYPE_LIST)
    return model_manager

def test_backhaul_ban_influence(seed, repeat, inner_repeat, data_size_list):
    result_folder = './puppy/results/backhaul_ban_ratio_influence/'
  
    # 首先，把模型全部生成出来
    load_para = 0.3
    model_managers = []
    ratio_num = 11
    x = np.zeros(ratio_num)

    data_size_dict = {CODE_TASK_TYPE_VR:data_size_list[0], CODE_TASK_TYPE_VA:data_size_list[1], CODE_TASK_TYPE_IoT:data_size_list[2]}
    come_size_dict = {CODE_TASK_TYPE_VR:2, CODE_TASK_TYPE_VA:2, CODE_TASK_TYPE_IoT:2}

    file_prefix = 'ratio-%.2f_%.2f_%.2f-' % (data_size_dict[CODE_TASK_TYPE_VR]/come_size_dict[CODE_TASK_TYPE_VR], \
            data_size_dict[CODE_TASK_TYPE_VA]/come_size_dict[CODE_TASK_TYPE_VA], \
            data_size_dict[CODE_TASK_TYPE_IoT]/come_size_dict[CODE_TASK_TYPE_IoT])
    
    for i in range(ratio_num):
        inverse = 1 + i/2.5 # 从1到5，共十组
        x[i] = inverse
        backhaul_ban_ratio = 1/inverse
        model_managers.append(create_model_manager_v2(backhaul_ban_ratio, load_para=load_para, seed=seed, data_size_dict=data_size_dict, com_size_dict=come_size_dict))
    np.savetxt(result_folder + file_prefix + '%d_servers-seed_%d-ratio_list.txt' % \
        (len(model_managers[0].get_tree_net_graphs()[0]['tng'].getServerList()), seed), 1/x, fmt = '%f')
    
    # np.random.seed(seed)
    for u in range(repeat):
        poops =  np.zeros((ratio_num, 6))
        np.random.seed(u)
        for i in range(ratio_num):
            task_type_dict = model_managers[i].get_task_type_dict()
            BETA_H = []
            BETA_D = []
            TASK_TYPE_LIST = []
            for k in task_type_dict.keys():
                TASK_TYPE_LIST.append(k)
                BETA_D.append(task_type_dict[k]['data_size_mean'])
                BETA_H.append(task_type_dict[k]['com_size_mean'])
            # print(1/x[i])
            print("-------------------------backhaul_ban_ratio %f----------------------------" % (1/x[i]))
            # print(BETA_D,BETA_H, TASK_TYPE_LIST)

            tng = model_managers[i].get_tree_net_graphs()[0]
            result = test_multi(tng['tng'], beta_d=BETA_D, beta_h=BETA_H, task_type_list=TASK_TYPE_LIST, repeat=inner_repeat)
            poops[i,:] = np.reshape(result, result.shape[0]*result.shape[1])
        np.savetxt(result_folder + file_prefix + '%d_servers-seed_%d-innerrepeat_%d.txt' % (len(tng['tng'].getServerList()), u, inner_repeat), poops, fmt = '%f')

def test_data_com_ratio_influence(repeat, seed, inner_repeat):
    result_folder = './puppy/results/data_com_ratio_influence/'
  
    # 首先，把模型全部生成出来
    load_para = 0.3
    model_managers = []
    ratio_num = 4

   
    backhaul_ban_ratio = 1/3
    for i in range(ratio_num):
        data_size_dict = {CODE_TASK_TYPE_VR:4+i*0.5, CODE_TASK_TYPE_VA:4, CODE_TASK_TYPE_IoT:4-i*0.5}
        come_size_dict = {CODE_TASK_TYPE_VR:2, CODE_TASK_TYPE_VA:2, CODE_TASK_TYPE_IoT:2}
        file_prefix = 'ratio-%.2f_%.2f_%.2f-' % (data_size_dict[CODE_TASK_TYPE_VR]/come_size_dict[CODE_TASK_TYPE_VR], \
            data_size_dict[CODE_TASK_TYPE_VA]/come_size_dict[CODE_TASK_TYPE_VA], \
            data_size_dict[CODE_TASK_TYPE_IoT]/come_size_dict[CODE_TASK_TYPE_IoT])
        model_manager = create_model_manager_v2(backhaul_ban_ratio, load_para=load_para, seed=seed, data_size_dict=data_size_dict, com_size_dict=come_size_dict)
        # model_managers.append(model_manager)
        poops = np.zeros((repeat, 6))
        for j in range(repeat):
            task_type_dict = model_manager.get_task_type_dict()
            BETA_H = []
            BETA_D = []
            TASK_TYPE_LIST = []
            for k in task_type_dict.keys():
                TASK_TYPE_LIST.append(k)
                BETA_D.append(task_type_dict[k]['data_size_mean'])
                BETA_H.append(task_type_dict[k]['com_size_mean'])
            tng = model_manager.get_tree_net_graphs()[0]
            result = test_multi(tng['tng'], beta_d=BETA_D, beta_h=BETA_H, task_type_list=TASK_TYPE_LIST, repeat=inner_repeat)
            poops[j,:] = np.reshape(result, result.shape[0]*result.shape[1])
        np.savetxt(result_folder + file_prefix + '%d_servers-seed_%d-innerrepeat_%d.txt' % (len(tng['tng'].getServerList()), i, inner_repeat), poops, fmt = '%f')

def test_numerical_method_time_consumption(seed, repeat=5, gen_feasible='penalty', file_prefix='penalty-'):
    np.random.seed(seed)
    model_manager = create_model_manager()
    
    for tng in model_manager.get_tree_net_graphs():
        print("name: %s, info:" % tng['name'])
        tng['tng'].print_info()

    result_folder = './puppy/results/time_consumption/'
    config_folder = './puppy.config/'
    config_file = config_folder + 'time_consumption.json'
    # performance_result_folder = './puppy/results/performance/'

    tng_list = model_manager.get_tree_net_graphs()


    model_num = len(tng_list)
    time_matrix_for_normal = np.zeros((model_num, repeat))
    
    for u in range(model_num):
        print(tng_list[u]['name'])
        for v in range(repeat):
            start_time = time.time()
            test_normal_iteration(tng_list[u]['tng'], 200, 1, gen_feasible=gen_feasible)
            end_time = time.time()
            time_matrix_for_normal[u, v] = end_time -start_time
            print("-------------repeat %d------------------" % v)
        # np.savetxt(performance_result_folder + p_file_prefix + 'normal_result.txt', performance_result, fmt = '%f')
    print("time consumption for normal")
    print(time_matrix_for_normal[0:model_num,:])
    np.savetxt(result_folder + '%sseed_%dnormal.txt' % file_prefix, time_matrix_for_normal, fmt = '%f')

    time_matrix_for_partial = np.zeros((model_num, repeat))
    # print("test for %s" % tng_list[0]['name'])
    for u in range(model_num):
        print(tng_list[u]['name'])
        for v in range(repeat):
            start_time = time.time()
            test_partial_iteration(tng_list[u]['tng'], 200, 1, gen_feasible=gen_feasible)
            end_time = time.time()
            time_matrix_for_partial[u, v] = end_time -start_time
            print("-------------repeat %d------------------" % v)
    print("time consumption for partial")
    print(time_matrix_for_partial[0:model_num,:])
    np.savetxt(result_folder + '%sseed_%d-partial.txt' % (file_prefix, seed), time_matrix_for_partial, fmt = '%f')

def test_hierarchy_method_time_consumption(seed, repeat=5, gen_feasible='penalty', file_prefix='penalty-'):
    np.random.seed(seed)
    model_manager = create_model_manager()

    result_folder = './puppy/results/time_consumption/'
    config_folder = './puppy.config/'
    config_file = config_folder + 'time_consumption.json'

    tng_list = model_manager.get_tree_net_graphs()


    model_num = len(tng_list)
    time_matrix_for_normal = np.zeros((model_num, repeat))
    
    for u in range(model_num):
        print(tng_list[u]['name'])
        for v in range(repeat):
            start_time = time.time()
            # test_normal_iteration(tng_list[u]['tng'], 200, 1, gen_feasible=gen_feasible)
            test_hierarchy(tng_list[u]['tng'], max_iteration=30, repeat=1)
            
            end_time = time.time()
            time_matrix_for_normal[u, v] = end_time -start_time
            print("-------------repeat %d, %s, time is %d------------------" % (v, tng_list[u]['name'], end_time-start_time))
    print("time consumption for hierarchy")
    print(time_matrix_for_normal[0:model_num,:])
    np.savetxt(result_folder + '%sseed_%d-hierarchy.txt' % (file_prefix, seed), time_matrix_for_normal, fmt = '%f')
    # tng = tng_list[10]
    # for v in range(repeat):
    #     start_time = time.time()
    #     # test_normal_iteration(tng_list[u]['tng'], 200, 1, gen_feasible=gen_feasible)
    #     test_hierarchy(tng['tng'], max_iteration=30, repeat=1)
            
    #     end_time = time.time()
    #     time_matrix_for_normal[8, v] = end_time -start_time
    #     print("-------------repeat %d, %s, time is %d------------------" % (v, tng['name'], end_time-start_time))

def test_numerical_method_performance(seed, repeat, gen_feasible='linear', need_timing=False):
    np.random.seed(seed)
    model_manager = create_model_manager()

    model_num = len(model_manager.get_tree_net_graphs())

    result_folder = './puppy/results/performance/'
    config_folder = './puppy.config/'
    config_file = config_folder + 'time_consumption.json'
    time_consumption_folder = './puppy/results/time_consumption/'
    tc_file_prefix = time_consumption_folder + "%d_models-%s-seed_%d-" % (model_num, gen_feasible, seed)

    tc_slsqp = np.zeros((model_num, repeat))
    tc_pvi = np.zeros((model_num, repeat))

    try:
        tc_s = np.loadtxt( tc_file_prefix + 'normal.txt')
        tc_p = np.loadtxt( tc_file_prefix + 'partial.txt')
        tc_slsqp[0:tc_s.shape[0],:] = tc_s
        print(tc_slsqp)
        tc_pvi[0:tc_p.shape[0],:] = tc_p
        print(tc_pvi)

        finished_num = tc_s.shape[0]
    except:
        finished_num = 0

    index = 0
    for tng in model_manager.get_tree_net_graphs():
        print("------------%s--------------" % tng['name'])
        if index < finished_num:
            print("Already finished.")
            index = index + 1
            continue
        time_consumption_dict = compare_numerical_method_performance(tng['tng'],max_iteration=200, repeat=repeat, seed=seed, file_prefix=result_folder+tng['name']+'-'+'seed_'+str(seed)+'-', need_timing=need_timing)
        # break
        if need_timing == True:
            tc_slsqp[index,:] = np.array(time_consumption_dict['slsqp'])
            tc_pvi[index,:] = np.array(time_consumption_dict['pvi'])
            np.savetxt( tc_file_prefix + 'normal.txt', tc_slsqp[0:index+1,:], fmt = '%f')
            np.savetxt( tc_file_prefix + 'partial.txt', tc_pvi[0:index+1,:], fmt = '%f')
        index = index + 1

def test_hierarchy_method_performance(seed, repeat, need_timing=False):
    np.random.seed(seed)
    model_manager = create_model_manager()
    result_folder = './puppy/results/performance/'
    config_folder = './puppy.config/'
    config_file = config_folder + 'time_consumption.json'
    time_consumption_folder = './puppy/results/time_consumption/'
    model_num = len(model_manager.get_tree_net_graphs())
    tc_file_prefix = time_consumption_folder + "%d_models-%s-seed_%d-" % (model_num, "linear", seed)

    tc_hier = np.zeros((model_num, repeat))
    try:
        tc_h = np.loadtxt( tc_file_prefix + 'hierarchy.txt')
        tc_hier[0:tc_s.shape[0],:] = tc_h
        print(tc_hier)
        finished_num = tc_h.shape[0]
    except:
        finished_num = 0

    index = 0
    for tng in model_manager.get_tree_net_graphs():
        print("------------%s--------------" % tng['name'])
        if index < finished_num:
            print("Already finished.")
            index = index + 1
            continue
        results, tc = test_hierarchy(tng['tng'], repeat = repeat, need_timing=True)
        file_prefix = tng['name']+'-'+'seed_'+str(seed)+'-'
        np.savetxt(result_folder + '%shierarchy_result.txt' % file_prefix, results, fmt = '%f')

        if need_timing == True:
            tc_hier[index,:] = np.array(tc)
            np.savetxt( tc_file_prefix + 'hierarchy.txt', tc_hier[0:index+1,:], fmt = '%f')
        index = index + 1
    # tng = model_manager.get_tree_net_graphs()[8]
    # results = test_hierarchy(tng['tng'], repeat = 1)
    # file_prefix = tng['name']+'-'+'seed_'+str(seed)+'-'
    # np.savetxt(result_folder + '%shierarchy_result.txt' % file_prefix, results, fmt = '%f')

def test_time_vs_performance(seed, file_prefix, method='normal', gen_feasible='linear', max_iteration=200,repeat=5):
    #既要记录耗时，还要记录它的目标函数值。
    np.random.seed(seed)
    model_manager = create_model_manager()

    result_folder = './puppy/results/time_vs_performance/'

    tng_list = model_manager.get_tree_net_graphs()


    model_num = len(tng_list)
    time_matrix = np.zeros((model_num, repeat))
    if method == 'partial':
        solver = test_partial_iteration
    elif method == 'normal':
        solver = test_normal_iteration
    elif method == 'hierarchy':
        solver = test_hierarchy
    
    performance_matrix = np.zeros((model_num, repeat))
    for u in range(model_num):
        print(tng_list[u]['name'])
        for v in range(repeat):
            start_time = time.time()
            # test_normal_iteration(tng_list[u]['tng'], 200, 1, gen_feasible=gen_feasible)
            result = solver(tng_list[u]['tng'], max_iteration=max_iteration, repeat=1, gen_feasible=gen_feasible, return_result=True)
            
            
            end_time = time.time()
            print(result)
            performance_matrix[u,v] = result[0]
            time_matrix[u, v] = end_time -start_time
            print("-------------repeat %d, %s, time is %d------------------" % (v, tng_list[u]['name'], end_time-start_time))
    performance_matrix = np.array(performance_matrix)
    print("time consumption for %s" % method)
    print(time_matrix[0:model_num,:])
    print("performance for %s" % method)
    print(performance_matrix)
    np.savetxt(result_folder + '%sseed_%d-%s.txt' % (file_prefix, seed, method), time_matrix, fmt = '%f')
    np.savetxt(result_folder + '%sseed_%d-%s_result.txt' % (file_prefix, seed, method), performance_matrix, fmt = '%f')
    # pass


def draw_comparing_time_consumption(seed, model_num, repeat, gen_feasible):
    result_folder = './puppy/results/time_consumption/'
    config_folder = './puppy.config/'

    # test_numerical_method_time_consumption()
    # y1 = np.loadtxt(result_folder+'linear-normal.txt')
    # y2 = np.loadtxt(result_folder+'linear-partial.txt')
    y1 = np.loadtxt(result_folder+'%d_models-%s-seed_%d-normal.txt' % (model_num, gen_feasible, seed))
    y2 = np.loadtxt(result_folder+'%d_models-%s-seed_%d-partial.txt' % (model_num, gen_feasible, seed))
    # y3 = np.loadtxt(result_folder+'%d_models-%s-seed_%d-hierarchy.txt' % (model_num, gen_feasible, seed))

    para_normal = linear_regression(seed, model_num, 'normal')
    para_partial = linear_regression(seed, model_num, 'partial')

    model_num = y1.shape[0]

    y1 = np.array(y1)*1000
    y2 = np.array(y2)*1000
    # y3 = np.array(y3)*1000

    x = np.ones(y1.shape)
    y_normal = []
    y_partial = []
    # y_hierarchy = []
    x_l = []
    for i in range(model_num):
        y_normal.append(np.average(y1[i,:]))
        y_partial.append(np.average(y2[i,:]))
        # y_hierarchy.append(np.average(y3[i,:]))
        x_l.append(i+2)

    for i in range(model_num):
        x[i,:] = x[i,:] + i + 1
    
    y1 = np.reshape(y1, (model_num*repeat,1))
    y2 = np.reshape(y2, (model_num*repeat,1))
    # y3 = np.reshape(y3, (model_num*repeat,1))
    
    x = np.reshape(x, (model_num*repeat,1))
    # print(x)
    # print(y1)
    plt.title("Time consumption of numerical methods")
    plt.xlabel('Server amount')
    plt.ylabel('Time (ms)')
    
    plt.scatter(x, y1, alpha=0.6, color='blue', marker='x')
    plt.scatter(x, y2, alpha=0.3, color='green', marker='s')
    # plt.scatter(x, y3, alpha=0.6, color='red', marker='+')

    # plt.plot(x_l, y_normal, color='blue', linestyle='--', label='slsqp', marker='x')
    # plt.plot(x_l, y_partial, color='green',linestyle=':', label='pvi', marker='s')
    # plt.plot(x_l, y_hierarchy, color='red', linestyle='dashdot', label='hier', marker='+')

    x_l = np.array(x_l)

    y_normal = np.exp(para_normal[0]*x_l + para_normal[1])
    y_partial = np.exp(para_partial[0]*x_l + para_partial[1])
    plt.plot(x_l, y_normal, color='blue', linestyle='--', label='slsqp', marker='x')
    plt.plot(x_l, y_partial, color='green',linestyle=':', label='pvi', marker='s')
    # plt.plot(x_l, y_hierarchy, color='red', linestyle='dashdot', label='hier', marker='+')

    plt.legend(loc=2)
    plt.yscale('log')
    plt.show()

def draw_comparing_time_consumption_comb_version():
    model_num = 11
    repeat = 5
    result_folder = './puppy/results/time_consumption/'
    config_folder = './puppy.config/'

    file_prefix = '11_models-linear-'
   
    y1 = np.loadtxt(result_folder+'%snormal.txt' % file_prefix)
    y2 = np.loadtxt(result_folder+'%spartial.txt' % file_prefix)
    # y3 = np.loadtxt(result_folder+'%shierarchy.txt' % file_prefix)

    y1 = np.array(y1)*1000
    y2 = np.array(y2)*1000
    # y3 = np.array(y3)*1000

    x = np.ones(y1.shape)
    y_normal = []
    y_partial = []
    y_hierarchy = []
    x_l = []
    for i in range(model_num):
        y_normal.append(np.average(y1[i,:]))
        y_partial.append(np.average(y2[i,:]))
        # y_hierarchy.append(np.average(y3[i,:]))
        x_l.append(i+2)

    for i in range(model_num):
        x[i,:] = x[i,:] + i + 1
    
    y1 = np.reshape(y1, (model_num*repeat,1))
    y2 = np.reshape(y2, (model_num*repeat,1))
    # y3 = np.reshape(y3, (model_num*repeat,1))
    
    x = np.reshape(x, (model_num*repeat,1))
    # print(x)
    # print(y1)
    
    # plt.xlabel('Server amount')
    # plt.ylabel('Time (ms)')
    plt.figure(figsize=(8, 4))
    
    plt.subplot(121)
    plt.title("Linear function as objective")
    plt.xlabel('Server amount')
    plt.ylabel('Time (ms)')
    plt.scatter(x, y1, alpha=0.6, color='blue', marker='x')
    plt.scatter(x, y2, alpha=0.3, color='deeppink', marker='s')
    # plt.scatter(x, y3, alpha=0.6, color='red', marker='+')

    plt.plot(x_l, y_normal, color='blue', linestyle='--', label='slsqp', marker='x')
    plt.plot(x_l, y_partial, color='deeppink',linestyle=':', label='pvi', marker='s')
    # plt.plot(x_l, y_hierarchy, color='red', linestyle='dashdot', label='hier', marker='+')
    plt.legend(loc=2)
    plt.yscale('log')

    file_prefix = '11_models-com-'
   
    y1 = np.loadtxt(result_folder+'%snormal.txt' % file_prefix)
    y2 = np.loadtxt(result_folder+'%spartial.txt' % file_prefix)
    # y3 = np.loadtxt(result_folder+'%shierarchy.txt' % file_prefix)

    y1 = np.array(y1)*1000
    y2 = np.array(y2)*1000
    # y3 = np.array(y3)*1000

    x = np.ones(y1.shape)
    y_normal = []
    y_partial = []
    # y_hierarchy = []
    x_l = []
    for i in range(model_num):
        y_normal.append(np.average(y1[i,:]))
        y_partial.append(np.average(y2[i,:]))
        # y_hierarchy.append(np.average(y3[i,:]))
        x_l.append(i+2)

    for i in range(model_num):
        x[i,:] = x[i,:] + i + 1
    
    y1 = np.reshape(y1, (model_num*repeat,1))
    y2 = np.reshape(y2, (model_num*repeat,1))
    # y3 = np.reshape(y3, (model_num*repeat,1))
    
    x = np.reshape(x, (model_num*repeat,1))
  
    
    plt.subplot(122)
    plt.title("Computation latency as objective")
    plt.xlabel('Server amount')
    # plt.ylabel('Time (ms)')
    plt.scatter(x, y1, alpha=0.6, color='blue', marker='x')
    plt.scatter(x, y2, alpha=0.3, color='deeppink', marker='s')
    # plt.scatter(x, y3, alpha=0.6, color='red', marker='+')

    plt.plot(x_l, y_normal, color='blue', linestyle='--', label='slsqp', marker='x')
    plt.plot(x_l, y_partial, color='deeppink',linestyle=':', label='pvi', marker='s')
    # plt.plot(x_l, y_hierarchy, color='red', linestyle='dashdot', label='hier', marker='+')
    plt.legend(loc=2)
    plt.yscale('log')
    plt.show()


def draw_comparing_performance(model_num, seed, repeat, subgraph=None):
    mid_name = "_servers-seed_%d-" % seed
    lower = mid_name+'lower_result.txt'
    normal = mid_name+'normal_result.txt'
    partial = mid_name+'partial_result.txt'
    seperate = mid_name+'seperate_result.txt'
    hierarchy = mid_name+'hierarchy_result.txt'

    result_folder = './puppy/results/performance/'

    f_lower = []
    f_normal = []
    f_partial = []
    f_seperate = []
    f_hierarchy = []
    for u in range(model_num):
        f_lower.append(np.array(np.loadtxt(result_folder + str(u+2) + lower)))
        f_normal.append(np.array(np.loadtxt(result_folder + str(u+2) + normal)))
        f_partial.append(np.array(np.loadtxt(result_folder + str(u+2) + partial)))
        f_seperate.append(np.array(np.loadtxt(result_folder + str(u+2) + seperate)))
        f_hierarchy.append(np.array(np.loadtxt(result_folder + str(u+2) + hierarchy)))

    y_lower = np.zeros((model_num, repeat))
    y_normal = np.zeros((model_num, repeat))
    y_partial = np.zeros((model_num, repeat))
    y_seperate = np.zeros((model_num, repeat))
    y_hierarchy = np.zeros((model_num, repeat))


    for i in range(model_num):
        lower_bound = np.min(f_lower[i])
        print(lower_bound)
        for j in range(repeat):
            # y_lower[i,j] = f_lower[i][j]
            y_normal[i,j] = f_normal[i][j]/lower_bound
            y_partial[i,j] = f_partial[i][j]/lower_bound
            y_seperate[i,j] = f_seperate[i][j]/lower_bound
            y_hierarchy[i,j] = f_hierarchy[i][j]/lower_bound

    x = np.zeros((model_num, repeat))
    for i in range(model_num):
        x[i,:] = x[i,:] + 2 + i


    y_n = []
    y_p = []
    y_s = []
    y_h = []
    x_l = []
    for i in range(model_num):
        x_l.append(i+2)
        y_n.append(np.min(y_normal[i]))
        y_p.append(np.min(y_partial[i]))
        y_s.append(np.min(y_seperate[i]))
        y_h.append(np.min(y_hierarchy[i]))
    
    x = np.reshape(x, (model_num*repeat, 1))
    y_normal = np.reshape(y_normal, (model_num*repeat, 1))
    y_partial = np.reshape(y_partial, (model_num*repeat, 1))
    y_seperate = np.reshape(y_seperate, (model_num*repeat, 1))
    y_hierarchy = np.reshape(y_hierarchy, (model_num*repeat, 1))
    
    plt.title("Performance")
    plt.xlabel('Server amount')
    plt.ylabel('Ratio')
    plt.yscale('log')
    plt.scatter(x, y_normal, alpha=0.6, color='blue', marker='x')
    plt.scatter(x, y_partial, alpha=0.3, color='green', marker='s')
    plt.scatter(x, y_seperate, alpha=0.3, color='deeppink', marker='o')
    plt.scatter(x, y_hierarchy, alpha=0.6, color='red', marker='+')

    normal_variance = []
    partial_variance = []
    seperate_variance = []

    plt.plot(x_l, y_n, color='blue', linestyle='--', label='slsqp',marker='x')
    plt.plot(x_l, y_p, color='green',linestyle=':', label='pvi', marker='s')
    plt.plot(x_l, y_s, color='deeppink', linestyle='dashdot', label='sep', marker='o')
    plt.plot(x_l, y_h, color='red', linestyle='dashdot', label='hier', marker='+')
    plt.ylim((1,15))
    plt.legend(loc=2)
    plt.show()

def draw_comparing_performance_histogram(model_num, seed, repeat, subgraph=None):
    mid_name = "_servers-seed_%d-" % seed
    lower = mid_name+'lower_result.txt'
    normal = mid_name+'normal_result.txt'
    partial = mid_name+'partial_result.txt'
    seperate = mid_name+'seperate_result.txt'
    hierarchy = mid_name+'hierarchy_result.txt'

    result_folder = './puppy/results/performance/'

    f_lower = []
    f_normal = []
    f_partial = []
    f_seperate = []
    f_hierarchy = []
    for u in range(model_num):
        f_lower.append(np.array(np.loadtxt(result_folder + str(u+2) + lower)))
        f_normal.append(np.array(np.loadtxt(result_folder + str(u+2) + normal)))
        f_partial.append(np.array(np.loadtxt(result_folder + str(u+2) + partial)))
        f_seperate.append(np.array(np.loadtxt(result_folder + str(u+2) + seperate)))
        f_hierarchy.append(np.array(np.loadtxt(result_folder + str(u+2) + hierarchy)))

    y_lower = np.zeros((model_num, repeat))
    y_normal = np.zeros((model_num, repeat))
    y_partial = np.zeros((model_num, repeat))
    y_seperate = np.zeros((model_num, repeat))
    y_hierarchy = np.zeros((model_num, repeat))


    for i in range(model_num):
        lower_bound = np.min(f_lower[i])
        print(lower_bound)
        for j in range(repeat):
            # y_lower[i,j] = f_lower[i][j]
            y_normal[i,j] = f_normal[i][j]/lower_bound
            y_partial[i,j] = f_partial[i][j]/lower_bound
            y_seperate[i,j] = f_seperate[i][j]/lower_bound
            y_hierarchy[i,j] = f_hierarchy[i][j]/lower_bound

    x = np.zeros((model_num, repeat))
    for i in range(model_num):
        x[i,:] = x[i,:] + 2 + i


    y_n = []
    y_p = []
    y_s = []
    y_h = []
    x_l = []

    std_n = []
    std_p = []
    std_s = []
    std_h = []

    y_m_n = []
    y_m_p = []
    y_m_s = []
    y_m_h = []
    for i in range(model_num):
        x_l.append(i+2)
        y_n.append(np.average(y_normal[i]))
        y_p.append(np.average(y_partial[i]))
        y_s.append(np.average(y_seperate[i]))
        y_h.append(np.average(y_hierarchy[i]))

        std_n.append(standard_error(y_normal[i]))
        std_p.append(standard_error(y_partial[i]))
        std_s.append(standard_error(y_seperate[i]))
        std_h.append(standard_error(y_hierarchy[i]))

        y_m_n.append(np.min(y_normal[i]))
        y_m_p.append(np.min(y_partial[i]))
        y_m_s.append(np.min(y_seperate[i]))
        y_m_h.append(np.min(y_hierarchy[i]))
    

    
    x = np.reshape(x, (model_num*repeat, 1))

    fig = plt.figure(1)
    for i in range(5):
        ax1 = fig.add_subplot(151+i)
        index = 2*i+2
        y = [y_n[index], y_p[index], y_s[index], y_h[index]]
        # y_err = [std_n[index], std_p[index], std_s[index], std_h[index]]
        ax1.bar(range(1, 2),y[0:1] , color='#FF9999' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="X")#, yerr=y_err)
        ax1.bar(range(2, 3),y[1:2] , color='#99CCCC' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="*")#, yerr=y_err)
        ax1.bar(range(3, 4),y[2:3] , color='#CCCCFF' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="//")#, yerr=y_err)
        ax1.bar(range(4, 5),y[3:4] , color='#CCCC99' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="o")#, yerr=y_err)
        # ax1.bar(range(1, 4), y_1[3:6], bottom=y_1[0:3],alpha=0.5, color='#99CCCC', edgecolor='black', hatch='//')
        ax1.set_xticks([1, 2, 3, 4])
        ax1.set_ylim((0,6))
        ax1.set_xticklabels(['slsqp','pvi','sep', 'hier'])
    # plt.show()

    fig = plt.figure(2)
    for i in range(5):
        print(i)
        ax1 = fig.add_subplot(151+i)
        index = 2*i+2
        y = [y_m_n[index], y_m_p[index], y_m_s[index], y_m_h[index]]
        ax1.bar(range(1, 2),y[0:1] , color='#FF9999' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="X")#, yerr=y_err)
        ax1.bar(range(2, 3),y[1:2] , color='#99CCCC' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="//")#, yerr=y_err)
        ax1.bar(range(3, 4),y[2:3] , color='#CCCCFF' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="*")#, yerr=y_err)
        ax1.bar(range(4, 5),y[3:4] , color='#CCCC99' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="o")#, yerr=y_err)
        # ax1.bar(range(1, 4), y_1[3:6], bottom=y_1[0:3],alpha=0.5, color='#99CCCC', edgecolor='black', hatch='//')
        ax1.set_xticks([1, 2, 3, 4])
        ax1.set_ylim((0,6))
        ax1.set_xticklabels(['slsqp','pvi','sep', 'hier'])
    plt.show()
    pass

def draw_time_vs_performance(model_num, repeat, seed, file_prefix):

    result_folder = './puppy/results/time_vs_performance/'
    y = np.loadtxt(result_folder+'%sseed_%d-normal.txt' % (file_prefix, seed))
    size = np.loadtxt(result_folder+'%sseed_%d-normal_result.txt' % (file_prefix, seed))
    x = np.zeros(y.shape)
    for i in range(model_num):
        x[i,:] = i + 2

    y = np.reshape(y, (model_num*repeat, 1))
    size = np.reshape(size, (model_num*repeat, 1))
    # size = np.log(size*1000)
    x = np.reshape(x, (model_num*repeat, 1))

    plt.title("Time VS Performance")
    plt.xlabel('Server amount')
    plt.ylabel('Time (ms)')
    plt.scatter(x, y, alpha=0.6, color='deeppink', marker='o', s=size)
    plt.yscale('log')
    plt.show()
    
def draw_backhaul_influence(file_prefix, repeat, inner_repeat, seed, server_num, group_num, task_type_num):
    result_folder = './puppy/results/backhaul_ban_ratio_influence/'
    ratio_list = np.loadtxt(result_folder + '%d_servers-seed_%d-ratio_list.txt' % (server_num, seed))
    y = np.zeros((len(ratio_list), group_num*task_type_num))
    for i in range(repeat):
        y = y + np.loadtxt(result_folder + file_prefix +'%d_servers-seed_%d-innerrepeat_%d.txt' % (server_num, i, inner_repeat))
    y = y/repeat
    plt.title("Backhaul Link Influence")
    plt.xlabel('Bandwith Ratio')
    plt.ylabel('Load')
    colors = ['red', 'green', 'blue', 'orange']
    linestyles = ['--',':','dashdot','dashed','solid']
    markers = ['x', 'o', 's', '+']
    alphas = [0.6, 0.3, 0.3, 0.6]
    for i in range(group_num):
        for j in range(task_type_num):
            plt.plot(1/ratio_list, y[:, i*task_type_num + j], linestyle=linestyles[j], alpha=alphas[j], color=colors[i], marker=markers[j])
    plt.show()
    pass
    
def draw_data_com_ratio_influence():
    result_folder = './puppy/results/data_com_ratio_influence/'
    file_1 = 'ratio-2.00_2.00_2.00-4_servers-seed_0-innerrepeat_4.txt'
    file_2 = 'ratio-2.25_2.00_1.75-4_servers-seed_1-innerrepeat_4.txt'
    file_3 = 'ratio-2.50_2.00_1.50-4_servers-seed_2-innerrepeat_4.txt'
    file_4 = 'ratio-2.75_2.00_1.25-4_servers-seed_3-innerrepeat_4.txt'

    # xticks = ['A', 'B', 'C', 'D', 'E']
    y_1 = np.loadtxt(result_folder+file_1)
    repeat_times = y_1.shape[0]
    y_1 = np.sum(y_1, 0)/repeat_times
    y_2 = np.sum(np.loadtxt(result_folder+file_2), 0)/repeat_times
    y_3 = np.sum(np.loadtxt(result_folder+file_3), 0)/repeat_times
    y_4 = np.sum(np.loadtxt(result_folder+file_4), 0)/repeat_times + 0.001 # 加0.001是为了防止绘图出现问题，因为数组中有0
    print(y_1)
    print(y_2)
    print(y_3)
    print(y_4)

    fig = plt.figure()
    ax1 = fig.add_subplot(141)
    ax1.bar(range(1, 4), y_1[0:3], color='#FF9999' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="X")
    ax1.bar(range(1, 4), y_1[3:6], bottom=y_1[0:3],alpha=0.5, color='#99CCCC', edgecolor='black', hatch='//')
    ax1.set_xticks([1, 2, 3])
    ax1.set_ylim((0,6.3))
    ax1.set_xticklabels(['2.00','2.00','2.00'])

    ax2 = fig.add_subplot(142)
    ax2.bar(range(1, 4), y_2[0:3], color='#FF9999' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="X")
    ax2.bar(range(1, 4), y_2[3:6], bottom=y_2[0:3],alpha=0.5, color='#99CCCC', edgecolor='black', hatch='//')
    ax2.set_xticks([1, 2, 3])
    ax2.set_ylim((0,6.3))
    ax2.set_xticklabels(['2.25','2.00','1.75'])

    ax3 = fig.add_subplot(143)
    ax3.bar(range(1, 4), y_3[0:3], color='#FF9999' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="X")
    ax3.bar(range(1, 4), y_3[3:6], bottom=y_3[0:3],alpha=0.5, color='#99CCCC', edgecolor='black', hatch='//')
    ax3.set_xticks([1, 2, 3])
    ax3.set_ylim((0,6.3))
    ax3.set_xticklabels(['2.50','2.00','1.50'])

    ax4 = fig.add_subplot(144)
    ax4.bar(range(1, 4), y_4[0:3], color='#FF9999' ,alpha=0.8,linewidth=1, edgecolor='black', hatch="X")
    ax4.bar(range(1, 4), y_4[3:6], bottom=y_4[0:3],alpha=0.5, color='#99CCCC', edgecolor='black', hatch='//')
    ax4.set_xticks([1, 2, 3])
    ax4.set_ylim((0,6.3))
    ax4.set_xticklabels(['2.75','2.00','1.25'])
    plt.show()
    pass

def draw_convergence(file_prefix):
    # obj_for_partial = np.loadtxt('./puppy/results/convergence/%spartial.txt' % file_prefix)
    # obj_for_normal = np.loadtxt('./puppy/results/convergence/%snormal.txt' % file_prefix)
    # obj_for_seperate = np.loadtxt('./puppy/results/convergence/%sseperate.txt' % file_prefix)

    plt.figure(figsize=(8, 4))
    

    # plt.title("Comparing between numerical methods") 
    # plt.xlabel("Iteration") 
    # plt.ylabel("Objective") 
    subgraph = 220
    for i in range(4):
        file_prefix = str(i*2 + 4) + "_servers-seed_4-"
        obj_for_partial = np.loadtxt('./puppy/results/convergence/%spartial.txt' % file_prefix)
        obj_for_normal = np.loadtxt('./puppy/results/convergence/%snormal.txt' % file_prefix)
        obj_for_seperate = np.loadtxt('./puppy/results/convergence/%sseperate.txt' % file_prefix)

        subgraph = subgraph + 1
        # print(subgraph)
        plt.subplot(subgraph)
        length = 100
        x = np.arange(1,length+1)
        plt.plot(x, np.log(np.abs(obj_for_partial)+1)[0:length],alpha=1, color="red",linewidth=1.5,linestyle='-.',label='pvi')#, marker='s', markersize=4)
        plt.plot(x, np.log(np.abs(obj_for_normal)+1)[0:length], color="blue",linewidth=1.5,linestyle='-.',label='slsqp')#, marker='x', markersize=4)
        plt.plot(x, np.log(np.abs(obj_for_seperate)+1)[0:length],alpha=1, color="green",linewidth=1.5,linestyle='dashdot',label='sep',)# marker='o', markersize=4)
        plt.plot(x, np.zeros(length), color='tab:pink', linestyle='dotted', linewidth=1)
        plt.title(str(i*2 + 4) + ' servers')
        if i == 2 or i == 3:
            plt.xlabel("Iteration") 
        if i == 0 or i == 2:
            plt.ylabel("Objective") 

        plt.legend(loc=1)
        plt.xlim(0,length)
        plt.ylim(0.5, 5)
    plt.show()


def standard_error(x):
    length = len(x)
    avg = np.sum(x)/length
    return np.sqrt(np.sum(abs(x-avg)**2)/length)

def linear_regression(seed, model_num, solver_name):
    result_folder = './puppy/results/performance/'
    config_folder = './puppy.config/'
    time_consumption_folder = './puppy/results/time_consumption/'
    tc_file = time_consumption_folder + "%d_models-%s-seed_%d-%s.txt" % (model_num, 'linear', seed, solver_name)


    y = np.loadtxt(tc_file)*1000 # y必须是二维的
    x = range(2, y.shape[0]+2)
    def obj(para):
        err = 0
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                err = err + np.square(para[0]*x[i] + para[1] - np.log(y[i,j]) )
        return err
    res = minimize(obj, [1,1])
    print(res)
        # print(avg)
    return res.x

if __name__ == "__main__":
    np.set_printoptions(formatter={'float':'{:.3f}'.format})
    seed = 40


    # linear_regression(30, model_num=19, solver_name='partial')

    # create_model_manager_v2(1)
    # test_backhaul_ban_influence(seed=seed, repeat=20, inner_repeat=1, data_size_list=[4.25,4,3.75])
    # draw_backhaul_influence(file_prefix="ratio-2.12_2.00_1.88-",repeat=20, inner_repeat=1 , seed=seed, server_num=4, group_num=2, task_type_num=3)


    # test_data_com_ratio_influence(10,seed, 4)

    # draw_convergence('8_servers-seed_4-')
    # draw_comparing_time_consumption_comb_version()

    # draw_data_com_ratio_influence()


    # file_prefix = '11_models-linear-'
    # test_time_vs_performance(seed=seed, file_prefix=file_prefix, repeat=5)
    # draw_time_vs_performance(model_num=11, seed=seed, repeat=5, file_prefix=file_prefix)


    # test_numerical_method_performance(seed=10, repeat=10)
    # draw_comparing_performance(model_num=9, seed=10, repeat=10)

    # 修改了下拓扑的结构
    # test_numerical_method_performance(seed=seed, repeat=5, need_timing=True)
    # test_hierarchy_method_performance(seed=seed, repeat=5, need_timing=True)
    # draw_comparing_performance(model_num=10, seed=15, repeat=5)
    # draw_comparing_performance_histogram(model_num=11, seed=15, repeat=5)

    # file_prefix = str(11)+'_models-'+'linear-'
    # test_hierarchy_method_performance(seed=15, repeat=5)
    # test_hierarchy_method_time_consumption(seed=seed, repeat=5, file_prefix=str(11)+'_models-'+'linear-')

    # draw_comparing_performance(model_num=11, seed=15, repeat=5)

    # test_numerical_method_time_consumption(seed=seed, gen_feasible='linear', file_prefix=str(11)+'_models-'+'linear-')
    # draw_comparing_time_consumption(model_num=11, repeat=5, file_prefix=str(11)+'_models-'+'linear-seed_%d-'% seed)
    # draw_comparing_time_consumption(model_num=11, repeat=5, file_prefix=str(11)+'_models-'+'linear-')

    draw_comparing_time_consumption(seed=35, model_num=19, repeat=5, gen_feasible='linear')

    
    
    