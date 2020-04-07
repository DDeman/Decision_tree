#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '任晓光'
__mtime__ = '2020/3/30'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import numpy as np
import pandas as pd
def clcShannonEnt(data,groupby='类别'):
    data_ = data.groupby(by=groupby).groups
    keys = data_.keys()
    D = len(data)
    H_D = 0
    for clas in keys:
        C_k = len(data_[clas])
        H_D -= (C_k / D) * np.log2(C_k / D)
    return H_D

def best_split(data,H_D):
    columns = data.iloc[:,:-1].columns
    D = len(data)
    G_D_A_list = []
    for column in columns:
        G_D_A = 0
        data_ = data.groupby(by=column).groups
        keys = data_.keys()
        H_D_I_ = 0
        for clas in keys:
            D_i = len(data[data[column] == clas])
            H_D_i = (D_i / D) * clcShannonEnt(data[data[column] == clas])
            H_D_I_ = H_D_I_ + H_D_i
        G_D_A_list.append(H_D - H_D_I_)
    result_dic = {}
    for i in range(len(columns)):
        result_dic[columns[i]] = G_D_A_list[i]
    result = sorted(result_dic.items(),key=lambda item:item[1],reverse=True)
    split_name = result[0][0]
    return result,split_name

def find_j_s(data,feature_name,class_name):
    sort_index = data.sort_values(by=feature_name).index
    j_s = dict()
    res_r = 9999   #最小的切分点的值
    res_s = None     #最小的那个切分点
    C1 = None
    C2 = None
    for index in range(len(sort_index)):
        s= data.iloc[index][feature_name]
        if len(sort_index[:index]) == 0:
            c1 = 0
        else:
            c1 = sum(data.iloc[sort_index[:index]][class_name]) / len(sort_index[:index])
        if len(sort_index[index:]) == 0:
            c2 = 0
        else:

            c2 = sum(data.iloc[sort_index[index:]][class_name]) / len(sort_index[index:])

        yi = data.iloc[index][class_name]
        value = ((yi - c1)**2 + (yi - c2) ** 2) / len(data)
        if s in j_s:
            j_s[s] = min(value,j_s[s])
        else:
            j_s[s] = value

        if value < res_r:
            res_r = value
            res_s = s
            C1 = c1
            C2 = c2
    return j_s,res_s,res_r,C1,C2
def best_split_cart(data):
    columns = data.columns    #数据的feature名字，包含y最后一列
    class_name = columns[-1]
    j_list = columns[:-1]
    best_j_list = []    #记录每个feature的最小的值
    best_j_s_list = []  #记录每个feature的最优s
    C1_list = []
    C2_list = []
    print(columns[:-1])
    # if len(columns[:-1]) == 0:
    #
    for feature_name in columns[:-1]:
        j_s, res_s, res_r,C1,C2 = find_j_s(data,feature_name,class_name)
        best_j_list.append(res_r)
        best_j_s_list.append(res_s)
        C1_list.append(C1)
        C2_list.append(C2)
        # print(j_s,res_s,res_r,C1,C2)
    print(best_j_list,best_j_s_list,C1_list,C2_list)
    best_j = j_list[best_j_list.index(min(best_j_list))]
    best_j_s = best_j_s_list[best_j_list.index(min(best_j_list))]
    result_c1 = C1_list[best_j_list.index(min(best_j_list))]
    result_c2 = C2_list[best_j_list.index(min(best_j_list))]


    return best_j,best_j_s,result_c1,result_c2