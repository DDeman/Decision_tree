#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '任晓光'
__mtime__ = '2020/3/31'
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
import pandas as pd
import numpy as np

def find_j_s(data,feature_name,class_name):
    sort_index = data.sort_values(by=feature_name).index
    C1 = []
    C2 = []
    s_name_list = []
    value_list = []
    # print(len(sort_index))
    if len(sort_index) <= 1:
        return None,None,None,None,True
    else:
        for index in range(1,len(sort_index)):

            #     c1 = 0
            # else:
            c1 = sum(data.iloc[sort_index[:index]][class_name]) / len(sort_index[:index])
            # if len(sort_index[index:]) == 0:
            #     c2 = 0
            # else:
            c2 = sum(data.iloc[sort_index[index:]][class_name]) / len(sort_index[index:])
            '''
            v1 = 0
            v2 = 0
            for i in sort_index[:index]:
                v1 += (data.iloc[i][class_name] - c1) ** 2
            for i in sort_index[index:]:
                v2 += (data.iloc[i][class_name] - c2) ** 2
            '''
            # yi = data.iloc[index][class_name]
            # value = ((yi - c1)**2) /len(sort_index[:index]) + ((yi - c2) ** 2) / len(sort_index[index:])
            # value = (yi - c1) ** 2 + (yi - c2) ** 2
            value = sum((data.iloc[sort_index[:index]][class_name] - c1) ** 2) + sum((data.iloc[sort_index[index:]][class_name] - c2) ** 2)
            value_list.append(value)
            s_name_list.append(data[feature_name].iloc[index])
            C1.append(c1)
            C2.append(c2)
        # print(value_list)
        # print(sort_index)
        index = value_list.index(min(value_list))
        value = value_list
        c1 = C1[index]
        c2 = C2[index]
        s_name = s_name_list[index]
        return s_name,value,c1,c2,False


def best_split_cart(data):
    columns = data.columns    #数据的feature名字，包含y最后一列
    class_name = columns[-1]

    s_name_list = []
    value_list = []
    c1_list = []
    c2_list = []

    if len(data) == 0:
        return None, None, None, None, True

    # if len(columns[:-1]) == 0:
    #     return None,None,None,None,True
    else:
        # print(columns[:-1])
        for feature_name in columns[:-1]:
            s_name, value, c1, c2,is_ = find_j_s(data,feature_name,class_name)
            if is_:
                return None, None, None, None, True
            else:
                s_name_list.append(s_name)
                value_list.append(value)
                c1_list.append(c1)
                c2_list.append(c2)
        index = value_list.index(min(value_list))

        best_j = columns[:-1][index]
        best_j_s = s_name_list[index]
        result_c1 = c1_list[index]
        result_c2 = c2_list[index]

        return best_j,best_j_s,result_c1,result_c2,False