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
import pandas as pd
import numpy as np
from unit import clcShannonEnt,best_split
import copy

class ID3_tree():
    def __init__(self,name,value,link=None,clas=None):
        self.name = name
        self.children = {}
        self.value = value
        self.parent = None
        self.clas = clas
        self.result = None
    def disp(self,indx,key=None):
        print(' '*indx,self.name,key,self.value,self.result)
        for key,child in self.children.items():
            child.disp(indx+1,key)

class ID3():

    def fit(self,data_pd,tree):
        if len(data_pd['类别'].value_counts()) == 0:
            return
        elif len(data_pd['类别'].value_counts()) == 1:
            tree.result = list(data_pd['类别'].items())[0][1]
            return
        else:
            H_D = clcShannonEnt(data_pd,groupby='类别')   #如果H_D为0结束递归
            result,split_name = best_split(data_pd,H_D)
            # print(result[0])
        #     tree.next_link = ID3_tree(split_name,result[0][1])
            for key in data_pd.groupby(by=split_name).groups.keys():
                tree.children[key] = ID3_tree(split_name,result[0][1])
                data_pd_ = data_pd[data_pd[split_name] == key]
                self.fit(data_pd_,tree.children[key])
    def predict(self,mytree,data_pd):
        result = []
        n_data_pd = len(data_pd)
        for i in range(n_data_pd):
            self.fun(mytree,data_pd.iloc[i],result)
        return result
    def fun(self,mytree,data_pd,result):
        if mytree.result is not None:
            result.append(mytree.result)
            return
        for tiaojian in mytree.children:
            feature = mytree.children[tiaojian].name
            break
        mytree = mytree.children[data_pd[feature][0]]
        self.fun(mytree,data_pd,result)

if __name__ == '__main__':
    data = [['青年', '否', '否', '一般', '否']
        , ['青年', '否', '否', '好', '否']
        , ['青年', '是', '否', '好', '是']
        , ['青年', '是', '是', '一般', '是']
        , ['青年', '否', '否', '一般', '否']
        , ['中年', '否', '否', '一般', '否']
        , ['中年', '否', '否', '好', '否']
        , ['中年', '是', '是', '好', '是']
        , ['中年', '否', '是', '非常好', '是']
        , ['中年', '否', '是', '非常好', '是']
        , ['老年', '否', '是', '非常好', '是']
        , ['老年', '否', '是', '好', '是']
        , ['老年', '是', '否', '好', '是']
        , ['老年', '是', '否', '非常好', '是']
        , ['老年', '否', '否', '一般', '否']]
    data_pd = pd.DataFrame(data, columns=['年龄', '有工作', '有自己的房子', '信贷情况', '类别'])
    ID = ID3()
    mytree = ID3_tree('mytree',1)
    ID.fit(data_pd,mytree)
    res = ID.predict(mytree,data_pd)
    mytree.disp(2)
    print(res)
    print(data_pd['类别'])