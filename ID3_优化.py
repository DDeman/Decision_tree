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
from unit import clcShannonEnt, best_split
import copy

#优化目标
'''
 criterion="gini",
 splitter="best",    #best,最好的分裂节点，random，是随机下最好的（会有一部分特征不用）
 max_depth=None,
 min_samples_split=2,
 min_samples_leaf=1,
 min_weight_fraction_leaf=0.,
 max_features=None,
 random_state=None,
 max_leaf_nodes=None,
 min_impurity_decrease=0.,
 min_impurity_split=None,
 class_weight=None,
 presort=False
'''


class Tree():
    def __init__(self, name, value, link=None, clas=None):
        self.name = name
        self.children = {}
        self.value = value
        self.parent = None
        self.clas = clas
        self.result = None

    def disp(self, indx, key=None):
        print(' ' * indx, self.name, key, self.value, self.result)
        for key, child in self.children.items():
            child.disp(indx + 1, key)


class ID3():
    def __init__(self,criterion='id3',max_depth=None,min_samples_leaf=1,max_features=None):
        self.criterion = criterion    #id3,c4.5,gini
        self.max_depth = max_depth
        min_samples_leaf = min_samples_leaf
        max_features = max_features
    def fit(self, X,y, tree):
        data_pd = pd.concat([X,y],axis=1)
        print(y)
        class_name = y.columns[0]
        if len(data_pd[class_name].value_counts()) == 0:
            return
        elif len(data_pd[class_name].value_counts()) == 1:
            tree.result = list(data_pd[class_name].items())[0][1]
            return
        else:
            H_D = clcShannonEnt(data_pd, groupby=class_name)  # 如果H_D为0结束递归
            result, split_name = best_split(data_pd, H_D)
            # print(result[0])
            #     tree.next_link = ID3_tree(split_name,result[0][1])
            for key in data_pd.groupby(by=split_name).groups.keys():
                tree.children[key] = Tree(split_name, result[0][1])
                data_pd_ = data_pd[data_pd[split_name] == key]
                self.fit(data_pd_.iloc[:,:-1],data_pd_.iloc[:,-1:], tree.children[key])
        return tree
    def predict(self, mytree, data_pd):
        result = []
        n_data_pd = len(data_pd)
        for i in range(n_data_pd):
            self.fun(mytree, data_pd.iloc[i], result)
        return result

    def fun(self, mytree, data_pd, result):
        if mytree.result is not None:
            result.append(mytree.result)
            return
        for tiaojian in mytree.children:
            feature = mytree.children[tiaojian].name
            break
        mytree = mytree.children[data_pd[feature][0]]
        self.fun(mytree, data_pd, result)


if __name__ == '__main__':



    ####test the  ID3
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
    mytree = Tree('mytree', 1)
    tree = ID.fit(data_pd.iloc[:,:-1],data_pd.iloc[:,-1:],mytree)

    # data = [['青年', '否', '否', '一般']]
    # data_pd_test = pd.DataFrame(data, columns=['年龄', '有工作', '有自己的房子', '信贷情况'])
    # res = ID.predict(tree, data_pd_test)
    mytree.disp(2)
    # print(res)
    # print(data_pd['类别'])
    
    '''




    from sklearn.tree import DecisionTreeClassifier
    DecisionTreeClassifier()
    '''