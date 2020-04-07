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
import numpy as np
import pandas as pd

class Tree():
    def __init__(self, j,s=None,value=None,is_lefa=False):
        self.j = j        #存放切分变量
        self.s = s        #存放切分变量的切分点
        self.value = value   #叶子节点的输出值
        self.is_leaf = is_lefa   #判断是否为叶子节点，用于预测时，递归的截止条件
        self.left = None     #存放左子树
        self.right = None    #存放右子树

    def disp(self, indx, key=None):
        print(' ' * indx, self.j, self.s, self.value, self.is_leaf)
        if self.left:
            self.left.disp(indx+3,key)
        if self.right:
            self.right.disp(indx+3,key)

class CART_regression():

    def split_data(self,data,j,s):
        '''
        :param data: dataset to be splited
        :param j: 切分变量
        :param s: 切分变量的切分点
        :return: 切分后的数据集
        '''
        left_data = data[data[j] < s].reset_index(drop=True)
        right_data = data[data[j] >= s].reset_index(drop=True)
        return left_data,right_data

    def best_split(self,data,ops=(1,114)):
        '''
        :param data:切分数据集
        :param ops:
        :return:
        '''
        if data.shape[0] == 1:  #所划分的数据集只有一个样本返回
            return None,None,data.iloc[:,-1].mean(),None,None
        tol_N = ops[0]
        m,n = data.shape
        best_value = np.inf      #设置最大值
        best_j = None
        best_s = None
        c1_ = None
        c2_ = None
        for j in data.columns[:-1]:    #遍历特征列表 ['面积','距离'.....
            colval = set(data[j])      #这个特征下，找不重复的值，避免多余计算（剪枝）
            for s in colval:          #遍历每个样本
                left_data,right_data = self.split_data(data,j,s)           #按照切分点，切分数据集，用以计算c1，c2...
                if len(left_data) == 0 or len(right_data) == 0:            #当切分点为最大或者最小时，如果不进行这一步判断，由于有一个集合会为空，下面会报错
                    continue
                else:
                    c1 = sum((left_data.iloc[:,-1] - left_data.iloc[:,-1].mean())**2)        #均方误差的和
                    c2 = sum((right_data.iloc[:,-1] - right_data.iloc[:,-1].mean())**2)
                    values =  c1 + c2
                    if values < best_value:   #cart回归树mse公式，找最小的value
                        best_j = j
                        best_s = s
                        best_value = values
                        c1_ = c1
                        c2_ = c2
        s = sum((data.iloc[:,-1] - data.iloc[:,-1].mean())**2)    #计算data的最后一列，也就是y，的均方误差的和用来剪枝
        print(abs(s - best_value))
        if abs(s - best_value) < ops[1]:              #当小于所设定的value最小下降度时，返回，（这一步会剪掉很大一部分树，如果不设置，会以0.0几的变化量下降，很慢，但同时设置了也会影响到R2的值
            return None, None, data.iloc[:, -1].mean(), None, None

        if best_j is None:             #best_j最初设置为None，这一步触发的条件是所有特征的colval都为1，此时不会计算c1，c2
            return None, None, data.iloc[:, -1].mean(), None, None
        else:
            left_data ,right_data = self.split_data(data,best_j,best_s)    #这一步进行数据集切分，是为了进行剪枝，判断切分后的数据集（是否满足叶子节点的要求）

        if len(left_data) < tol_N or len(right_data) < tol_N:       #剪枝，数据集是否满足
            return None,None,data[:,-1].mean(),None,None

        return best_j,best_s,best_value,c1_,c2_


    def fit(self,X,y,tree,ops=[1,114]):
        '''
        :param X: 要训练的样本x,格式为pandas，且有columns
        :param y: 训练样本的结果，格式为pandas，且有columns
        :param tree: 用来存放训练出来的cart树的结果
        :param ops: 俩个剪枝参数，ops[0]为叶子节点最小样本数，ops[1]为结果value的最小下降度，注意这里的value和树的value不是同一个，树的value为最后叶子节点上样本输出的mean
        :return:生成一个cart回归树
        '''
        data_pd = pd.concat([X, y], axis=1)
        j,s,value,c1,c2 = self.best_split(data=data_pd)  #寻找最佳切分变量和切分点
        if j == None:      #此时为叶子节点
            tree.is_leaf = True
            tree.value = value    #叶子节点输出值，用以预测
            return
        tree.left = Tree(j,s,c1)
        tree.right = Tree(j,s,c2)
        left_data,right_data = self.split_data(data_pd,j,s)

        self.fit(left_data.iloc[:,:-1],left_data.iloc[:,-1],tree.left)
        self.fit(right_data.iloc[:, :-1], right_data.iloc[:, -1],tree.right)


    def predict(self,tree,data_pd):
        result = []       #存放结果
        n_data_pd = len(data_pd)
        for i in range(n_data_pd):
            self.fun(tree, data_pd.iloc[i], result)
        return result

    def fun(self,tree,data_pd,result):
        if tree.is_leaf:
            result.append(tree.value)
            return
        feature = tree.left.j   #这里tree.left.j和tree.right.j是一样的，是用来存放切分变量的，不管是左还是又，切分变量是一个
        if data_pd[feature] < tree.left.s:
            self.fun(tree.left,data_pd,result)
        else:
            self.fun(tree.right,data_pd,result)




if __name__ == '__main__':
    '''
    data = [['青年', '否', '否', '一般', 1]
        , ['青年', '否', '否', '好', 1.3]
        , ['青年', '是', '否', '好', 4.6]
        , ['青年', '是', '是', '一般', 2.4]
        , ['青年', '否', '否', '一般', 2]
        , ['中年', '否', '否', '一般', 3]
        , ['中年', '否', '否', '好', 4]
        , ['中年', '是', '是', '好', 7]
        , ['中年', '否', '是', '非常好', 7]
        , ['中年', '否', '是', '非常好', 9]
        , ['老年', '否', '是', '非常好', 1]
        , ['老年', '否', '是', '好', 2]
        , ['老年', '是', '否', '好', 3]
        , ['老年', '是', '否', '非常好', 1]
        , ['老年', '否', '否', '一般', 6]]
    data_pd = pd.DataFrame(data, columns=['年龄', '有工作', '有自己的房子', '信贷情况', '类别'])

    data = [['老年', '否', '是', '非常好']]
    data_pd_test = pd.DataFrame(data, columns=['年龄', '有工作', '有自己的房子', '信贷情况'])
    # print(data_pd_test)
    # print(c)
    cart_huigui = CART_huigui()
    mytree = Tree('mytree', 1)
    cart_huigui.fit(data_pd.iloc[:,:-1],data_pd.iloc[:,-1:],mytree)
    result = cart_huigui.predict(mytree,data_pd.iloc[:,:-1])
    # mytree.disp(2)
    print(result)
    '''


    from sklearn.datasets import load_boston
    from sklearn.metrics import accuracy_score,mean_squared_error,r2_score


    data = load_boston()
    x, y = data.data, data.target
    x_pd = pd.DataFrame(x, columns=data.feature_names)
    y_pd = pd.DataFrame(y, columns=['result'])
    data_pd = pd.concat([x_pd, y_pd], axis=1)

    # x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 19],[6, 3, 8, 4, 2, 5, 7, 1, 9, 9, 12, 19]]
    # x = np.array(x).reshape(-1, 2)
    # y = [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.87, 9, 9.05, 8, 0]
    #
    # x_pd = pd.DataFrame(x, columns=['f1','f2'])
    # y_pd = pd.DataFrame(y, columns=['result'])

    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 19]
    # y = [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.87, 9, 9.05, 8, 0]
    # x_pd = pd.DataFrame(x, columns=['f1'])
    # y_pd = pd.DataFrame(y, columns=['result'])

    cart_huigui = CART_regression()
    mytree = Tree('mytree',1)
    ops = [1,114]
    cart_huigui.fit(x_pd,y_pd,mytree,ops)

    mytree.disp(1)
    y_pred = cart_huigui.predict(mytree,x_pd.iloc[:,:])
    #
    print(y_pred)
    print(y)
    #
    print(mean_squared_error(y_pd,y_pred))
    print(r2_score(y_pd,y_pred))
