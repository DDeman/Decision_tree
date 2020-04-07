#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '任晓光'
__mtime__ = '2020/4/2'
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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

class CART_classification():
    def gini(self,data):  #计算基尼系数
        classes = set(data.iloc[:,-1])
        n = len(data)
        p = 0
        for clas in classes:
            num = (data.iloc[:,-1] == clas).value_counts()[0]
            p += (num / n) * (1 - num / n)
        return p

    def split_data(self,data,j,s):
        '''
        :param data: dataset to be splited
        :param j: 切分变量
        :param s: 切分变量的切分点
        :return: 切分后的数据集
        '''
        left_data = data[data[j] == s].reset_index(drop=True)
        right_data = data[data[j] != s].reset_index(drop=True)
        return left_data,right_data

    def best_split(self,data,ops=(1,114)):
        '''
        :param data:切分数据集
        :param ops:
        :return:
        '''
        if len(set(data.iloc[:,-1])) == 1:  #如果数据集分类结果只有一种，则返回为叶子节点

            return None,None,data.iloc[:,-1][0]
        best_value = np.inf      #设置最大值
        best_j = None
        best_s = None

        for j in data.columns[:-1]:    #遍历特征列表 ['面积','距离'.....
            colval = set(data[j])      #这个特征下，找不重复的值，避免多余计算（剪枝）
            for s in colval:          #遍历每个样本
                # print('样本数',len(data),j,s)
                left_data,right_data = self.split_data(data,j,s)           #按照切分点，切分数据集，用以计算c1，c2...
                c1,c2 = 0,0
                if len(left_data) ==0 or len(right_data) ==0:    #这里如果不判断，会导致计算基尼系数时，分母有零的情况
                    continue
                if  len(set(left_data.iloc[:,-1])) > 1:          #这里保证俩个数据集，其中一个分类结果只有一种，但另一个分类结果不是，也就是会有基尼系数，如果不判断会漏掉这一部分
                    c1 = self.gini(left_data) * (len(left_data) / len(data))
                if len(set(right_data.iloc[:,-1])) > 1:
                    c2 = self.gini(right_data) * (len(right_data) / len(data))
                values =  c1 + c2

                if values < best_value:   #基尼系数
                    best_j = j
                    best_s = s
                    best_value = values

        if best_j is None:             #best_j最初设置为None，这一步触发的条件是所有特征的colval都为1，此时不会计算c1，c2
            return None, None, data.iloc[:, -1][0]

        return best_j,best_s,best_value


    def fit(self,X,y,tree):
        '''
        注意输入的数据集，x如果是数值型变量,也就是1.2，3.4此类的，要进行分箱（数据预处理）否则计算会特别慢，因为set（）会很多很多！！！！
        :param X: 要训练的样本x,格式为pandas，且有columns
        :param y: 训练样本的结果，格式为pandas，且有columns
        :param tree: 用来存放训练出来的cart树的结果
        :return:生成一个cart分类树
        '''
        data_pd = pd.concat([X, y], axis=1)
        # print(data_pd)
        j,s,value = self.best_split(data=data_pd)  #寻找最佳切分变量和切分点
        print(j,s,value)
        if j == None:      #此时为叶子节点
            tree.is_leaf = True
            tree.value = value    #叶子节点输出值，用以预测
            return
        tree.left = Tree(j,s,value)
        tree.right = Tree(j,s,value)
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
        if data_pd[feature] is tree.left.s:
            self.fun(tree.left,data_pd,result)
        else:
            self.fun(tree.right,data_pd,result)



if __name__ == '__main__':


    from sklearn.datasets import load_boston,load_breast_cancer
    from sklearn.metrics import accuracy_score,mean_squared_error,r2_score,confusion_matrix

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
    x_pd = data_pd.iloc[:,:-1]
    y_pd = data_pd.iloc[:,-1]




    cart_huigui = CART_classification()
    mytree = Tree('mytree',1)
    cart_huigui.fit(x_pd.iloc[:,:],y_pd,mytree)

    mytree.disp(1)
    y_pred = cart_huigui.predict(mytree,x_pd.iloc[:,:])
    #
    print(y_pred)
    print(accuracy_score(y_pd,y_pred))
    print(confusion_matrix(y_pd,y_pred))