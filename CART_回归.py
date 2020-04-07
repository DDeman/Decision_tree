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
# from unit import best_split_cart
from unit_2 import best_split_cart
class Tree():
    def __init__(self, j,s=None,value=None,is_lefa=False):
        self.j = j
        self.s = s
        self.value = value
        self.is_leaf = is_lefa
        self.left = {}
        self.right = {}

    def disp(self, indx, key=None):
        print(' ' * indx, self.j, self.s,key, self.value, self.is_leaf)
        for key,left in self.left.items():
            left.disp(indx+3,key)
        for key,right in self.right.items():
            right.disp(indx+3,key)

class CART_huigui():

    def fit(self,X,y,tree):
        data_pd = pd.concat([X,y],axis=1)

        # print(class_name)

        best_j, best_j_s,result_c1,result_c2,is_stop = best_split_cart(data_pd)
        print(best_j, best_j_s,result_c1,result_c2,is_stop)
        # if best_j_s is None:
        #     return
        # else:

        # print(best_j,best_j_s,result_c1,result_c2,is_stop)
        tree.left['left'] = Tree(j=best_j, s=best_j_s,value=result_c1)
        tree.right['right'] = Tree(j=best_j, s=best_j_s, value=result_c2)
        # data = data_pd.iloc[:,data_pd.columns != best_j]

        if is_stop:
            # tree.left['left'].is_leaf = True
            # tree.right['right'].is_leaf = True
            tree.is_leaf = True
            return
        else:
            left_data = data_pd[data_pd[best_j] < best_j_s].reset_index(drop=True)
            right_data = data_pd[data_pd[best_j] >= best_j_s].reset_index(drop=True)
            print(len(left_data),len(right_data))
            if len(left_data) == 0 or len(right_data) == 0:            #可以设定叶子节点最小样本数
                # tree.right['right'].is_leaf = True
                # tree.left['left'].is_leaf = True

                return
            self.fit(left_data.iloc[:, :-1], left_data.iloc[:, -1:], tree.left['left'])
            self.fit(right_data.iloc[:, :-1], right_data.iloc[:, -1:], tree.right['right'])
        '''
        left_data = data_pd[data_pd[best_j] <= best_j_s].reset_index(drop=True).iloc[:,data_pd.columns != best_j]
        right_data = data_pd[data_pd[best_j] > best_j_s].reset_index(drop=True).iloc[:,data_pd.columns != best_j]
        # print(len(left_data))
        # print(left_data.columns)
        tree.left['left'] = Tree(j=best_j, s=best_j_s,value=result_c1)
        tree.right['right'] = Tree(j=best_j, s=best_j_s, value=result_c2)

        if len(left_data.columns) == 1 or len(right_data.columns) == 1:
            tree.left['left'].is_leaf = True
            tree.right['right'].is_leaf = True
            return
        else:
            self.fit(left_data.iloc[:, :-1], left_data.iloc[:, -1:], tree.left['left'])
            self.fit(right_data.iloc[:, :-1], right_data.iloc[:, -1:], tree.right['right'])
        '''

    def predict(self,tree,data_pd):
        result = []
        n_data_pd = len(data_pd)
        for i in range(n_data_pd):
            # temp = []
            self.fun(tree, data_pd.iloc[i], result)
            # result.append(temp)
        return result

    def fun(self,tree,data_pd,result):
        if tree.left['left'].is_leaf and tree.right['right'].is_leaf:
        # if tree.is_leaf:
            result.append(tree.value)
            return
        # print(tree.is_leaf)
        feature = tree.left['left'].j
        # print(feature)
        if data_pd[feature] < tree.left['left'].s:
            # result.append(tree.left['left'].value)
            self.fun(tree.left['left'],data_pd,result)
        else:
            # result.append(tree.right['right'].value)
            self.fun(tree.right['right'],data_pd,result)




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

    '''
    data = load_boston()
    x, y = data.data, data.target
    x_pd = pd.DataFrame(x, columns=data.feature_names)
    y_pd = pd.DataFrame(y, columns=['result'])
    data_pd = pd.concat([x_pd, y_pd], axis=1)
    '''
    x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 19],[6, 3, 8, 4, 2, 5, 7, 1, 9, 9, 12, 19]]
    x = np.array(x).reshape(-1, 2)
    y = [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.87, 9, 9.05, 8, 0]
    x_pd = pd.DataFrame(x, columns=['f1','f2'])
    y_pd = pd.DataFrame(y, columns=['result'])



    cart_huigui = CART_huigui()
    mytree = Tree('mytree',1)
    cart_huigui.fit(x_pd.iloc[:,:],y_pd,mytree)

    mytree.disp(1)
    y_pred = cart_huigui.predict(mytree,x_pd.iloc[:,:])

    print(y_pred)
    print(y_pd)

    print(mean_squared_error(y_pd,y_pred))
    print(r2_score(y_pd,y_pred))
