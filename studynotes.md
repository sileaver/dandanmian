# ***一份学习报告***

## *变量，字符串，数字*。

没什么感觉，可能就是注意一下数字需要用str（）才能和字符串一起输出。

## *列表*

挺方便的，啥都能往里丢，相当于加强版的c语言数组

使用sort()对列表进行永久性排序，默认正序，想逆序sort（reverse=True）

使用sorted（）对列表进行临时性排序，接下来同上

使用reverse（）使列表倒着，想恢复再来一回reverse（），毕竟负负得正（笑）

len（）确定长度

索引依旧从零开始

利用range创建数字列表，range（1,5）不打印5，有min，max，sum这种解放双手的函数

切片就是创造一个列表的副本，想咋整咋整

## *元组*

我的理解是这玩意是个常数数组

 ## 元组不能迭代

## *循环*（末尾打冒号）

注意缩进（o(╥﹏╥)o）

### *for循环*

区别不大，不过好像没有可以花式整活的（；；）有点怀念

### *if语句*（同上）

Python对大小写敏感

#### and

#### or

#### 检查特点值是否不包含在列表中，使用关键字not in

示例：if user not in users：

#### if else语句

#### if -elif-else

同c的else if

## *字典*

啥都能丢，键值对是它最大的特点

### items

### key（）

### value（）

## *input（）*

## *while循环*

break之流不多赘述

## *函数*

## *模块*

### as取别名

### 类

这个算是Python的特点了，但是我现在还没悟出来有啥用

#### 继承

#### Python标准库

## *文件和异常*

文件就丢个名字上去，异常为了处理traceback使其更加安全

### try-except

#### 避免崩溃，处理异常

# 对神经网络有简单了解并能搭建一个简单的bp，虽然是改的。

## pycharm

ctrl+shift+alt+点击鼠标左键.可以在任意位置增加光标.

# softmax



输入向量{\displaystyle [1,2,3,4,1,2,3]}![{\displaystyle [1,2,3,4,1,2,3]}](https://wikimedia.org/api/rest_v1/media/math/render/svg/d068344b5d5265343f7fdf213e30f73afe408278)对应的Softmax函数的值为{\displaystyle [0.024,0.064,0.175,0.475,0.024,0.064,0.175]}![{\displaystyle [0.024,0.064,0.175,0.475,0.024,0.064,0.175]}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c9679f3b44ebb0dbf20990e3d86b29b33e57c564)。输出向量中拥有最大权重的项对应着输入向量中的最大值“4”。这也显示了这个函数通常的意义：对向量进行归一化，凸显其中最大的值并抑制远低于最大值的其他分量。



# bn

（2）批规范化BN，标准化Standardization，正则化Regularization
批规范化（Batch Normalization，BN）：在minibatch维度上在每次训练iteration时对隐藏层进行归一化
标准化（Standardization）：对输入数据进行归一化处理
正则化（Regularization）：通常是指对参数在量级和尺度上做约束，缓和过拟合情况，L1 L2正则化

如果把 ![[公式]](https://www.zhihu.com/equation?tex=x+%5Cin+%5Cmathbb%7BR%7D%5E%7BN+%5Ctimes+C+%5Ctimes+H+%5Ctimes+W%7D) 类比为一摞书，这摞书总共有 N 本，每本有 C 页，每页有 H 行，每行 W 个字符。BN 求均值时，相当于把这些书按页码一一对应地加起来（例如第1本书第36页，第2本书第36页......），再除以每个页码下的字符总数：N×H×W，因此可以把 BN 看成求“平均书”的操作（注意这个“平均书”每页只有一个字)，求标准差时也是同理。

目的是为了得到所有图片在同一个通道的均值和方差，进而归一化

从公式看它们都差不多：无非是减去均值，除以标准差，再施以线性映射。

![image-20220109194840057](C:\Users\a\AppData\Roaming\Typora\typora-user-images\image-20220109194840057.png)

### yolo

锚点，全局，中心框偏移

# deepsort

## 卡尔曼滤波

借助模型和测量推断

[(7 封私信 / 80 条消息) 如何通俗并尽可能详细地解释卡尔曼滤波？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/23971601/answer/194464093)

## 匈牙利算法

[简单理解增广路与匈牙利算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/208596378)最大匹配

### 马氏距离

[马氏距离(Mahalanobis Distance) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/46626607)

### 余弦距离

[余弦距离介绍 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/108508605)

## ReID

[小白入门系列——ReID(一)：什么是ReID？如何做ReID？ReID数据集？ReID评测指标？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/83411679)

