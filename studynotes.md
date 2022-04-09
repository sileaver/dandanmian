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

# 关于点线距离的计算方法

之前一直以为计算点线距离需要写很复杂的表达式，利用点线距离公式去计算，直到我看到了一种精妙的方法。将点线距离转化为求以中心点，起点，终点所构成的三角形的高度，基于numpy库进行数学运算，得到点线距离。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722135043131.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI5OTU3NDU1,size_16,color_FFFFFF,t_70)

![image-20220409184643417](C:\Users\a\AppData\Roaming\Typora\typora-user-images\image-20220409184643417.png)

利用三角形面积相等原则，可以转换为

![image-20220409184723418](C:\Users\a\AppData\Roaming\Typora\typora-user-images\image-20220409184723418.png)

# 代码

import numpy as np

def point_distance_line(point,line_point1,line_point2):
	#计算向量
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance

point = np.array([5,2])
line_point1 = np.array([2,2])
line_point2 = np.array([3,3])
print(get_distance_from_point_to_line(point,line_point1,line_point2))
print(point_distance_line(point,line_point1,line_point2))

# cv2

## 高斯滤波

```
cv2.GaussianBlur
```

高斯滤波器的目的是减少图像中的噪声。

- 语法：GaussianBlur（src，ksize，sigmaX [，dst [，sigmaY [，borderType]]]）-> dst
  ——src输入图像；图像可以具有任意数量的通道，这些通道可以独立处理，但深度应为CV_8U，CV_16U，CV_16S，CV_32F或CV_64F。

- 这个函数使用一个称为高斯核的核函数，用于对图像进行归一化。

  ——dst输出图像的大小和类型与src相同。
  ——ksize高斯内核大小。 ksize.width和ksize.height可以不同，但它们都必须为正数和奇数，也可以为零，然后根据sigma计算得出。
  ——sigmaX X方向上的高斯核标准偏差。
  ——sigmaY Y方向上的高斯核标准差；如果sigmaY为零，则将其设置为等于sigmaX；如果两个sigmas为零，则分别从ksize.width和ksize.height计算得出；为了完全控制结果，而不管将来可能对所有这些语义进行的修改，建议指定所有ksize，sigmaX和sigmaY

# canny边缘检测

这是我们检测图像边缘的地方，它所做的是计算像素强度的变化(亮度的变化)在一个图像的特定部分。幸运的是，OpenCV使它变得非常简单。

cv2.Canny函数有3个参数，(img, threshold-1, threshold-2)。

- img参数定义了我们要检测边缘的图像。
- threshold-1参数过滤所有低于这个数字的梯度(它们不被认为是边缘)。
- threshold-2参数决定了边缘的有效值。
- 如果两个阈值之间的任何梯度连接到另一个高于阈值2的梯度，则将考虑该梯度。

```
lines = cv2.HoughLinesP(isolated, rho=2, theta=np.pi/180, threshold=100, np.array([]), minLineLength=40, maxLineGap=5)
```

这一行代码是整个算法的核心，它被称为霍夫变换(Hough Transform)，将孤立区域的白色像素簇转换为实际的线条。



- 参数1:孤立梯度

- 参数5:占位符数组

- 参数6:最小行长

- 参数7:最大行间距

- 霍夫线变换会返回一个lines，分别为拟合出直线的起点和终点。配合上cv2.line

- cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → img

  img，背景图
  pt1，直线起点坐标
  pt2，直线终点坐标
  color，当前绘画的颜色。如在BGR模式下，传递(255,0,0)表示蓝色画笔。灰度图下，只需要传递亮度值即可。
  thickness，画笔的粗细，线宽。若是-1表示画封闭图像，如填充的圆。默认值是1.
  lineType，线条的类型，
  可以将拟合出来的直线可视化，这就是车道线检测。

```
gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
```

cvtcolor-颜色空间转换函数

这里是为了配合canny边缘检测对原图进行灰度化。

由于边缘检测的概念挺复杂，在此不过多赘述。

```
mask=np.zeros_like(frame)
cv2.fillPoly(mask,polygons,(255,255,255))
masked_image=cv2.bitwise_and(frame,mask)
```

这三行代码可以用于提取掩码范围内的图片，bitwise_and，位与运算，取较小的那个值，因为白色全为255，所以位运算后必然全取原值，白色区域范围外的黑色范围始终为黑色。（0,0,0）在此不多解释，fillpoly（填充多边形）第三个参数color请务必用元组，不用元组你会后悔的。

mask：掩码

polygons：多边形坐标

```
cap = cv2.VideoCapture(r'D:\Vehicle-Detection-And-Speed-Tracking\Car_Opencv\闯红灯.mp4')
```

视频流获取

```
ret, frame = cap.read()
```

读取视频流，并返回结果。

```
frame=cv2.addWeighted(lane_image,0.8,line_image,1,1)
```

图像加权融合。
