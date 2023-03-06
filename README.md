[TOC]

# 项目作业：血液细胞检测Object Detection

## 一、项目简介

### 1.数据集介绍

####  数据集描述

`BCCD_dataset` 数据集共有三类364张图像，3个类别中有4888个标签（有0个空示例）。下图是网站得到的可视化数据（三个类别细胞标注数量计数）

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221129204602931.png" alt="image-20221129204602931" style="zoom: 33%;" />

对于此数据集， [nicolaschen1](https://github.com/nicolaschen1) 开发了两个Python 脚本来制作准备数据（CSV 文件和图片) ，用于在医学图像上识别血细胞异常。

- export.py: 它创建文件"test.csv"，其中包含所需的所有数据：filename, class_name, x1,y1,x2,y2。 
- plot.py: 它为每个图像绘制方框，并将其保存在 imagesBox 目录中。

一个样本中包括三种细胞（如右图所示）：

- RBC (Red Blood Cell，红细胞) 

- WBC (White Blood Cell，白细胞) 

- Platelets (血小板) 

  <img src="C:\Users\13521\Desktop\sxzt\4\BCCD_Dataset\imagesBox\BloodImage_00000.jpg" alt="BloodImage_00000" style="zoom: 50%;" />



####  数据集结构

```
├── BCCD
│   ├── Annotations
│   │       └── BloodImage_00XYZ.xml (364 items)  
│   ├── ImageSets       # 包括四个 Main/*.txt，分割了数据集为：test,train,trainval,val
│   └── JPEGImages
│       └── BloodImage_00XYZ.jpg (364 items) # 364个图像的jpg文件
├── dataset
│   └── mxnet           # mxnet的一些预处理脚本
├── scripts
│   ├── split.py        # 在ImageSets中生成四个.txt的脚本
│   └── visualize.py    # 用于生成像example.jpg这样有标签的img的脚本

├── example.jpg         # 一个由visualize.py生成的标有img的示例
├── LICENSE
└── README.md
```


* `ImageSets`


  * train 训练数据（均为图片名没有后缀）
  * val  验证数据
  * trainval 则是所有训练和验证数据
  * test 测试数据

* 对于 `JPEGImages`:

  * **图片类型** : *jpeg(JPEG)*
  * **Width** x **Height** : *640 x 480*

* `ImageSets/Main`：ImageSets文件夹下主要是Main文件夹中有四个文本文件test.txt、train.txt、trainval.txt、val.txt, 其中分别存放的是测试集图片的文件名、训练集图片的文件名、训练验证集合集的文件名、验证集图片的文件名；txt文件中每一行包含一个图片的名称，末尾会加上±1表示正负样本；

* `Annotations` :  每张图片打完标签所对应的XML文件，其中XML文件和图像文件名称一致（除后缀名）。对象检测的VOC格式 `.xml`，由标签工具自动生成。

### 2.项目要求

• 请根据各类细胞形状特性，检测图像中的各类细胞，标注区域（三色框）【plot.py】， 并对照标注文件给出准确率评价。

• 使用数据库所有文件进行评测，真值在“Annotation”中；

 • 用数字图像处理的基本方法，不能使用神经网络的方法做为实现，但是可以用神经网络做对比实验； 

## 二、基本思路

### 1.算法设计

#### 技术路线

目标检测的通用整体流程如图所示，传统方法是给定图片的输入，对候选框进行特征提取，利用分类器判定，利用NMS进行候选框的合并，得到输出结果；深度学习的方法采用特征提取和直接回归的方法进行目标区域的提取，直接回归的方式是用，再用NMS对目标框进行合并得到结果。受到目标检测算法的启发，我们为血液细胞检测任务来设计算法。

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221130105528223.png" alt="image-20221130105528223" style="zoom: 33%;" />

我们的主要思路是：

- 在图像中确定约1000-2000个候选框 (使用选择性搜索)。

- 每个候选框内图像块缩放至相同大小，进行HOG特征提取和PCA降维，得到特征向量。 

- 对候选框中提取出的特征向量，使用分类器（SVM多分类）判别是否属于一个特定类。

- 对于属于某一类特征的候选框，用NMS算法进一步调整其位置。

### 2.训练过程

#### 准备训练集

我们通过xml文件和原始图像分出已知类别的细胞图片，对应代码为`crop_subimage_based_on_xml.py`。我们取数据集中标签为train 和 val 的图片数据来作为SVM分类器的训练输入数据，根据xm文件提供的坐标信息，我们把训练集中的图像按照细胞类型分割为小的子图，分类保存在文件夹下面。其中每类87.5%作为训练集,12.5%作为预测集。以灰度图的形式读取图片，将图片重塑大小至256×256，最后给数据集的血小板，红细胞，白细胞分别打上1,2,3的标签，以便训练预测以及后续的性能评估。

#### 特征提取和降维

HOG特征是一种在计算机视觉和图像处理中用来进行物体检测的特征描述子。它通过计算和统计图像局部区域的梯度方向直方图来构成特征。最早出现在2005年CVPR上，法国的研究人员Navneet Dalal 和Bill Triggs利用HOG特征+SVM进行行人检测,在当时得到了较好的检测效果。主要流程如下:

![image-20221214164354271](C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221214164354271.png)

主要代码参考自 [基于HoG 的图像特征提取及其分类研究 - 代码天地 (codetd.com)](https://www.codetd.com/article/13626009#21	_11)，我们对处理后的图像提取HOG特征，再用PCA方法对特征向量进行降维。

HOG特征提取的步骤为：

1)采用不同的Gamma值(默认2.2)校正法对输入图像进行颜色空间的标准化，目的是调节图像的对比度,降低图像局部的阴影和光照变化所造成的影响,同时可以抑制噪音的干扰。

2)计算图像每个像素的梯度(包括大小和方向),主要是为了捕获轮廓信息,同时进一步弱化光照的干扰。本实验使用Sobel算子进行计算。

3)将图像划分成小cells,因考虑到设备的性能及维度过大问题,而本实验一个cell的大小为32×32个像素。统计每个cell的梯度直方图(不同梯度的个数)，形成每个cell的描述子。

4)对每个cell的梯度方向进行投票，本实验将cell的梯度方向180度分成9个方向块即9个bins，每个bin的范围为20，cell梯度的大小作为权值加到bins里面。

5)本实验将每四个cell组成一个block(2×2cell/block)，一个block内所有cell的特征描述子串联起来便得到该block的HOG特征描述子。

6)将图像image内的所有block的bins直方图向量串联起来就可以得到该图片的HoG特征向量。这一步中由于是串联求和过程,使得梯度强度的变化范围非常大。需要对block的梯度强度做归一化，归一化能够进一步地对光照、阴影和边缘进行压缩。

输入不同细胞图像后HOG特征向量可视化的结果如下：

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221214164459708.png" alt="image-20221214164459708" style="zoom: 200%;" />

![image-20221214164638397](C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221214164638397.png)

​    我们选择HOG特征的原因是，与其他的特征描述方法相比，HOG有很多优点。由于HOG是在图像的局部方格单元上操作，所以它对图像几何的和光学的形变都能保持很好的不变性，这两种形变只会出现在更大的空间领域上。其次，在粗的空域抽样、精细的方向抽样以及较强的局部光学归一化等条件下，目标细微的形变可以被忽略而不影响检测效果。

PCA（Principal Component Analysis）是一种常用的数据降维方法。PCA通过线性变换将原始数据变换为一组各维度线性无关的表示，将数据从原来的高维空间投影到低维空间来减少数据中属性数量，可用于提取数据的主要特征分量，由PCA创建的新属性捕获数据中最大的变化量。

PCA降维的步骤为：

在提取特征的步骤中,我们可以算出每一张图片提取出的特征向量的维度是2×2×9×7×7等于1764维，这是较大的维数，里面包含了许多冗余信息，所以我们应考虑对特征向量进行降维。本实验实现的是PCA(主成分分析)算法的降维,其主要过程如下:

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221214161549027.png" alt="image-20221214161549027" style="zoom:67%;" />

1)对所有样本进行中心化

2)计算样本的协方差矩阵  $xx^T $

3)对协方差矩阵  $xx^T $ 做特征值分解

4)取最大的$k$个特征值所对应的单位特征向量

#### 训练SVM分类器

本实验的分类器利用了sklearn库中的svm模型，设置不同的惩罚参数 (默认1.0) 和核函数 (默认rbf) 的类型进行多次训练和测试，对模型进行调优，训练完成后，保存模型参数。

本实验采用不同的惩罚程度、核函数、gamma校正值、特征维数的分类性能进行对比，采用了精确率、召回率和micro_F1分数对实验结果进行性能评估，在对参数进行调优的过程中，有如下结论：

- 模型随着惩罚函数（松弛变量）的减少训练集的正确率逐渐降低，这是因为对误分类的惩罚减小，允许容错，此时模型的泛化能力应增强。

- 在rbf、linear和poly三种核函数中，高斯核函数rbf的性能是最好的，这是因为训练的样本数n远远大于特征维数m，这也导致了linear性能最差；而多项式核函数poly的训练集的精确率也比较高，原因是多项式的阶数很高，网络变得更加复杂，但这也导致了过拟合。
- 随着gamma值从减小，模型性能减小，发现模型存在较为明显的过拟合问题。
- 特征维数增加，训练集精确率增加，性能减小，原因是随着特征维数的增大,保留了更多的冗余信息，导致模型过拟合。

HoG特征与SVM的结合之所以对数据分类有较好的原因是因为HoG特征有对光照的不敏感，即使存在部分遮挡也可检测出来、能够反映物体的轮廓，并且它对图像中的物体的亮度和颜色变化不敏感的优点，因此HoG特征适合行人检测和车俩检测等领域。如果HoG无法从含有狗和蛇的图片中有效提取它们的梯度和梯度方向，分类效果就会变差，是否可以通过对图片的预处理来提高精度，还需要后续的实验。

以上过程的代码写在`SVM`.py 文件中，具体的调参过程参见结果分析。

### 3.检测过程

#### 候选框选取

在two-stage目标检测算法中，一般先要产生候选区域(region proposal)。一般可以在图片上使用穷举法或者滑动窗口选出所有物体可能出现的区域框，对这些区域框提取特征并进行使用图像识别分类方法，得到所有分类成功的区域后，通过非极大值抑制输出结果。在图片上使用穷举法或者滑动窗口选出所有物体可能出现的区域框，就是在原始图片上进行不同尺度不同大小的滑窗，获取每个可能的位置。这样做的缺点是复杂度太高，产生了很多的冗余候选区域，而且由于不可能每个尺度都兼顾到，因此得到的目标位置也不可能那么准，在现实中不可行。

Selective search方法（选择性搜索）有效地去除冗余候选区域，使得计算量大大的减小。在选择性搜索中，首先将每个像素作为一组，然后计算每一组的纹理，将两个最接近的组结合起来，我们通常对较小的组先分组，合并区域知道所有区域都合并在一起。这种方法将图像作为输入，输出可能是包含目标对象子块的选择框。这些候选区域可能是嘈杂的、重叠的，并且可能不能完美地包含这个对象，但是在这些候选区域中，将会有一个非常接近于图像中的实际对象的候选区域。然后我们可以使用目标识别模型对这些候选区域进行分类。有分数最高的候选区域就是该物体的位置。

我们借鉴R-CNN的思想，使用选择性搜索算法提取目标区域，通过调用Open CV 中的库函数，选择性搜索算法思想来源于[selectiveSearchDraft.pdf (huppelen.nl)](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)，代码写在`selective_saerch.py`中，我们的输入为一张图像，输出为这张图像中的候选框四个坐标信息，保存在`ss_for_nms.csv`文件中，并根据坐标信息在原图中分割图像，将输出的子图保存在对应的文件夹中。

Selective search方法基本思路如下：

1、使用一种过分割手段，将图像分成很多小区域，生成区域集R。

2、初始化一个相似集合为空集S。

 3、计算区域集R中每个相邻区域的相似度（包括颜色、纹理、尺寸、交叠），放入集合S中，集合S保存的其实是一个区域对以及它们之间的相似度。

 4、查看现有小区域，合并可能性最高的两个区域（基于颜色、纹理等），找出S中相似度最高的区域对，将其合并为新区域，并从S中删除与它们相关的所有相似度和区域对。重新计算这个新区域与周围区域的相似度，放入集合S中，并将这个新合并的区域放入集合R中，重复这个步骤直到S为空。

5、从R中找出所有区域的包围该区域的最小矩形框，它们就是候选框。

如图展示的是一张图像通过我们的选择性搜索算法产生的候选框：

<center class="half">
    <img src="C:\Users\13521\Desktop\sxzt\4\object-detection-BCCD\data\JPEGImages\BloodImage_00000.jpg" alt="BloodImage_00000" style="zoom: 39%;" />
   <img src="C:\Users\13521\Desktop\sxzt\Figure_3.png" alt="Figure_3" style="zoom: 50%;" />
</center>


在这张图像中共生成了1475个候选框，根据返回的坐标信息对原图进行裁剪，保存在crop_ss_文件夹中，并将分出的各个候选框在原图中的坐标位置信息保存在ss_for_nms.csv文件中。再把这些数据送入分类器中进行检测，计算出每个图像对应的类别及其概率。

#### 使用模型对候选框分类

我们将根据selective search算法计算出来的每个图像的候选框保存为图片，并同时保存每个候选框的坐标信息。每张图片会常常产生几千个候选框，我们将这些候选框进行预处理，拉伸为统一尺寸（245*256），转换为灰度图像，并对其提取HOG特征向量，再对特征向量使用PCA降维，输入到已经训练好的SVM分类器中，输出每个图像预测的类别和置信度大小，置信度将作为NMS消除多余候选框的依据。每个候选框的置信度来源于模型的置信概率矩阵，根据样本与分类边界的距离远近，对其预测类别的可信程度进行量化，离边界越近的样本，置信概率越低，反之，离边界越远的样本，置信概率高。

对候选框进行提取特征和降维，送入前面已经训练好的分类器，得到候选框的类别，代码在`test.py`文件中，并输出每个候选框的置信度和分类的类别，保存在`type_confidence.csv`文件夹中。

分类结果如图所示：

![image-20221214214650769](C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221214214650769.png)

#### NMS降低候选框数量

NMS（Non-Maximun Suppression）非极大值抑制就是抑制不是极大值的元素。该方法主要是为了降低候选框数量，我们在之前提取出目标的候选框数量非常多（几千个），每个候选框经过分类器会有一个属于某个类别的概率值，为了消除多余的框，我们选取领域里分数最高的窗口。IOU的概念来源于数学中的集合，用来描述两个集合A和B之间的关系，它等于两个集合的交集里面所包含的元素个数除以并集里面所包含的元素个数，具体的计算公式：$\mathrm{IOU}=\frac{\mathrm{A} \cap \mathrm{B}}{\mathrm{A} \cup \mathrm{B}}$我们将用这个概念来描述两个框之间的重合度。

我们的目标分类任务有3类，以数据集0000号图片为例，我们在第一阶段得到1475个候选框，输出置信概率矩阵为1475*3，每列对应一类，每行是各个建议框的得分，有1475个，NMS算法步骤如下： 

1）对1475×3维矩阵中的每列按从大到小进行排序（概率值越大排名越靠前）；

2）从每列最大的得分候选框开始，分别与该列后面的候选框进行IOU计算，若IOU>给定阈值（如0.5），则剔除得分较小的候选框，剩余多个候选框我们认为图像中可能存在多个该类目标；

3）依次对得分越来越小的候选框重复步骤②，同样剔除IOU得分较小的候选框； 

4）重复步骤③直到遍历完该列所有建议框；

5）遍历完1475×20维矩阵所有列，即所有物体种类都做一遍非极大值抑制；

以上代码保存在 `nms.py` 中，考虑到细胞检测的特殊性，我们可以直接排除面积过小和过大的候选框，因为它们不符合细胞形状的一般大小。

保存在`nms_for_plot.csv` 文件中，

#### 绘制色框

在原图中根据子图的坐标信息绘制出三色框，代码文件是`myplot.py`，代码修改自数据集中自带的python文件，读取`nms_for_plot.csv` 的中的框的坐标位置信息，在对应的原图上画出标注框。

#### 准确率计算和评价

在多分类问题中，准确率（Accuracy）定义为：正确分类的样本个数占总样本个数，  A = (TP + TN) / N。对于具体的某个目标来讲，我们可以使用预测框与实际框的贴合程度来判断检测的质量，通常使用IoU（Intersection of Union）来量化贴合程度，作为衡量指标。我们选取一个阈值0.5，来确定预测框是正确的还是错误的，当两个框的IoU大于0.5时，我们认为预测框才是一个有效的检测，否则属于无效的匹配。

我们首先用`export.py`读取xml文件中图像正确的框的坐标信息和对应分类，并输入到csv文件中对照标注文件计算准确率`metrics.py`。评测需要每张图片的预测框的位置和实际框的位置信息，分别被我们保存在`nms_for_plot.csv` 和 `test.csv` 文件中。

算法流程如下：

1）对于某一个实例，遍历预测框，对于遍历中的某一个预测框，计算其与该图中同一类别的所有预测框的IoU，如果所有IoU小于阈值，则将当前预测框标记为误检框。

2）如果IoU大于阈值，还要看对应的实际框是否被访问过，如果前面已经有得分更高的预测框与该实际框对应了，即使现在的IoU大于阈值，也会被标记为误检框，如果没有被访问过，则将当前预测框标记为正检框，并将该实际框标记为访问过，以防止后面还有预测框与其对应。

3）在遍历完所有的预测框之后，我们会得到每一个预测框的属性，即正检框和误检框。

代码修改自 [目标检测算法之常见评价指标的详细计算方法及代码解析 - 简书 (jianshu.com)](https://www.jianshu.com/p/0244d76d9673) ，并经过人工核对正确检测的预测框，得到本次实验的准确率为0.2%，我们推测造成如此低的准确率的原因有以下几点：

- 分类器的检测精度不够

  - 图像在送入分类器时进行了拉伸与截断，损失了部分信息
  - 图像在送入分类器的时候包含背景和一些粘连细胞的信息，也会造成分类器的错误分类
  - 分类器存在过拟合的现象

- 选择性搜索和NMS算法产生的候选框不准确

  - 选择性搜索产生的错误候选框过多，使得准确率计算的分母变大。
  - 分类器的检测精度不够导致NMS的判断依据置信度不够可信

  - NMS需要手动设置阈值，阈值的设置会直接影响重叠目标的检测，太大造成误检，太小达不到理想情况。

## 三、结果分析

### 1.典型实验结果分析

#### 结果一

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221213225952829.png" alt="image-20221213225952829" style="zoom: 25%;" />



第一次训练出来的结果如图所示，可以看到，经过NMS算法后候选框数量得到了明显减少，约为200个左右，但是误检测出很多框，由于选择性搜索算法选出了很多无实体的框，导致特征向量不明显，最终都被分类器认为是红细胞，通过分析检测出来的错误的框，我们对模型做出了一些修改，通过增加一些阈值来限定判断。具体的阈值选择，我们通过对测试集的细胞尺寸和分类进行可视化，得到如下直观性结论（代码修改自CSDN网站）：

![image-20221213212801995](C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221213212801995.png)

正常情况下在一张图片中，白细胞出现的次数不超过3次，血小板出现的次数不超过7次。

![image-20221213213033005](C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221213213033005.png)

在细胞尺寸方面，由于细胞是近球形的，宽和高相差不大，血小板的宽高聚集在在40左右，红细胞宽高聚集在100-200之间，白细胞的尺寸最大且无明显的聚集趋势。

- 白细胞的尺寸通常比较大，因此可以从白细胞分类中剔除细胞尺寸低于某阈值的细胞，出现个数小于三次。
- 血小板的尺寸很小且数量较少，因此可以从血小板分类中设置细胞尺寸阈值，并设定一张图中最多的血小板个数，对于无意义的框造成的误检，设置血小板置信度低于某阈值的数据不可信。
- 红细胞尺寸设置阈值。

#### 结果二

<img src="C:\Users\13521\Desktop\sxzt\BloodImage_00000.jpg" alt="BloodImage_00000" style="zoom:50%;" />

经过一些条件的限制后，NMS算法产生的候选框更少了，但是检测的效果不佳，我们推测是由于训练集不够平衡导致的，将对SVM分类器模型进行修改：

- 使数据集更加平衡，增加白细胞和血小板训练集的个数。
- 人工排除一些分割不准确的测试集来增强模型的检测精确度。

因为待检测目标具有旋转不变性，就可以对目标做上下翻转、左右反转、90°*3 旋转等操作，如果目标中存在模糊的情况，在扩充的时候也可以适当做一些高斯模糊，如果不希望有形变，就统计下数据集中图像的最大长和最大宽边，做一张空白画布，然后其他图像在画布尺寸上进行等比例缩放，空白处填0处理。所以我们使用`image_expansion.py`文件对白细胞和血小板数据集进行扩充，扩充的图像是由原数据集旋转90°，180°，270°和上下左右翻转得到的。由扩充后的数据集，我们再次训练SVM分类器，得到新的模型保存为**model_5.pkl**（前4个为之前训练得到的模型），查看训练结果：

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221214111655783.png" alt="image-20221214111655783" style="zoom:33%;" />

预测模型分类的准确度很低（0.56），但是训练模型的score函数得分很高为（0.98，最好的得分是1.0），推测是模型产生了过拟合，解决过拟合的办法是为SVM引入了**松弛变量（slack variable）**，我们再尝试对数据集进行扩充，减少特征向量的维度，最终尝试出的模型最大得分为0.56。

#### 结果三

<img src="C:\Users\13521\Desktop\sxzt\4\object-detection-BCCD\imagesBox\BloodImage_00003.jpg" alt="BloodImage_00003" style="zoom: 50%;" />

经过一些优化之后，我们得到如图所示的结果，可以看到错误检测的数量很多，模型效果不佳。

分析原因有以下：

- HOG特征只关注了物体的边缘和形状信息，对目标的表观信息并没有有效记录，所以很难处理遮挡问题，而且由于梯度的性质，该特征对噪点敏感，手工设计的特征对于多样性的变化没有很好的鲁棒性。

- 在生物医学领域，血细胞本质上是复杂的，实际细胞图像检测的应用中会面临各种复杂的情况。传统的图像处理方法不能完全解决现存的细胞检测问题，特别在细胞重叠度高的区域其准确率上达不到要求。

### 2.模型的不足

本模型借鉴了RCNN的思想，存在以下明显的问题：

　　1）多个候选区域对应的图像需要预先提取，占用非常大的磁盘空间。

　　2）SVM需要固定尺寸的输入图像，crop/warp（归一化）产生物体截断或拉伸，会导致输入的信息丢失，模型准确率比较低。

　　3）每一个候选框都需要进入SVM分类器计算，上千个候选框存在大量的范围重叠，重复的特征提取带来巨大的计算浪费。

​		5）由于提取器包含的参数较少，并且人工设计的鲁棒性较低，因此特征提取的质量并不高，存在特征鲁棒性不够的问题。

这也反应了传统的目标检测算法存在两个主要缺陷：首先是在区域选择上，不能较为准确的估计出目标所在位置，并且对尺度变化较大的目标，不能得到召回率高的候选区域；其次在特征提取阶段，不能提取到高鲁棒性的有效目标特征，特别是传统方法中目标定位、特征提取和分类都是不同的算法，因而很难得到较满意的性能。

### 3.实验收获

​		近些年来，深度学习也得到了迅速发展，使得图像特征提取的效率大大提升，各种分类任务的正确率不断的刷新提升。而深度学习存在着较差的可解释性和海量数据需求的问题，与之相反的是，传统特征提取方法可视性非常强，且现有的卷积神经网络可能与这些特征提取方法有一定类似性，因为每个滤波权重实际上是一个线性的识别模式，与这些特征提取过程的边界与梯度检测类似。因此，虽然传统特征提取方法已经过时，但对传统特征提取方法的学习是不可少的。

​		本次实验是的算法流程是我参考了很多资料，一步步设计出来的，与以往常用神经网络实现端到端的实验不同，这次实验因为只能使用传统算法，步骤较为繁琐，需要我去思考每一步的输入输入和如何对它们进行有效的保存，尤其是选择性搜索算法得到的候选框的图片，达到的子图数量有45万多张，每次运行都要耗费很长的时间，我想删除一个子图文件夹发现需要一小时三十分钟才能删除完，因为花费了很多的时间在等待运行出结果。在用神经网络做实验的时候，分类效果和准确率都很高，我常常能收获极大的成就感，但是对于具体的实现却是不明所以，很多网络通过调用库函数就能解决问题。而这次实验我对整个流程都比较清晰，一共用了17天做完，但是出来的效果却不尽如人意，很多时候真的是挫败感满满，尽力修改模型也达不到预期的效果。本实验采用的算法对血液细胞分类的效果不好，但是我当发现血液细胞的候选框检测可以使用边缘检测算法来更加准确的定位候选框，分类的时候可以使用细胞的颜色和轮廓特征进行判断准确率相当高的时候，所剩的时间已经不多了，只能硬着头皮把不太好的模型做完。但是这次的检测流程还可以使用在行人检测，车辆检测等目标轮廓突出的目标检测领域上，应该会取得更好的效果。

​		下面我们将使用YOLOv7来做对比实验，体验一下深度学习的超强性能。

## 四、基于YOLOv7的对比实验

传统的目标检测算法需要手动设计特征，如HOG特征，基于传统分类器，步骤繁琐，准确度和实时性都比较差，但是深度学习的算法可以实现使用深度网络学习特征，直接回归候选框或者端到端的策略，准确度高且实时性好。

YOLO 系列见证了深度学习时代目标检测的演化，核心思想是将整张图片作为网络输入，通过端到端的方式直接在输出层回归出边界框的位置和边界框的类别。2016年，YOLOv1(You Only Look Once) 诞生，是继 R-CNN，Fast R-CNN 和 faster R-CNN 之后，Ross Girshick 针对 DL 目标检测速度问题提出的另一种框架，其核心思想是将两阶段（two stage）算法用一套网络的一阶段（one stage）算法替代，直接在输出层回归边界框的位置和所属类别。同年，Ross Girshick 吸收 fast R-CNN 和 SSD 算法，设计了 YOLOv2，在精度上利用一些训练技巧，在速度 上应用了新的网络模型 DarkNet19，在分类任务上采用联合训练方法，结合 wordtree 等方法，用 YOLOv2 的检测种类扩充到了上千种。

我们选择YOLOv7模型，YOLOv7 在 5 FPS 到 160 FPS 范围内，速度和精度都超过了所有已知的目标检测器，并在 GPU V100 上，30 FPS 的情况下达到实时目标检测器的最高精度 56.8% AP。我们参考了 [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors | Papers With Code](https://paperswithcode.com/paper/yolov7-trainable-bag-of-freebies-sets-new) 中的来源于 https://github.com/rkuo2000/yolov7 的代码来进行血液细胞的识别。整个实验过程的代码保存在`yolov7.ipynb`文件中。本实验使用的YOLOv7 是目前 YOLO 系列最先进的算法，YOLOv7 相同体量下比 YOLOv5 精度更高，速度更快（120%FPS）。 其大部分继承自 YOLOv5，包括整体网络架构、配置文件的设置和训练、 推理、验证过程等等。 此外，v7 也有不少继承自 YOLOR，包括不同网络的设计、超参数设置以及隐性只是学习的加入，在帧样本匹配时仿照了 YOLOX 的 SimOTA 策略。YOLOv7 包括 了近几年最新的策略：高效聚合网络、重参数化卷积、辅助头检测、缩放模型等。

### 1.训练数据准备

yolo7训练开源工具包需要的数据结构如下：

```
├── BCCD
│   ├── Annotations
│   │       └── BloodImage_00XYZ.xml (364 items)  
│   ├── ImageSets       # 包括四个 Main/*.txt，分割了数据集为：test,train,trainval,val
│   └── JPEGImages
│       └── BloodImage_00XYZ.jpg (364 items) # 364个图像的jpg文件
│   └──Labels          

```

### 2.模型训练

​		我们使用预训练的yolov7-tiny.pt，这个权重对于数据量比较小的模型，训练效果要更好。克隆源代码后，修改相应的yaml配置文件，调整模型的相关参数，使其适应我们使用的BCCD数据集，在训练过程中，发现BCCD数据标注导出时出现问题，因此我们对`train.py`，`utils/loss.py` ，`utils/datasets.py`文件进行了修改。
​    	我们使用的GPU为3060Ti，设置batch-size=8，设置batch-size时发现若设置为32或更高时，会出现显卡显存不够用的情况，因此将其设置为8。

epoch=50时的各项性能指标如图所示：

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221215123930812.png" alt="image-20221215123930812" style="zoom: 33%;" />

epoch=100时的各项性能指标如图所示：

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221215123737624.png" alt="image-20221215123737624" style="zoom: 33%;" />

epoch=200时的各项性能指标如图所示：

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221215123758635.png" alt="image-20221215123758635" style="zoom: 33%;" />

我们发现训练迭代次数超过50以后，损失函数已经收敛，因此认为迭代次数设置较高对模型性能影响不大，所以综合考虑将迭代次数设置为100。

### 3.模型评估

训练了100回合之后，将此时的模型作为最终的模型，使用测试集对模型进行评估，得到如下结果：

**混淆矩阵**

<img src="C:\Users\13521\Desktop\exp\confusion_matrix.png" alt="confusion_matrix" style="zoom:15%;" />

可以看到每个细胞正确检测出细胞类型的概率都达到了90%以上，模型性能非常好，但是大量的背景部分被错误检测为了红细胞，这其实是因为文件标注没有标注出全部的红细胞而导致的问题。

**P（准确率）曲线**

$P=\frac{T P}{T P+F P}$，即预测结果中真正的正例的比例。

在重合度为0.87的时候

<img src="C:\Users\13521\Desktop\exp\P_curve.png" alt="P_curve" style="zoom: 15%;" />

**R（召回率）曲线**

$R=\frac{T P}{T P+F N}$，即所有正例中被正确预测出来的比例。

<img src="C:\Users\13521\Desktop\exp\R_curve.png" alt="R_curve" style="zoom: 15%;" />

**PR召回率曲线**

PR曲线体现了准确性（precision）以及召回率（recall），理想情况下，准确率与召回率应都很高，但是一般情况下准确率高、召回率就低，召回率低、准确率就高，因此P-R 曲线越靠近右上角性能越好。

<img src="C:\Users\13521\Desktop\exp\PR_curve.png" alt="PR_curve" style="zoom:15%;" />

**F1曲线**

$F 1=\frac{2 \times P \times R}{P+R}$，F1分数是精准率与召回率的调和平均数，是一个权衡精准率与召回率指标的值，综合考虑了P值和R值。

<img src="C:\Users\13521\Desktop\exp\F1_curve.png" alt="F1_curve" style="zoom:15%;" />

标注文件如图所示：

<img src="C:\Users\13521\Desktop\exp\test_batch0_labels.jpg" alt="test_batch0_labels" style="zoom: 33%;" />

最终的模型检测结果如图：

<img src="C:\Users\13521\Desktop\exp\test_batch0_pred.jpg" alt="test_batch0_pred" style="zoom: 33%;" />

可以发现，使用YOLOv7来检测血液细胞的准确率非常高，处理图像边缘处的细胞，基本上所有的细胞都可以被正确的检测出来，导致了检测精度不能进一步提高的原因是：

- 检测图像中存在一些重复标注的候选框
- 标注文件的不规范标注，很多红细胞在标注文件上并没有标注出来，但是被却检测到了，经人工核对确实是红细胞，但是在测评是只能被算为误判。

## 五、参考资料

[1]李尧. 基于深度学习的目标检测算法研究[D].山西师范大学,2019.DOI:10.27287/d.cnki.gsxsu.2019.000781.

[1]马莉莉,王志明.血液细胞图像分割方法综述[J].农业网络信息,2008(10):28-30.

[1]项磊,徐军.基于HOG特征和滑动窗口的乳腺病理图像细胞检测[J].山东大学学报(工学版),2015,45(01):37-44+63.

[Shenggan/BCCD_Dataset: BCCD (Blood Cell Count and Detection) Dataset is a small-scale dataset for blood cells detection. (github.com)](https://github.com/Shenggan/BCCD_Dataset)

[(32条消息) 目标检测（Object Detection）_Alex_996的博客-CSDN博客_目标检测](https://blog.csdn.net/weixin_43336281/article/details/113059311)

[第十九节、基于传统图像处理的目标检测与识别(HOG+SVM附代码) - 大奥特曼打小怪兽 - 博客园 (cnblogs.com)](https://www.cnblogs.com/zyly/p/9651261.html)

[(32条消息) 如何使用 Roboflow 标注关键点_求则得之，舍则失之的博客-CSDN博客_关键点标注方法](https://blog.csdn.net/weixin_43229348/article/details/123502639?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-7-123502639-blog-123154453.pc_relevant_recovery_v2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-7-123502639-blog-123154453.pc_relevant_recovery_v2&utm_relevant_index=12)

[(32条消息) 用BCCD数据集学习rcnn家族（一）——介绍BCCD数据集及预处理_冰西瓜是生活动力的博客-CSDN博客_bccd数据集](https://blog.csdn.net/qq_20491295/article/details/109312771)

[(32条消息) 常用数据集格式介绍，自制，比例划分，图片集重组及其转换——VOC（持续更新）_我宿孤栈的博客-CSDN博客_voc数据集有多大](https://blog.csdn.net/qq_37346140/article/details/127469968?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-3-127469968-blog-110940489.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-3-127469968-blog-110940489.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=6)

[目标检测数据集PASCAL VOC详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/362044555)

[02-01 目标检测问题定义_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1wJ411P7uD?p=2&vd_source=552ab0690ae1fe359766d3a3fd8b34b0)

[(32条消息) 目标检测的图像特征提取之（一）HOG特征_zouxy09的博客-CSDN博客_hog特征](https://blog.csdn.net/zouxy09/article/details/7929348)

[(33条消息) 使用判别训练的部件模型进行目标检测 Object Detection with Discriminatively Trained Part Based Models_masikkk的博客-CSDN博客_我们想在图像的不同位置和比例上定义一个分数.这是通过使用特征金字塔来完成的,它](https://blog.csdn.net/masibuaa/article/details/17924671)

[计算机视觉：目标检测 先验框和模型 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/337479794)

[(33条消息) 目标检测（三）传统目标检测与识别的特征提取——基于HOG特征的目标检测原理_失了志的咸鱼的博客-CSDN博客_hog目标检测](https://blog.csdn.net/qq_40959462/article/details/124695675)

[目标检测系列之一（候选框、IOU、NMS） - 腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1632451)

[(33条消息) 目标检测项目中面对高分辨率图像的滑动窗口技术（一）（代码开源，超简便API封装，直接调用进行切图及保存）_小文大数据的博客-CSDN博客_滑动窗口目标检测](https://blog.csdn.net/weixin_46707493/article/details/126321787)

[(33条消息) Datawhale -- opencv学习 -- Hog特征+SVM分类（行人检测）_何小义的AI进阶路的博客-CSDN博客_hog+svm](https://blog.csdn.net/hzy459176895/article/details/107145116/)

[(33条消息) 『ML笔记』HOG特征提取原理详解+代码_布衣小张的博客-CSDN博客_hog代码](https://blog.csdn.net/abc13526222160/article/details/102574369)

[(33条消息) Adaboost算法原理（二分类及多分类）_kalath_aiur的博客-CSDN博客_adaboost 多分类](https://blog.csdn.net/kalath_aiur/article/details/105234675)

[第九节、人脸检测之Haar分类器 - 大奥特曼打小怪兽 - 博客园 (cnblogs.com)](https://www.cnblogs.com/zyly/p/9410563.html)

[(33条消息) HOG+SVM实现图像分类_Lemon_Yam的博客-CSDN博客_hog+svm](https://blog.csdn.net/steven_ysh/article/details/125541934?spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-18-125541934-blog-8869969.pc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-18-125541934-blog-8869969.pc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=19)

[(33条消息) XGBoost算法介绍_月落乌啼silence的博客-CSDN博客_xgboost](https://blog.csdn.net/qq_18293213/article/details/123965029)

[基于HoG 的图像特征提取及其分类研究 - 代码天地 (codetd.com)](https://www.codetd.com/article/13626009)  ！！！

[(33条消息) sklearn.svm.SVC()函数解析（最清晰的解释）_我是管小亮的博客-CSDN博客_svm.svc](https://blog.csdn.net/TeFuirnever/article/details/99646257)

[(33条消息) 选择性搜索算法(Selective Search)超详解（通俗易懂版）_迪菲赫尔曼的博客-CSDN博客_selective search](https://blog.csdn.net/weixin_43694096/article/details/121610856)

[(33条消息) 边框回归(Bounding Box Regression)详解_南有乔木NTU的博客-CSDN博客_bounding box regression](https://blog.csdn.net/zijin0802034/article/details/77685438?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-2-77685438-blog-87527316.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-2-77685438-blog-87527316.pc_relevant_aa&utm_relevant_index=3)

[目标检测 — two-stage检测 - 走看看 (zoukankan.com)](http://t.zoukankan.com/eilearn-p-9061816.html)

[Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation (thecvf.com)](https://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
