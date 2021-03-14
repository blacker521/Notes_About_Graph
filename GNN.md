---
typora-root-url: img
---

# [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

用于图结构的半监督学习的可扩展方法

GCN的最原始论文，基于图卷积
![](/g_1001.png)
![](/g_1002.png)
![](/g_1003.png)
![](/g_1004.png)
![](/g_1005.jpg)
![](/g_1006.png)
![](/g_1007.jpg)
![](/g_1008.jpg)
![](/g_1008.jpg)
![](/g_1007.png)

# [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

GraphSAGE

通过采样和聚合节点的本地邻域要素来生成嵌入

比GCN能够适应网络的变化，不用重新训练网络参数
**GraphSAGE是利用一组聚合函数进行学习，这些聚合函数可以从节点的邻域中学习到节点的特征信息，即使是新节点也可以通过其邻域信息进行学习**
利用节点的属性（文本属性）来生成节点嵌入
识别节点邻域的结构属性同时显示本地角色及其全局位置
![](/g_2001.jpg)

1. 首先对节点的一阶和二阶邻居节点进行采样

2. 然后根据聚合函数聚合邻居节点的特征

3. 最后得到节点的Embedding向量

  **聚合邻居信息，然后不断迭代**

  ![](/g_2002.png)

第一行是我们要计算的节点的特征输入

第二行是第一个for循环遍历深度，可以理解为网络的层数

第三行是第二个for循环是遍历图中的所有节点

第四行是从上一层神经网络中利用聚合函数聚合当前节点的新特征

第五行是将当前节点的特征和邻居特征拼接并经过一个全连接网络得到当前节点的新特征

第七行是归一化

第八行是通过k层GCN后进行输出

**用k-1层的节点邻居信息和自身信息更新k层节点的信息**

为了使用SGD，通过mini-batch的方法来正向和反向传播
![](/g_2003.jpg)
## Loss
  ![](/g_2004.png)
## 聚合器
  ![](/g_2005.png)

# [GraRep: Learning Graph Representations with Global Structural Information](https://www.researchgate.net/profile/Qiongkai-Xu/publication/301417811_GraRep/links/5847ecdb08ae8e63e633b5f2/GraRep.pdf)

使用全局结构信息学习图形表示

学习低维向量来表示图形中出现的顶点，将图形的全局结构信息集成到学习过程中

图形中的每个顶点都由一个低维向量表示，在该向量中可以准确捕获图形所传达的有意义的语句，关系和结构信息

具有不同K值（K阶邻居）的不同顶点之间的K步关系与图相关的有用的全局结构信息。

定义了不同的损失函数--> 捕获不同的k 阶本地关系信息(即不同的k）

优化每个模型,通过组合从不同模型中学到的不同表示来构造每个顶点的全局表示-->全局表示可以⽤作进⼀步处理的特征
  ![](/g_3001.png)

