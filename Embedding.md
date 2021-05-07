---
typora-root-url: img
---

# [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf%C3%AF%C2%BC%E2%80%BA)

通过在网络中随机游走捕获网络局部信息，将游走序列等效为句子，并用Skip-Gram学习Embedding向量，从而完成网络表示学习

**适应性：**由于网络是动态的，网络表示应适应网络的变化

**社区意识：**节点间的编码距离应能反应出成员之间的社会相似性

**低维度：**维度低占用内存小，泛化能力强

**连续性：**连续的表征不仅可以提供节点的可视化，也可以在分类时更加健壮性

### 随机游走

随机游走是个随机的过程，每次在网络中心行走时的选择都是随机的，无法基于过去的行为预测未来的行为。可以并行化，网络发生细微变化时只需要重新提取局部网络结构

### 语言模型

![d_1001](/d_1001.png)

![d_1001](/d_1002.png)

![d_1001](/d_1003.png)
![d_1001](/d_1004.png)

# [LINE:Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf%C3%AF%C2%BC%E2%80%BA)

LINE模型致力于将这种大型的信息网络嵌入到低维的向量空间中，且该模型适用于任何类型(有向、无向亦或是有权重)的信息网络。并提出了一种解决经典随机梯度下降限制的边缘采样算法，提高了推理的有效性和效率。可用于可视化，节点分类以及关系预测等方面。

DeepWalk采用分布式并行方式来训练模型，但不适合大型网络

以前的网络没有捕捉到节点间更多的关系，只有一阶邻居相似性

LINE提出了**二阶相似性**，不是通过节点间的强弱来判定的，而是**通过节点的共享邻域结构来确定相似性**。
![d_1001](/e_1001.png)
一方面，节点6和7之间的权值比较大，所以具有较高的一阶相似性，他们之间的嵌入向量距离比较近。
另一方面，5,6虽然没有联系，但是他们有很多共同邻居，二阶相似性高，所以他们的嵌入向量也应该近。

### 一阶邻居
![d_1001](/e_1002.png)

**经验分布和联合概率分布越近越好**

**一阶邻居只适合无向图**

### 二阶邻居

**用预测的经验分布去接近节点共现概率(条件概率)**

![d_1001](/e_1003.png)



### 分别训练一阶近似和二阶近似的模型，然后将其得到的Embedding连接起来




### 负采样

目标函数为:
![d_1001](/e_1004.png)
### 边的权重与负采样
![d_1001](/e_1005.png)
![d_1001](/e_1006.png)
![d_1001](/e_1007.png)
![d_1001](/e_1008.png)
![d_1001](/e_1009.png)
# [Node2vec: Scalable Feature Learning for Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5108654/)

采用有偏的随机游走算法并结合Skip-gram算法学习表示，通过超参数来设置搜索策略

同质性：属于同一集群的节点更加相似，如S1和U

结构等价性：两个具有相似结构的节点更加相似，如S6和U

# [struc2vec: Learning Node Representations from Structural Identity](https://arxiv.org/pdf/1704.03165.pdf)

定义了层次结构相似度,专注于节点结构性信息进行Embedding,在每层上用有偏的随机游走获得节点序列,用Word2Vec训练Embedding

关注**不同的节点在网络中的所处的角色**

DeepWalk或node2vec这一类的方法在判断节**点的结构是否等价**的分类任务上往往并不能取得好的效果。其根本原因在于网络中的节点具有**同质性**（homohily），即两个节点有边相连是因为它们有着某种十分相似的特征。因此在网络中相距比较近的节点在嵌入空间也比较近，因为他们有着共同的特征；而在网络中相距比较远的节点，则认为它们没有共同特征，因此在嵌入空间的距离也会比较远，尽管两个节点可能在局部的拓扑结构上是相似的。

一个好的可以反映节点结构特性的方法必须满足以下两个特征：

- 嵌入空间的距离得能反映出节点之间的结构相似性，两个局部拓扑结构相似的节点在嵌入空间的距离应该相近。
- 节点的结构相似性不依赖于节点或边的属性甚至是节点的标签信息。

算法可以分成四步：

1. 根据不同距离的邻居信息分别算出每个节点对的结构相似度，这涉及到了不同层次的结构相似度的计算。
2. 构建一个多层次的带权重网络M，每个层次中的节点皆由原网络中的节点构成。
3. 在M中生成随机游走，为每个节点采样出上下文。
4. 使用word2vec的方法对采样出的随机游走序列学习出每个节点的节点表示。

![](https://pic1.zhimg.com/80/v2-29623a06c6576f23d51d80a77e114c54_1440w.jpg)

- 每一层节点相同,一共k*V个节点
- 包含两种边的类型,**黑色**为第k层节点u与v的层次结构相似性,**橙色**为层间的权重

## 符号

![[公式]](https://www.zhihu.com/equation?tex=G%3D%28V%2CE%29) ：无向带权网络，V表示节点集合，E表示边集合。

![[公式]](https://www.zhihu.com/equation?tex=n%3D%7CV%7C) ：网络中的节点数。

![[公式]](https://www.zhihu.com/equation?tex=k%5E%2A) ：网络的直径，即网络中任意两点距离的最大值。

![[公式]](https://www.zhihu.com/equation?tex=R_k%28u%29) ：与节点u距离为k的节点集合，等同与以u为根的BFS树上第k层的节点集合。例如，![[公式]](https://www.zhihu.com/equation?tex=R_1%28u%29%E2%80%8B) 就是u的直接邻居。

![img](https://pic2.zhimg.com/80/v2-ca0e1f088af75f4cd4b107674477e029_720w.jpg)

![[公式]](https://www.zhihu.com/equation?tex=s%28S%29) ：对某个节点集合V中的节点按照度的从小到大顺序排序后形成的序列。

![[公式]](https://www.zhihu.com/equation?tex=f_k%28u%2Cv%29) ：考虑两个节点的k跳邻域（k-hop neighborhoods）时（小于等于k跳的所有邻居均要考虑），两个节点的在结构距离（structural distance）。表达式如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+f_%7Bk%7D%28u%2C+v%29%3D%26+f_%7Bk-1%7D%28u%2C+v%29%2Bg%5Cleft%28s%5Cleft%28R_%7Bk%7D%28u%29%5Cright%29%2C+s%5Cleft%28R_%7Bk%7D%28v%29%5Cright%29%5Cright%29++%5C%5C++%26+k+%5Cgeq+0+%5Ctext+%7B+and+%7D%5Cleft%7CR_%7Bk%7D%28u%29%5Cright%7C%2C%5Cleft%7CR_%7Bk%7D%28v%29%5Cright%7C%3E0+%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=f_k%28u%2Cv%29) 就是距离u，v相距为k的那些节点之间的结构距离。这是一个递归定义，![[公式]](https://www.zhihu.com/equation?tex=f_%7Bk-1%7D%28u%2Cv%29) 表示考虑k-1跳邻域时的距离，再加上只考虑k跳邻居的距离，就形成了k跳邻域的距离了，初始值![[公式]](https://www.zhihu.com/equation?tex=f_%7B-1%7D%3D0) 。

![[公式]](https://www.zhihu.com/equation?tex=g%28D_1%2CD_2%29) 表示两个有序的序列 ![[公式]](https://www.zhihu.com/equation?tex=D_1%2CD_2) 的距离。 ![[公式]](https://www.zhihu.com/equation?tex=s%28R_%7Bk%7D%28u%29%29%2C+s%28R_%7Bk%7D%28v%29%29%29%E2%80%8B) 分别表示与u，v距离为k的节点按照度大小排序后的度序列。

注意到![[公式]](https://www.zhihu.com/equation?tex=f_k%28u%2Cv%29)的计算是在![[公式]](https://www.zhihu.com/equation?tex=f_%7Bk-1%7D%28u%2Cv%29)上加上一个非负的值，因此**该函数关于k是一个单调不降的函数**。并且这个函数只有在两个节点同时存在k跳邻域的时候才有定义。

下面给出一个具体的例子：

![img](https://pic1.zhimg.com/80/v2-43c850fbe56246b952aef51a648945d4_720w.jpg)

## 设计思想

两节点度相同,结构就相似,若邻居还有相同的度,就更相似

### 节点u,v之间的距离

节点u,v在前k层的距离$f_k(u,v)$前k-1层的距离 ${+}$ k阶邻居的度序列的距离

![](https://pic1.zhimg.com/80/v2-e1960da0d24c8039de20b8cb4fef5854_1440w.jpg) 

- g()是DTW算法,计算两个不一致序列的相似度

### 相似度

距离越小,w越大
![](https://pic1.zhimg.com/80/v2-7c439c12ee8644071afeae6343341c6c_1440w.jpg)

### 归一化

### ![img](https://pic3.zhimg.com/80/v2-97930359afd1e81587565f8bf7e52b3a_1440w.jpg)

## 不同层之间的边权重

若u在当前层有很多相似的节点,则应该更往上一层,获得更细的信息去区分,所以k$k -> k + 1$的权重更大

![](https://pic2.zhimg.com/80/v2-2c2cda2e9988635d3e83ee1f09b01265_720w.jpg)

-  如果u在当前层有很多相似的节点,可能当前信息区分度不够，应该更往上一层，获得更多的信息去区分。
-  ![](https://www.zhihu.com/equation?tex=%5CGamma_k%28u%29%3D%5Csum_%7Bv%5Cin+V%7DI%28w_k%28u%2Cv%29+%3E+%5Cbar%7Bw_k%7D%29)表示节点u在第k层具有相似节点的数量,相似节点的定义为权重大于第k层平均边权的邻居

其中![[公式]](https://www.zhihu.com/equation?tex=%5CGamma_k%28u%29)表示第k层中，所有指向u的边中权重大于该层平均权重的数量。具体的式子：

![[公式]](https://www.zhihu.com/equation?tex=+%5CGamma_%7Bk%7D%28u%29%3D%5Csum_%7Bv+%5Cin+V%7D+%5Cmathbb%7B1%7D%5Cleft%28w_%7Bk%7D%28u%2C+v%29%3E%5Coverline%7Bw_%7Bk%7D%7D%5Cright%29++%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Cbar+w_k) 第k层所有边权的平均值。![[公式]](https://www.zhihu.com/equation?tex=%5CGamma_k%28u%29) 实际上表示了第k层中，有多少节点是与节点u相似的，如果u与很多节点都相似，说明此时一定处于低层次，考虑的信息太少，那么![[公式]](https://www.zhihu.com/equation?tex=%5CGamma_k%28u%29)将会很大，即![[公式]](https://www.zhihu.com/equation?tex=w%28u_k%2Cu_%7Bk%2B1%7D%29%3Ew%28u_k%2Cu_%7Bk-1%7D%29)，对于这种情况，就不太适合将本层中的节点作为上下文了，应该考虑跳到更高层去找合适的上下文，所以去喜高层的权重更大。

![img](https://pic4.zhimg.com/80/v2-759ad172c57fef8a82f1f838b9c5fb93_720w.jpg)

### 归一化

![](https://pic2.zhimg.com/80/v2-18140b3b08063bdd8c371cf7d041af5d_720w.jpg)

## 随机游走

设置随机游走步数、随走游走器个数、当前层游走的概率，对每个随机游走器：

1. 从第0层开始，先某一概率q决定是在当前层游走 还是改变层。
2. 若在当前层游走，则通过权重 ![[公式]](https://www.zhihu.com/equation?tex=P_k%28u%2Cv%29) 有偏游走。
3. 若改变层，通过中 ![[公式]](https://www.zhihu.com/equation?tex=P_k%28u_k%2C+u_%7Bk%2B1%7D%29+%2C+P_k%28u_k%2C+u_%7Bk-1%7D%29+) 决。定是往上层还是往下层
4. 将每一层上访问到的节点加到上下文中（不考虑所处层）。
5. 直至满足停止条件：达到设定的游走步数。

## DTW

DTW语音处理中衡量两个长度不一致的有序序列相似度比较经典的方法，是一种动态规划的方法，规整两个序列，使长度保持一致，再计算相似度。

算法思路：是将两个序列组成矩阵网格，寻找一条通过该网格中若干个点的路径，路径通过的格点即为两个序列进行计算的对齐的点。 这个条路径长度可以作为即为两个序列的相似度衡量

![](https://pic4.zhimg.com/80/v2-5540d4fb83ff7256bace19eae660d71f_720w.jpg)

- X、Y轴分别为长度不一致的u、v节点k阶邻居度序列index
- 单元格大数字d(i,j): 表示两个序列index对应元素的距离／相似度公式：

![](https://pic2.zhimg.com/80/v2-cda5294b93743c16741eade9e26c34e9_720w.jpg)

- - 相比L1,L2距离，用该距离公式的优点是 认为：度为100与101距离，比度各自为1与2的距离是不同的，差异更小

- 单元格小数字g(i,j)表示：每一格，若从左下角出发，最少要路径可以达到。

- - 只能朝下图所示三个方向游走：

![img](https://pic3.zhimg.com/80/v2-754a3df66a0b6e2b5c27256f1264acf2_720w.jpg)

- - 到g(i,j)的最短路径长度计算

对于两个序列A、B，对任意的![[公式]](https://www.zhihu.com/equation?tex=a+%5Cin+A%2Cb%5Cin+B)定义一个距离函数![[公式]](https://www.zhihu.com/equation?tex=d%28a%2Cb%29)表示a与b的距离，DTW想利用这样定义的距离函数找到序列A、B的最小距离。举个例子：

![img](https://pic4.zhimg.com/80/v2-2f7c52af5aa25038d9569a2801f215b7_720w.jpg)

> 设A=(1,1,3,5,8),B=(1,2,2,6)是两个已经排序过的度序列，给出距离函数的定义为![[公式]](https://www.zhihu.com/equation?tex=d%28a%2Cb%29%3D%5Cfrac%7Bmax%28a%2Cb%29%7D%7Bmin%28a.b%29%7D-1%E2%80%8B) ，那么对A、B中所有元素两两之间计算距离d，得到左图中红色的距离。接下来我们利用这个矩阵来计算序列A、B的距离。
> 计算用到的算法是动态规划，递推式为： ![[公式]](https://www.zhihu.com/equation?tex=g%28i%2C+j%29%3D%5Cmin+%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bl%7D%7Bg%28i-1%2C+j%29%2Bd%28i%2C+j%29%7D+%5C%5C+%7Bg%28i-1%2C+j-1%29%2B2+d%28i%2C+j%29%7D+%5C%5C+%7Bg%28i%2C+j-1%29%2Bd%28i%2C+j%29%7D%5Cend%7Barray%7D%5Cright.+%5C%5C) 比如要计算 ![[公式]](https://www.zhihu.com/equation?tex=g%283%2C2%29) ,那么首先找到 ![[公式]](https://www.zhihu.com/equation?tex=g%282%2C2%29%3D1%2Cg%282%2C1%29%3D0%2Cg%283%2C1%29%3D2%2Cd%283%2C2%29%3D0.5)，
> ![[公式]](https://www.zhihu.com/equation?tex=g%282%2C2%29%2Bd%283%2C2%29%3D1%2B0.5%3D1.5%E2%80%8B)
> ![[公式]](https://www.zhihu.com/equation?tex=g%282%2C1%29%2B2d%283%2C2%29%3D0%2B2%5Ctimes+0.5%3D1) ​
> ![[公式]](https://www.zhihu.com/equation?tex=g%283%2C1%29%2B+d%283%2C2%29%3D2%2B0.5%3D2.5)
> 因此 ![[公式]](https://www.zhihu.com/equation?tex=g%283%2C2%29%3Dmin%7B1.5%2C1%2C2.5%7D%3D1)，然后我们用一个箭头标注![[公式]](https://www.zhihu.com/equation?tex=g%283%2C2%29)是从![[公式]](https://www.zhihu.com/equation?tex=g%282%2C1%29)计算出来的。
> 通过这样逐个计算，直到下标到达A、B的最大长度，则![[公式]](https://www.zhihu.com/equation?tex=g%285%2C4%29)就是A、B这两个序列的距离。
> 观察到这样一个现象，当A=(1,1,3,5)时，其实与B的度序列非常相似，此时计算出来两个序列的距离为1.9（上图的![[公式]](https://www.zhihu.com/equation?tex=g%284%2C4%29)），而给A加了一项很大的值后，两个度序列的距离一下子就变成了3.9。
> 最后，为什么这么定义距离函数？看个例子，如果a=1，b=2，两者差值为1，![[公式]](https://www.zhihu.com/equation?tex=d%28a%2Cb%29%3D1)；而若a=100，b=101，两者差值仍然为1，此时![[公式]](https://www.zhihu.com/equation?tex=d%28a%2Cb%29%3D0.01)。这说明后者比前者距离更小，更相似。