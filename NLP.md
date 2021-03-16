---
typora-root-url: img

---

# [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

Word2Vec原文

提出了两种新的模型结构，用于计算非常大数据集中单词的连续矢量表示。这些表示的质量是在一个词相似性任务中测量的，并将结果与以前基于不同类型神经网络的最佳表现技术进行比较。

1. Word2Vec两个算法模型的原理是什么，网络结构怎么画？
2. 网络的输入输出是什么？隐藏层的激活函数是什么？输出层的激活函数是什么？
3. Loss？
4. Word2Vec如何获取词向量？
5. 参数如何更新？
6. 分层的softmax怎么做的？二叉树，左右两个概率
7. Word2Vec的参数有哪些？0 
8. 局限性？

## 编码方式

### One-Hot

以字符变量的种类作为向量长度，向量中仅有一个元素为1，其余为0，数据稀疏，不适合作为网络的输入，且不能显示词与词之间的关系

### 分布式编码

把字符变量映射到固定长度的向量中，向量空间中的表示字符，且字符间的距离是有意义的，越相似，越相近

![](/n_1001.png)

COBW用上下文预测当前单词，Skip-Gram用当前次预测上下文

![](/n_1002.png)

**网络的输入是One-Hot向量wk，隐藏层没有激活函数，输出层有Softmax函数，输出的是概率分布，预测目标为One-Hot向量wj。层与层之间是全连接的**

![](/n_1003.png)

![](/n_1004.png)

## CBOW

![](/n_1005.png)

## Skip-Gram

![](/n_1006.png)
![](/n_1007.png)

## 分层的Softmax

使用哈夫曼二叉树的**叶节点**来表示语料库的所有单词
![](/n_1008.png)
![](/n_1009.png)
![](/n_1010.png)
![](/n_1011.png)

## 负采样

以一定的概率选取负样本，使得每次迭代只需要修改一部分参数，给定一些变量及其概率，随机采样使得其满足变量出现的概率。
节省了计算量
保证了模型训练的效果，其一模型每次只需要更新采样的词的权重，不用更新所有的权重，那样会很慢，其二中心词其实只跟它周围的词有关系，位置离着很远的词没有关系，也没必要同时训练更新
negative sampling 每次让一个训练样本仅仅更新一小部分的权重参数，从而降低梯度下降过程中的计算量。
如果 vocabulary 大小为1万时， 当输入样本 ( "fox", "quick") 到神经网络时， “ fox” 经过 one-hot 编码，在输出层我们期望对应 “quick” 单词的那个神经元结点输出 1，其余 9999 个都应该输出 0。在这里，这9999个我们期望输出为0的神经元结点所对应的单词我们为 negative word. negative sampling 的想法也很直接 ，将随机选择一小部分的 negative words，比如选 10个 negative words 来更新对应的权重参数。
![](/n_1012.png)

# [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ)

采用的是双向的Transformer的Econder结构

通过预训练模型可以显著提高NLP的下游任务

现有的模型都是单向的语言模型，无法充分了解单词所在的上下文结构。

BERT受完型填空的启发，使用随机屏蔽一些单词。然后让模型根据上下文来预测被遮挡的单词。

BERT是真正的结合上下文进行训练，而ELMoz只是左右分别训练

还提出了一种，下一个句子预测的任务预训练文本对,将token级提升到句子级以应用不同种类的下游任务
![](/n_2001.png)

Pre-training在未标记的数据上进行**无监督学习**

Fine-tuning阶段,BERT首先利用预训练得到的参数初始化模型,然后利用下游任务标记好的数据进行**有监督学习**,并对所有参数进行微调.所有下游任务都有单独的Fine-tuning模型,即使是使用相同预训练参数

![](/n_2002.png)

BERT是双向的Transformer Enconder

## 输入/输出

![](/n_2003.png)

**Token Embeddings **采用的 **WordPiece Embedding**, 共有 30000 个 token。每个 sequence 会以一个特殊的classification token [CLS] 开始，同时这也会作为分类任务的输出; 句子间会以 special token [SEP] 进行分割。**WordPiece Embedding:** $\mathrm{n}$ -gram 字符级 Embedding，采用 BPE 双字节编码，可以将单词拆分，比如"Ioved" "Ioving" "loves" 会拆分成“lov", "ed", "ing", "es".

**Segment Embedding** 也可以用来分割句子，但主要用来区分句子对。Embedding A 和 Embedding B 分别代表左右句子，如果是普通的句子就直接用 Embedding $\mathrm{A}$

**Position Embedding** 是用来给单词定位的，直接使用 one-hot 编码。BERT 最终的 input 是三种不同的 Embedding 直接相加.

## 预训练

BERT 采用两种非监督任务来进行预训练，一个是 token-level 级别的 Masked LM, 一个是 sontence-level 级别的Next Sentence Prediction

两个任务同时训练,所以BERT的损失函数是两个任务的损失函数想加

### Masked LM

双向会导致数据泄露的问题,即模型可以间接看到要预测的单词

Masked LM:随机屏蔽一些token并通过上下文预测这些token,在实验过程中,BERT会随机屏蔽每个序列中的15%的token并用[MASK]来代替

但这样会带来一个新的问题：[MASK] token 不会出现在下游任务中。为了缓解这种情况，谷歌的同学采用以下三种方式来代替 [MASK] token:

- $80 \%$ 的 [MASK] token 会继续保持 [MASK];
- $10 \%$ 的 [MASK] token 会被随机的一个单词取代;
- $10 \%$ 的 [MASK] token 会保持原单词不变 (但是还是要预测) 

最终 Masked ML 的损失函数是只由被 [MASK] 的部分来计算。
这样做的目的主要是为了告诉模型 [MASK] 是噪声，以此来忽略标记的影响。

### Next Sentence Prediction

为了解决这个问题，谷歌的同学训练了一个 sentence-level 的分类任务。具体来说，假设有 $\mathrm{AB}$ 两个句对，在训练过程 $50 \%$ 的训练样本 $\mathrm{A}$ 下句接的是 $\mathrm{B}$ 作为正例; 而剩下 $50 \%$ 的训练样本 $\mathrm{A}$ 下句接的是随机一个句子作为负例。并且通过 classification token 连接 Softmax 输出最后的预测概率。

**Input** $=$[CLS] the man went to $[\mathrm{MASK}]$ store $[\mathrm{SEP}]$
				he bought a gallon [MASK] milk [SEP]
**Label **$=$ IsNext

**Input** $=$ (CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight #less birds [SEP]
**Label** $=$ NotNext

## Fine-tuning

![](/n_2004.png)

预训练的Transformer已经完成了句子和句子对的表示学习,针对不同的下游任务,可以将具体的输入和输出适配到BERT中,并采用端到端的训练去微调模型参数

- a,b是句子级的任务,类似句子分类,情感分析等,输入句子或者句子对,在[CLS]位置接入Softmax输出Label
- c是token级别的任务,比如QA问题,输入问题和段落,在Paragraph对应输出的hidden vector后接上两个softmax,分别训练出Span的Start index和End index作为Question答案
- d是token级的任务,实体命名,接上softmax层可以输出具体的分类

# [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

**Transformer**,完全基于Attention机制

基于RNN的Seq2Seq模型难以处理长序列句子,因此顺序性也无法并行处理,基于CNN的Seq2Seq模型虽然可以实现并行,但消耗内存

Self-Attention是一种将序列不同位置联系起来并计算序列表示的注意力机制

Transformer完全使用Self-Attention

## Seq2Seq

Seq2Seq可以理解为输入一个序列,然后经过一个黑盒可以得到另一个序列

![](https://mmbiz.qpic.cn/mmbiz_gif/2QhWuEVMoTKpqMXT7rJMXR6kWsz4Yx2CmDRw1m5WOyBBibEfbEy1CIaKY0BoKUsF05zk5UA8gJvBf9ccyv1NPNg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

黑盒中是Encoder-Decoder框架.输入体一个序列,然后编码器进行编码得到上下文信息C然后通过解码器逐一解码,得到另一个序列

- 输入/输出序列都是Embedding向量
- 上下文信息C是一个向量,其维度与编码器的数量有关,256,512
- 逐一解码,解码器根据上下文C和先前生成的历史信息生成此时刻的输出
  ![](https://mmbiz.qpic.cn/mmbiz_gif/2QhWuEVMoTKpqMXT7rJMXR6kWsz4Yx2CbKkUFmiaibpc0GcuiaCN7S8dPWxBj3uSDofrMnIv7vFticxkKYX7TpzwiaA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

## 传统RNN

![](https://mmbiz.qpic.cn/mmbiz_gif/2QhWuEVMoTKpqMXT7rJMXR6kWsz4Yx2CS6v9vcjLt0AEatPFqfN3me6X4WVG0I2wB8JOznTYphLmZZBvxic2byA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

在编码器之间进行传递的其实是隐藏层的状态

![](https://mmbiz.qpic.cn/mmbiz_gif/2QhWuEVMoTKpqMXT7rJMXR6kWsz4Yx2CjHnHUEX37gD0zE2Bs5c2zVMP75MXTdL7UWibJdBrBuDcB2bQsQHvDmQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)\

编码器中最后一个 RNN 的隐藏状态就是要传给解码器的上下文信息 Context。

## Attention

对于Encoder-Decoder框架,上下文C的质量决定着模型的性能.RNN无法处理长序列的问题
Attention允许模型根据需要来关注输入序列的某些相关部分
Attention主要应用于Seq2Seq的解码器中.
![](/n_3003.png)
注意力机制使解码器能够子啊生成英语翻译之前将注意力集中在单词etudiant上

### Attention与Seq2Seq主要区别

- 更多的Context信息:编码器不传递编码阶段的最后一个隐藏状态.而是将所有的隐藏状态传递给解码器
- 解码时加入Attention

![](https://mmbiz.qpic.cn/mmbiz_gif/2QhWuEVMoTKpqMXT7rJMXR6kWsz4Yx2CPPJgmibtBGQ0YzjN2jpdibURI9hP32iaTspfvteqdTA3vHqWcPc22CsRw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

解码器中加入Attention的具体步骤

1. 查看编码器隐藏状态的集合（每个编码器隐藏状态都与输入句子的某个单词有很大关联）；

2. 给每个隐藏状态打分（计算编码器的隐藏状态与解码器的隐藏状态的相似度）；

3. 将每个隐藏状态打分通过的 Softmax 函数计算最后的概率；

4. 将第 3 步计算的概率作为各个隐藏状态的权重，并加权求和得到当前 Decoder 所需的 Context 信息。

   这种 Attention 操作在解码器每次解码的时候都需要进行

![](https://mmbiz.qpic.cn/mmbiz_gif/2QhWuEVMoTKpqMXT7rJMXR6kWsz4Yx2Cm3ehwsMtUdMA1ic30WHrIxlPib1a1YtAaQ4lRBCDmZCcXiamVPaReLP0A/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

Attention 的工作流程：

1. 解码器中的第一个 RNN 有两个输入：一个是表示 <END> 标志的 Embedding 向量，另一个来自解码器的初始隐藏状态；
2. RNN 处理两个输入，并产生一个输出和一个当前 RNN 隐藏层状态向量（h4），输出将直接被舍去；
3. 然后是 **Attention 操作**：利用**编码器传来的隐藏层状态集合**和刚刚得到 RNN 的隐藏层状态向量（h4）去计算当前的上下文向量（C4）；
4. 然后拼接 h4 和 C4，并将拼接后的向量送到前馈神经网络中；
5. 前馈神经网络的到的输出即为当前的输出单词的 Embedding 向量；
6. 将此 RNN 得到的单词向量并和隐藏层状态向量（h4），作为下一个 RNN 的输入，重复计算直到解码完成。

![](https://mmbiz.qpic.cn/mmbiz_gif/2QhWuEVMoTKpqMXT7rJMXR6kWsz4Yx2CibPOpAGbRySs3P3x3nwPq0IicMSDTzmsJoF22icj4ibmJAE08Bdv1peuVw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

![](https://mmbiz.qpic.cn/mmbiz_gif/2QhWuEVMoTKpqMXT7rJMXR6kWsz4Yx2CpeB8agV2UhHbZObT07oHZEJiaU5GA61f7lcB3jHTMtPs1f2FBNb7gOQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

左边是隐藏层状态的集合，右边是对隐藏的一个加权结果。

这里的模型并不是盲目地将输出中的第一个单词与输入中的第一个单词对齐，事实上，它从训练的时候就已经学会了如何排列语言对中的单词。下面再给出 Attention 论文中的一个例子（模型在输出 “European Economic Area” 时，其与法语中的顺序恰好的是相反的）：

## Transformer

### 模型结构

![](https://mmbiz.qpic.cn/mmbiz_png/2QhWuEVMoTKpqMXT7rJMXR6kWsz4Yx2CMs7YIwNqWNBibnP77NawyibtY0Jia7Z8EiaKGl3icZR77icl2WticDmHJD3kw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

编码部分是堆了六层编码器，解码部分也堆了六个解码器。

![](https://mmbiz.qpic.cn/mmbiz_png/2QhWuEVMoTKpqMXT7rJMXR6kWsz4Yx2Cuvdcy3CR81axNTb7DlGhxgbnCToD1MibMr8bfib5ur7S0kCb1J4I0CKQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

所有的编码器在结构上都是相同的，每一个都被分成两个子层

![](http://jalammar.github.io/images/t/Transformer_encoder.png)

编码器的输入首先经过一层 Self-Attention，这一层主要用来帮助编码器在对特定的单词进行编码时查看输入句子中的其他单词，后面我们会仔细研究 Self-Attention。

Self-Attention 层的输出进入给前馈神经网络（Feed Forward Neural Network，以下简称 Feed Forward），所有前馈神经网络的结构都相同并且相互独立。

解码器拥有三层，除了 Self-Attention 和 Feed Forward 外，还有一层 Encoder-Decoder Attention 层（以下简称 Attention 层，区别于 Self-Attention），Attention 层位于 Self-Attention 和 Feed Forward 层之间，主要用来帮助解码器将注意力集中在输入语句的相关部分（类似于 Seq2Seq 模型中的 Attention）。

![](http://jalammar.github.io/images/t/Transformer_decoder.png)

##  Encoder Side编码器

在 Self-Attention 层中，这些单词之间存在依赖关系；但 Feed Forward 层没有依赖，所以可以在 Feed Forward 层并行化训练。

### Self-Attention

![](http://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

谷歌论文中首次提出 Self-Attention 概念，我们来看一下它是怎么工作的。

假设我们现在要翻译下面的这个这句话：

“The animal didn't cross the street because it was too tired”

这个句子中的 it 指的是什么呢？是指 street 还是 animal 呢？这对人来说比较简单，但是对算法来说就没那么简单了。

而 Self-Attention 就可以解决这个问题，在处理 it 时会将它和 animal 联系起来。

Self-Attention 允许当前处理的单词查看输入序列中的其他位置，从而寻找到有助于编码这个单词的线索。

以下图为例，展示了第五层的编码器，当我们在编码 it 时，Attention 机制将 The animal 和自身的 it 编码到新的 it 上。

**第一步**，我们对于每个单词来说我们都一个 Embedding 向量，下图绿色部分。

此外，对于每个单词我们还会有三个向量分别为：查询向量（Querry）、键向量（Key）和值向量（Value），这些向量是单词的 Embedding 向量分别与对应的查询矩阵 $W^Q$、键矩阵 $W^K$和值矩阵$W^V$相乘得来的。

![](http://jalammar.github.io/images/t/transformer_self_attention_vectors.png)

对于 Embedding 向量来说是 512 维，而对于新向量来说是 64 维。（大小是架构设定的。）

**第二步**，我们来看下 Self-Attention 是如何计算的。

假设我们正在计算 Thinking 的 Self-Attention。我们需要将这个单词和句子中的其他单词得分，这个分数决定了我们将一个单词编码到某个位置时，需要将多少注意力放在句子的其他部分。

这个得分是通过当前单词的 Querry 向量和其他词的 Key 向量做内积得到的。

（也可以理解为当前单词的是由句子的所有单词加权求和得到的，现在计算的是当前单词和其他单词的分数，这个分数将用于后面计算各个单词对当前单词的贡献权重。）

![](http://jalammar.github.io/images/t/transformer_self_attention_score.png)

**第三步**，将这个分数处以 8（Value 向量是 64 维，取平方根，主要是为了稳定梯度）。然后将分数通过 Softmax 标准化，使它们都为正，加起来等于1。

![](http://jalammar.github.io/images/t/self-attention_softmax.png)

经过 Softmax 后的分数决定了序列中每个单词在当前位置的表达量（如，对着 Thinking 这个位置来说，= 0.88 * Thinking + 0.12 * Machines）。Softmax 分数越高表示与当前单词的相关性更大。

**第四步**，将每个单词的 Value 向量乘以 Softmax 分数并相加得到一个汇总的向量，这个向量便是 Self-Attention 层的输出向量。

![](http://jalammar.github.io/images/t/self-attention-output.png)

得到的这个向量会输送给下一层的 Feed Forward 网络。

在实际的实现过程中，为了快速计算，我们是通过矩阵运算来完成的。

首先是输入矩阵与查询矩阵、键矩阵和值矩阵。

![](http://jalammar.github.io/images/t/self-attention-matrix-calculation.png)

然后用 softmax 计算权重，并加权求和：

![](http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)

### **Multi-Head Attention**

论文中模型框架中画的 Multi-Head Attention 层，Multi-Head Attention 其实就是包含了多个 Self-Attention 。所以 Multi-Head Attention 有多个不同的查询矩阵、键矩阵和值矩阵，**为 Attention 层提供了多个表示空间**。

![](http://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)

Transformer 中每层 Multi-Head Attention 都会使用八个独立的矩阵，所以也会得到 8 个独立的 Z 向量：

![](http://jalammar.github.io/images/t/transformer_attention_heads_z.png)

但是 Feed Forward 层并不需要 8 个矩阵，它只需要一个矩阵（每个单词对应一个向量）。所以我们需要一种方法把这8个压缩成一个矩阵。这边还是采用矩阵相乘的方式将 8 个 Z 向量拼接起来，然后乘上另一个权值矩阵 W ，得到后的矩阵可以输送给 Feed Forward 层。

![](http://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)

### 总模型

![](http://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

###  **Positional Encoding**位置编码

模型可以完成单词的 Attention 编码了，但是目前还只是一个词袋结构，还需要描述一下单词在序列中顺序问题。

为了解决这个问题，Transformer 向每个输入的 Embedding 向量添加一个位置向量，有助于确定每个单词的绝对位置，或与序列中不同单词的相对位置：

![](http://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png)

这种方式之所以有用，大概率是因为，将配置信息添加到 Embedding 向量中可以在 Embedding 向量被投影到Q/K/V 向量后，通过 Attention 的点积提供 Embedding 向量之间有效的距离信息。

举个简单的例子，以 4 维为例：

![](http://jalammar.github.io/images/t/transformer_positional_encoding_example.png)

下图显示的是，每一行对应一个位置编码向量。所以第一行就是我们要添加到第一个单词的位置向量。每一行包含512个值——每个值的值在1到-1之间（用不同的颜色标记）。

![](http://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png)

这张图是一个实际的位置编码的例子，包含 20 个单词和 512 维的 Embedding 向量。可以看到它从中间一分为二。这是因为左半部分的值是由一个 Sin 函数生成的，而右半部分是由另一个 Cos 函数生成的。然后将它们连接起来，形成每个位置编码向量。这样做有一个很大的优势：他可以将序列扩展到一个非常的长度。使得模型可以适配比训练集中出现的句子还要长的句子。

### Residuals残差

每个编码器中的每个子层（Self-Attention, Feed Forward）都有一个围绕它的虚线，然后是一层 ADD & Normalize 的操作。
![](http://jalammar.github.io/images/t/transformer_resideual_layer_norm.png)

这个虚线其实就是一个残差，为了防止出现梯度消失问题。而 Add & Normalize 是指将上一层传过来的数据和通过残差结构传过来的数据相加，并进行归一化：
![](http://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png)
同样适用于解码器的子层：
![](http://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

## Decoder Side解码器

编码器首先处理输入序列，然后将顶部编码器的输出转换成一组 Attention 矩阵 K 和 V，这两个矩阵主要是给每个解码器的 ”Encoder-Decoder Attention“ 层使用的，这有助于解码器将注意力集中在输入序列中的适当位置：

![](http://jalammar.github.io/images/t/transformer_decoding_1.gif)

像我们处理编码器的输入一样，我们将输出单词的 Embedding 向量和位置向量合并，并输入到解码器中，然后通过解码器得到最终的输出结果。

在解码器中，Self-Attention 层只允许注意到输出单词注意它前面的单词信息。在实现过程中通过在将 Self-Attention 层计算的 Softmax 步骤时，屏蔽当前单词的后面的位置来实现的（设置为-inf）。

解码器中的 “Encoder-Decoder Attention” 层的工作原理与 “Multi-Head Attention” 层类似，只是它从其下网络中创建查询矩阵，并从编码器堆栈的输出中获取键和值矩阵（刚刚传过来的 K/V 矩阵）。

![](http://jalammar.github.io/images/t/transformer_decoding_2.gif)

###  **Softmax Layer**

解码器输出浮点数向量，我们怎么把它变成一个单词呢？

这就是最后一层 Linear 和 Softmax 层的工作了。

Linear 层是一个简单的全连接网络，它将解码器产生的向量投影到一个更大的向量上，称为 logits 向量。

假设我们有 10,000 个不同的英语单词，这时 logits 向量的宽度就是 10,000 个单元格，每个单元格对应一个单词的得分。这就解释了模型是怎么输出的了。

然后利用 Softmax 层将这些分数转换为概率。概率最大的单元格对应的单词作为此时的输出。

![](http://jalammar.github.io/images/t/transformer_decoder_output_softmax.png)

## **Training**训练

训练时我们需要一个标注好的数据集。

为了形象化，我们假设词汇表只包含 5 个单词和一个结束符号：

![](http://jalammar.github.io/images/t/vocabulary.png)

然后我们给每个单词一个 One-Hot 向量：

![](http://jalammar.github.io/images/t/one-hot-vocabulary-example.png)

假设我们刚开始进行训练，模型的参数都是随机初始化的，所以模型的输出和期望输出有所偏差。

![](http://jalammar.github.io/images/t/transformer_logits_output_and_label.png)

我们计算两者的损失函数并通过反向传播的方式来更新模型。

![](http://jalammar.github.io/images/t/output_target_probability_distributions.png)

在一个足够大的数据集上对模型进行足够长的时间的训练之后，我们希望生成的概率分布是这样的：

![](http://jalammar.github.io/images/t/output_trained_model_probability_distributions.png)

当训练好的得到模型后，我们需要为某个句子进行翻译。有两种方式确定输出单词：

- Greedy Decoding：直接取概率最大的那个单词；
- Beam Search：取最大的几个单词作为候选，分别运行一次模型，然后看一下哪组错误更少。这里的超参成为 Beam Size。