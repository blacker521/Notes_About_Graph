# Pytorch

## **Parameter**(torch.Tensor())

首先可以把这个函数理解为类型转换函数，将一个不可训练的类型`Tensor`转换成可以训练的类型`parameter`并将这个`parameter`绑定到这个`module`里面(`net.parameter()`中就有这个绑定的`parameter`，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个`self.v`变成了模型的一部分，成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

```python
Parameter(torch.Tensor(in_features, out_features))
```

## nn.Sequential()

torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。另外，也可以传入一个有序模块。

```python
# Sequential使用实例

model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

# Sequential with OrderedDict使用实例
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
```

## c.detach()

detach就是**截断反向传播的梯度流**

## np.random.permutation(nb_nodes) 

随机打乱数组