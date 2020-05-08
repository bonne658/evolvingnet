# evolvingnet
参照caffe和darknet，手写一个深度学习框架
## 1.设计
- Layer层记录输出作为下一层的输入
- Layer层记录delta作为反向传播的“介质”
- Layer层记录输入用于反向传播时更新权重
## 2.难点
- 权重行优先和列优先访问
- 反向传播的递推
## 3.问题
- vector使用push_back容易在访问时出错，最好先resize再赋值
- 权重初始化
- 数据归一化
- 前向和后向时忘记复位
- 使用Test类报free错误，未解决
## 4.MNIST
```
mkdir build
cd build
cmake ..
make
./net
```
- 输出每个样本的loss，共60000个
- 最后输出在10000个测试中正确的个数
## 5.其他
- 有问题可以在[这里](https://github.com/bonne658/evolvingnet)提问。