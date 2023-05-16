本实验中的数据集为5000张  
但由于文件太大目前没有上传  
因此在本CNN.ipynb文件中的数据集为联网导入版  
数据集扩展到了50000张  
相较于课内实验  
由于在数据集上扩展到了10倍的数据量  
因此在batch_size的选择上相较于课内的32扩展到了128  
本实验中的CNN基本架构为  
1.输入层：32x32彩色图像
2.卷积层1：输出通道数32，过滤器大小3x3，激活函数：ReLU
3.卷积层2：输出通道数64，过滤器大小3x3，激活函数：ReLU
4.最大池化层1：池化大小2x2
5.卷积层3：输出通道数128，过滤器大小3x3，激活函数：ReLU
6.卷积层4：输出通道数128，过滤器大小3x3，激活函数：ReLU
7.最大池化层2：池化大小2x2
8.卷积层5：输出通道数256，过滤器大小3x3，激活函数：ReLU
9.卷积层6：输出通道数256，过滤器大小3x3，激活函数：ReLU
10.最大池化层3：池化大小2x2
11.展平输出
12.全连接层1（稠密层）：1024个神经元，激活函数：ReLU
13.全连接层2（稠密层）：512个神经元，激活函数：ReLU
14.输出层（稠密层）：10个神经元，激活函数：Softmax
15.损失函数：分类交叉熵