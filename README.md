# teeth landmark/axes detection
该项目是对Dense Representative Tooth Landmark/axis Detection Network on 3D Model部分内容的复现。
该论文首次利用深度神经网络解决三维牙齿模型上的标志点和轴线检测问题，在口腔正畸学中发挥着不可替代的作用。
针对稀疏特征学习的挑战，我们设计了一种逐点场编码方法，巧妙地克服了基于深度学习预测稀疏特征的困难。
该论文的方法实现了无需人工干预的地标和坐标轴的自动检测。
还采用多尺度特征提取和特征增强模块来忠实地学习局部和全局特征。

Install dependencies: 
* torch==1.10.1
* cuda==11.3
* python==3.8



## 该项目可分为两类任务：
### 1:牙齿关键点检测
* 使用口扫数据
* [dataset.py](dataset.py)
* [pointnet2.py](models%2Fpointnet2.py)
* [train.py](train.py)
* [test.py](test.py)
* [visualization_landmarks.py](visualization_landmarks.py)
### 2:牙轴检测
* 使用CBCT数据
* [dataset_cep.py](dataset_cep.py)
* [pointnet2_cep.py](models%2Fpointnet2_cep.py)
* [train_cep.py](train_cep.py)
* [vis_cep.py](vis_cep.py)
* [le.py](le.py)

[register.py](register.py)可对口扫数据和CBCT数据进行配准


