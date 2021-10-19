# virtual adversarial mean teacher

#### 介绍
基于半监督深度学习的木马流量检测方法

#### 数据集

数据集为USTC-TFC2016，包含Benign正常流量以及Malware木马流量。

文件格式为pcap

#### 数据预处理

文件夹PcapToMnist包含数据预处理所需要的文件。

0_Tool:包含流量切分工具

1_Pcap:需处理的pcap文件存放位置

2_Session:经过2_PcapToSession处理后的会话文件存放位置

3_ProcessedSession:经过3_ProcessSession处理后的会话文件存放位置

4_Png:经过4_Session2png处理后的图像文件存放位置

5_Mnist:经过5_Png2Mnist处理后的mnist文件存放位置

处理流程：

1.将需要处理的pcap文件放入1_Pcap

2.依次执行2_PcapToSession，3_ProcessSession，4_Session2png，5_Png2Mnist
  1. windows运行2_PcapToSession.ps1(powershell脚本)(pcap所在路径不要有空格等特殊字符)
  2. python 3_ProcessSession.py D:\PcapToMnist\PcapToMnist\2_Session\AllLayers
  3. python 4_Session2png.py
  4. python 5_Png2Mnist.py
5.得到mnist文件

4.将mnist文件解压后的文件与converter.m文件放在同一目录

5.在matlab执行converter.m文件得到mat文件

#### 模型训练

文件夹tensorflow包含，模型训练的代码，代码的运行环境为MacOS 10.15.4， python 3.6.8，tensorflow 1.2.1.

模型训练流程：

1.将预处理后的mat文件放入data/images/compare文件夹

2.修改datasets/compare.py中TRAIN_NUM和TEST_NUM

  FILES = {

​    'train': Datafile(os.path.join(DIR, 'compare_train.mat'), TRAIN_NUM),

​    'test': Datafile(os.path.join(DIR, 'compare_test.mat'),TEST_NUM)

}

TRAIN_NUM可用matlab打开compare_train.mat文件，查看文件中的y值得到，TEST_NUM同理。

3.设置train_compare.py中n_labeled（标记样本数），training_length训练长度，rampup_length超参数a的变化长度

4.执行python train_compare.py进行模型训练

#### 对比实验

文件夹对比实验包含机器学习对比方法（ML）和论文对比方法(HTFS)：分别实现了三类对比实验

experiment1:在少标记样本下（10%、5%、1%、0.1%标记样本）的二分类（木马流量，正常流量）对比实验。

experiment2：检测未知样本的分类效果对比，即从十类样本中选取九类作为训练集，另一类作为测试集，总共重复十次。

experiment3：前述模型在少标记样本下的多分类效果对比。分别设置10类木马流量以及包含所有类的全集木马流量，标记样本占比为10%。
#### 实验结果
实验结果见文件夹 “图片”

