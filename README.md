### 额外文件

从这里下载http://d2l-data.s3-accelerate.amazonaws.com/glove.6B.100d.zip

解压缩并放置在NLP_Dataset文件夹内。

### 文件说明

train.py :训练模型
model.py :实现了两个模型双向LSTM和TextRNN

数据集设置：除了文本数据之外还需要预训练好的词嵌入模型参数，
数据集目录在 NLP_Dataset中

glove.6B.100d是我们额外需要的词嵌入模型的预训练参数。

Homework文件夹里是作业的数据集
原始的数据集我没上传，太大了。
train_1.csv 和val_1.csv是两个小型的训练集和验证集。

### 现阶段存在问题
我们的训练集中还有一个extra.csv里存储了评论的其他特征，比如
是否obscene等等。
而助教给的官方的evaluate文件中貌似也用到了这些特征，但我们交上去的submission里只
预测了toxicity(一个分数)，
所以要怎么用那些额外的特征呢？