

# 模型优化

## 可变形卷积

可变形卷积网络(Deformable Convolution Network, DCN)顾名思义就是卷积的位置是可变形的，并非在传统的$$N × N$$的网格上做卷积，这样的好处就是更准确地提取到我们想要的特征（传统的卷积仅仅只能提取到矩形框的特征）。本实验在CenterNet head中加入了DCN，具体新的CenterNet head代码见centernet_head_dcn.py。在head中加入dcn后，模型的MOTA从原来的34.9%上升为39.3%，增长了4.4%。

## 数据增强
Mixup 是最先提出的图像混叠增广方案，其原理简单、方便实现，不仅在图像分类上，在目标检测上也取得了不错的效果。为了便于实现，通常只对一个 batch 内的数据进行混叠。
Mixup原理公式为：
<center><img src="./images/mixup_formula.png" width=70%></center>
Mixup后图像可视化如下图所示：
<center><img src="./images/mixup.png" width=70%></center>
在baseline中加入dcn后再加入mixup数据增强，模型MOTA为36.8%，比只加入dcn下降了2.5%。
具体为在fairmot_reader_1088x608.yml文件中TrainReader的sample_transforms下加入- Mixup: {alpha: 1.5, beta: 1.5}，在TrainReader中加入mixup_epoch: 25。
mixup_epoch (int): 在前mixup_epoch轮使用mixup增强操作；当该参数为-1时，该策略不会生效。默认为-1。

## 指数移动平均（EMA）
在深度学习中，经常会使用EMA（指数移动平均）这个方法对模型的参数做平均，以求提高测试指标并增加模型鲁棒。指数移动平均（Exponential Moving Average）也叫权重移动平均（Weighted Moving Average），是一种给予近期数据更高权重的平均方法。
本实验在baseline中加入dcn的基础上加入了EMA，ema_decay=0.9998。模型MOTA为38.5%，比只加入dcn下降了0.8%。
具体为在fairmot_dla34.yml文件中添加
```
use_ema: true
ema_decay: 0.9998
```

