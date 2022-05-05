# Linux端基础训练预测功能测试

Linux端基础训练预测功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。


其中裁剪、量化、蒸馏非必须。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
|  :----  |   :----  |    :----  |  :----   |  :----   |  :----   |
|  DCLNet  | DCLNet | 正常训练 | - | - | - |


- 预测相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的预测功能汇总如下，

| 模型类型 | device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
| DCLNet | GPU | 1 | - | - | - |

## 2. 测试流程

### 2.1 准备数据

少量数据在`datasets/CUB_TINY`下，包含4种不同的鸟类图片。

运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。

### 2.2 安装依赖

安装以下依赖：

- PaddlePaddle >= 2.1
- numpy: 1.20.3
- pandas: 1.2.4
- Pillow: 8.2.0
- tqdm: 4.60.0 

### 2.3 功能测试

DCL的`Linux GPU/CPU 基础训练推理测试`命令如下所示。

```sh
bash test_tipc/test_train_inference_python.sh test_tipc/configs/DCLNet/train_infer_python.txt lite_train_lite_infer
```

TIPC结果：

![](../../resources/tipc0505-1.png)

![](../../resources/tipc0505-2.png)


## 3. 更多教程
本文档为功能测试用，更丰富的训练预测使用教程请参考：

- [模型训练、预测、推理](../../README.md)  