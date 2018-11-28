# MNIST 基础示例

```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

## 训练集准备:

国内下载速度较慢, 可以提前准备好下面的数据, 解压到对应文件夹(默认为`../data/raw`)中

```
http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

下载完成并解压后, 调用数据准备脚本生成训练用的数据集:

```bash
python prepare.py
```

然后再运行`main.py`即可.
