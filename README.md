## 基本原理

人脸识别和目标检测这些还不太一样，比如大家传统的训练一个目标检测模型，你只有对这个目标训练了之后，你的模型才能找到这样的目标，比如你的目标检测模型如果是检测植物的，那显然就不能检测动物。但是人脸识别就不一样，以你的手机为例，你发现你只录入了一次你的人脸信息，不需要训练，他就能准确的识别你，这里识别的原理是通过人脸识别的模型提取你脸部的特征向量，然后将实时检测到的你的人脸同数据库中保存的人脸进行比对，如果相似度超过一定的阈值之后，就认为比对成功。不过我这里说的只是简化版本的人脸识别，现在手机和门禁这些要复杂和安全的多，也不是简单平面上的人脸识别。

## 需要的python库

```
opencv-python
dlib==19.17.0
face_recognition
pyqt5
```

**需要使用到python3.7的虚拟环境**
![image-20220109232309780](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20220109232309780.png)
### 创建虚拟环境

创建虚拟环境前请大家先下载博客开头的码云源码到本地。

本次我们需要使用到python3.7的虚拟环境，命令如下：

```bash
conda create -n face python==3.7.3
conda activate face
```

跑轮子（原作者[这里](https://mbd.pub/o/bread/ZJeYkpdt?next=pay&author_name=肆十二&author_avatar=https%3A%2F%2Fcdn.2zimu.com%2Fmbd_file_1679134971152.jpg)是要钱的，轮子也是必要的，其实也有其他方法实现，但是**我也付费了的**，也请尊重知识付费）
```bash
pip install dlib-19.17.0-cp37-cp37m-win_amd64.whl
```

## 安装必要的库

```bash
pip install -r requirements.txt
```

执行下面的主文件即可

```bash
python main.py
```


