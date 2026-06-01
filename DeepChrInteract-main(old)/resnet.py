# -*- coding:utf-8 -*-

# residual blocks

import tensorflow as tf

KERNEL_SIZE = 24 # 基本块大小24
STRIDES = 2 # 隔开2



# 基本块：18和34使用的块，包含两个3*3卷积层
# 然后弄一下大小匹配，不匹配会报错
class BasicBlock(tf.keras.layers.Layer):

    # 内部结构
    def __init__(self, filter_num, stride = 1):
        
        super(BasicBlock, self).__init__()

        # 第一个24*1卷积层
        self.conv1 = tf.keras.layers.Conv2D(filters = filter_num,
                                            kernel_size = (KERNEL_SIZE, 1),
                                            strides = STRIDES,
                                            kernel_initializer = 'he_uniform',
                                            padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        # 第二个24*1卷积层
        self.conv2 = tf.keras.layers.Conv2D(filters = filter_num,
                                            kernel_size = (KERNEL_SIZE, 1),
                                            strides = STRIDES,
                                            kernel_initializer = 'he_uniform',
                                            padding = 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 通过1*1卷积层进行shape匹配
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters = filter_num,
                                                       kernel_size = (1, 1),
                                                       strides = stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())

        # shape匹配，直接短接
        else:
            self.downsample = lambda x: x



    def call(self, inputs, training = None, **kwargs):

        # 通过identity模块
        identity = self.downsample(inputs)

        # [b, h, w, c]，通过第一个卷积单元
        x = self.conv1(inputs)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        
        # 通过第二个卷积单元
        x = self.conv2(x)
        x = self.bn2(x, training = training)

        # 2条路径输出直接相加，再通过激活函数
        # output = tf.nn.relu(tf.keras.layers.add([identity, x]))
        output = tf.nn.relu(x)

        return output



# 瓶颈块：50和101和152使用的块，包含三个卷积层，分别是1*1，3*1，1*1（4x）
# 然后弄一下大小匹配，不匹配会报错
class BottleNeck(tf.keras.layers.Layer):
    
    def __init__(self, filter_num, stride = 1):
        
        super(BottleNeck, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters = filter_num,
                                            kernel_size = (1, 1),
                                            strides = 1,
                                            padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters = filter_num,
                                            kernel_size = (3, 3),
                                            strides = stride,
                                            padding = 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters = filter_num * 4,
                                            kernel_size = (1, 1),
                                            strides = 1,
                                            padding = 'same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters = filter_num * 4,
                                                   kernel_size = (1, 1),
                                                   strides = stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())



    def call(self, inputs, training = None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training = training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training = training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output



# 创建一个基本块：18和34使用的块
def make_basic_block_layer(filter_num, blocks, stride = 1):
    
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride = stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride = 1))

    return res_block



# 创建一个瓶颈块：50和101和152使用的块
def make_bottleneck_layer(filter_num, blocks, stride = 1):
    
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride = stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride = 1))

    return res_block



# resnet models

# 分类数量，这里要分成2类，所以用2
NUM_CLASSES = 2

KERNEL_SIZE = 24 # 基本块大小24
STRIDES = 2 # 隔开2



# 第一类ResNet，适用于18或34
class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters = 64,
                                            kernel_size = (KERNEL_SIZE, 1),
                                            strides = STRIDES,
                                            padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size = (2, 1),
                                               strides = STRIDES,
                                               padding = 'same')

        self.layer1 = make_basic_block_layer(filter_num = 64,
                                             blocks = layer_params[0],
                                             stride = 4)
        self.layer2 = make_basic_block_layer(filter_num = 64,
                                             blocks = layer_params[1],
                                             stride = 4)
        self.layer3 = make_basic_block_layer(filter_num = 64,
                                             blocks = layer_params[2],
                                             stride = 4)
        self.layer4 = make_basic_block_layer(filter_num = 64,
                                             blocks = layer_params[3],
                                             stride = 4)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units = NUM_CLASSES, activation = tf.keras.activations.softmax)


    def call(self, inputs, training = None, mask = None):
        x = self.conv1(inputs)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training = training)
        x = self.layer2(x, training = training)
        x = self.layer3(x, training = training)
        x = self.layer4(x, training = training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


    
# 第二类ResNet，适用于50或101或152
class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters = 64,
                                            kernel_size = (7, 7),
                                            strides = 2,
                                            padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size = (3, 3),
                                               strides = 2,
                                               padding = 'same')

        self.layer1 = make_bottleneck_layer(filter_num = 64,
                                            blocks = layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num = 128,
                                            blocks = layer_params[1],
                                            stride = 2)
        self.layer3 = make_bottleneck_layer(filter_num = 256,
                                            blocks = layer_params[2],
                                            stride = 2)
        self.layer4 = make_bottleneck_layer(filter_num = 512,
                                            blocks = layer_params[3],
                                            stride = 2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units = NUM_CLASSES, activation = tf.keras.activations.softmax)


    def call(self, inputs, training = None, mask = None):
        x = self.conv1(inputs)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training = training)
        x = self.layer2(x, training = training)
        x = self.layer3(x, training = training)
        x = self.layer4(x, training = training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output



# 不同大小的声明函数，注意，到了152就已经到头了，再大意义不大了，所以平时用用50或者101就行了
# 这里的设置以图片为准
def resnet_18():
    return ResNetTypeI(layer_params = [2, 2, 2, 2])

def resnet_34():
    return ResNetTypeI(layer_params = [3, 4, 6, 3])

def resnet_50():
    return ResNetTypeII(layer_params = [3, 4, 6, 3])

def resnet_101():
    return ResNetTypeII(layer_params = [3, 4, 23, 3])

def resnet_152():
    return ResNetTypeII(layer_params = [3, 8, 36, 3])
