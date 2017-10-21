import tensorflow as tf


def inference(input, *args):
    """
    搭建神经网络的基本结构
    :param input: 神经网络的输入
    :param args: 其它参数
    :return: 神经网络的输出
    """
    pass


def loss(prediction, y, *args):
    """
    构建损失函数
    :param prediction: 神经网络的输出，即inference的返回值
    :param y: 训练样本的真实值
    :param args: 其它参数
    :return: 损失函数op
    """
    pass


def training(loss, learning_rate, *args):
    """
    定义神经网络的梯度更新方式
    :param loss: 损失函数op，loss函数的返回值
    :param learning_rate: 学习率
    :param args: 其它参数
    :return: 训练op
    """
    pass


def evaluation(prediction, y, *args):
    """
    评估模型
    :param prediction: 神经网络的输出，即inference的返回值
    :param y: 训练样本的真实值
    :param args: 其它参数
    :return: 评估结果
    """
    pass


def main():
    """
    将各个模块组装成完整的神经网络，训练神经网络，并评估神经网络
    :return:
    """
    pass


# 抽象出一些在构造神经网络时可能会用到函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    """
    add one more layer and return the output of this layer
    author: morvan
    :param inputs: 本层的输入
    :param in_size: 本层神经元个数
    :param out_size: 下一层神经元个数
    :param n_layer: 第n_layer层
    :param activation_function: 激活函数
    :return: 本层神经网络的输出op
    """
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


def placeholder_input(batch_size, x_shape: list, y_shape: list):
    """
    生成x_placeholder及y_placeholder
    :param batch_size: mini_batch的大小
    :param x_shape: x的维度
    :param y_shape: y的维度
    :return: x_placeholder,y_placeholder
    """
    x_shape.insert(0, batch_size)
    y_shape.insert(0, batch_size)
    x_placeholder = tf.placeholder(tf.float32, shape=tuple(x_shape))
    y_placeholder = tf.placeholder(tf.float32, shape=tuple(y_shape))
    return x_placeholder, y_placeholder


def next_batch(raw_data, batch_size, *args):
    """
    生成下一批训练数据
    :param raw_data:原始数据集
    :param batch_size: mini_batch大小
    :param args: 其它参数
    :return: feed_dict
    """
    pass
