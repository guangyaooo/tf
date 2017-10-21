import numpy as np
import tensorflow as tf

from learn_tf_2.experiment.model_1 import test1
from learn_tf_2.experiment.model_2 import test2


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
    x_placeholder = tf.placeholder(tf.float32)
    y_placeholder = tf.placeholder(tf.int32)
    return x_placeholder, y_placeholder


def next_batch(raw_data, batch_size, x_placeholder, y_placeholder, *args):
    """
    生成下一批训练数据
    :param x_placeholder: 神经网络输入占位符
    :param y_placeholder: 真实标签占位符
    :param raw_data:原始数据集
    :param batch_size: mini_batch大小
    :param args: 其它参数
    :return: feed_dict
    """
    r = np.random.randint(0, args[0], size=batch_size)
    fill_x = raw_data[0][r, :]
    fill_y = raw_data[1][r, :]
    feed_dict = {
        x_placeholder: fill_x,
        y_placeholder: fill_y
    }
    return feed_dict


def inference(inputs, *args):
    """
    搭建神经网络的基本结构
    :param inputs: 神经网络的输入
    :param args: 其它参数
    :return: 神经网络的输出
    """
    hidden_1 = add_layer(inputs, 10, 1, 1, tf.nn.relu)
    outputs = add_layer(hidden_1, 1, 10, 2, None)
    return outputs


def loss(prediction, y, *args):
    """
    构建损失函数
    :param prediction: 神经网络的输出，即inference的返回值
    :param y: 训练样本的真实值
    :param args: 其它参数
    :return: 损失函数op
    """
    labels = tf.to_int64(y)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=prediction, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss_op, learning_rate, *args):
    """
    定义神经网络的梯度更新方式
    :param loss_op: 损失函数op，loss函数的返回值
    :param learning_rate: 学习率
    :param args: 其它参数
    :return: 训练op
    """
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)


def evaluation(prediction, y, *args):
    """
    评估模型
    :param prediction: 神经网络的输出，即inference的返回值
    :param y: 训练样本的真实值
    :param args: 其它参数
    :return: 评估结果
    """
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def main():
    """
    将各个模块组装成完整的神经网络，训练神经网络，并评估神经网络
    :return:
    """
    model_1 = test1()
    model_2 = test2()
    print("union model")
    train_data = np.add(model_1[0], model_2[0])
    train_labels = model_2[2]
    test_data = np.add(model_1[1], model_2[1])
    test_labels = model_2[3]

    with tf.Graph().as_default():
        x_placeholder, y_placeholder = placeholder_input(100, [10], [10])
        prediction = inference(x_placeholder)

        loss_op = loss(prediction, y_placeholder)

        train_op = training(loss_op, 0.1)

        correct_num = evaluation(prediction, y_placeholder)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(4000):
            feed_dict = next_batch((train_data, train_labels), 100, x_placeholder, y_placeholder, train_data.shape[0])
            _, loss_value = sess.run([train_op, loss_op], feed_dict=feed_dict)
            if i % 500 == 0:
                print("loss%.2f" % loss_value)
        feed_dict = {
            x_placeholder: test_data,
            y_placeholder: test_labels
        }
        c_num = sess.run(correct_num, feed_dict)
        res = c_num / test_data.shape[0]
        feed_dict = next_batch((train_data, train_labels), 5000, x_placeholder, y_placeholder, train_data.shape[0])
        c_num2 = sess.run(correct_num, feed_dict)
        res2 = c_num2 / 5000
        print("测试集:%.2f,训练集：%.2f" % (res, res2))


main()
