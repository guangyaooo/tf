import numpy as np
import tensorflow as tf


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
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


def main():
    # 生成测试数据
    x_data = np.linspace(-1, 1, 1000)[:, np.newaxis]
    y_data = np.square(x_data) + np.random.normal(size=x_data.shape)

    # 划分训练集和测试集
    shuffle = np.arange(1000)
    np.random.shuffle(shuffle)
    test_input = x_data[shuffle[0:200], :]
    test_labels = y_data[shuffle[0:200], :]

    train_input = x_data[shuffle[200:1000], :]
    train_labels = x_data[shuffle[200:1000], :]

    # 构建网络
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 1])
        y = tf.placeholder(tf.float32, shape=[None, 1])
    with tf.name_scope("hidden_1"):
        W_1 = tf.Variable(tf.random_normal(shape=[1, 10]), dtype=tf.float32)
        bias_1 = tf.Variable(tf.zeros([1, 10]) + 0.1, name='b', dtype=tf.float32)
        hidden1 = tf.add(tf.matmul(x, W_1), bias_1)
    with tf.name_scope("hidden_2"):
        W_2 = tf.Variable(tf.random_normal(shape=[10, 1]), dtype=tf.float32)
        bias_2 = tf.Variable(tf.zeros([1, 1]) + 0.1, name='b', dtype=tf.float32)
        prediction = tf.add(tf.matmul(hidden1, W_2), bias_2)
    # 将prediction的变化显示在HISTOGRAM中
    tf.summary.histogram("prediction", prediction)
    with tf.name_scope("train"):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction),
                                            reduction_indices=[1]))
        # 将prediction的变化显示在SCALAR中
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    tf.summary.scalar("loss", loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    # 配置summary writer,使用了俩个不同的summary_writer,最终，将会在log_dir目录下
    # 生成两个子目录/test和/train,当将"log_dir/"传给tensorboard时，tensorboard将
    # 以log_dir为根，遍历访问整个目录，并将得到的数据传给前段，用于前段组织渲染数据
    if tf.gfile.Exists("log_dir"):
        tf.gfile.DeleteRecursively("log_dir")
    tf.gfile.MakeDirs("log_dir")
    test_writer = tf.summary.FileWriter("log_dir/test", sess.graph)
    train_writer = tf.summary.FileWriter("log_dir/train")

    merged = tf.summary.merge_all()
    for i in range(2000):
        if i % 10 == 0:
            summary, loss_value, pre, _ = sess.run([merged, loss, prediction, train_step], feed_dict={
                x: train_input,
                y: train_labels
            })
            train_writer.add_summary(summary, i)
            print("train:%.3f" % loss_value)
        elif i % 15 == 0:
            summary, loss_value = sess.run([merged, loss], feed_dict={
                x: test_input,
                y: test_labels
            })
            test_writer.add_summary(summary, i)
            print("test:%.3f" % loss_value)
        else:
            sess.run(train_step, feed_dict={
                x: train_input,
                y: train_labels
            })


main()
