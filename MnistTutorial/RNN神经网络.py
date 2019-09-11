import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入图片是28*28
n_inputs = 28  # 输入一行的长度 取决于图片一行28像素
max_time = 28  # 一共28行
lstm_size = 100  # 100个隐层单元
n_classes = 10  # 10个分类
batch_size = 50  # 每批次50样本
n_batch = mnist.train.num_examples // batch_size  # 计算一共多少批次

# 这里的none表示第一个维度可以任意，取决于放多少张图片
x = tf.placeholder(tf.float32, [None, 784])
# 正确的标签
y = tf.placeholder(tf.float32, [None, 10])

weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# RNN
def RNN(X, Weights, Biases):
    # inputs = [batch_size,max_time,n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM基本CELL
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[0]是cell state
    # final_state[1]是hidden_state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], Weights) + Biases)
    return results


# 计算RNN的返回结果
prediction = RNN(x, weights, biases)
# loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# AdamOptimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("I:" + str(epoch) + " acc=" + str(acc))


# I:0 acc=0.6748
# I:1 acc=0.8068
# I:2 acc=0.828
# I:3 acc=0.8442
# I:4 acc=0.85
# I:5 acc=0.9252
# I:6 acc=0.9308
# I:7 acc=0.9383
# I:8 acc=0.9427
# I:9 acc=0.9481
# I:10 acc=0.9432
# I:11 acc=0.9473
# I:12 acc=0.9559
# I:13 acc=0.9537
# I:14 acc=0.9541
# I:15 acc=0.9535
# I:16 acc=0.9578
# I:17 acc=0.9559
# I:18 acc=0.9635
# I:19 acc=0.961
# I:20 acc=0.9656
