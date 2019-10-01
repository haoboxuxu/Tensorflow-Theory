import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 批次大小
batch_size = 128
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 创建神经网络
W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([1, 2000]))
# 激活层
layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# drop层
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 500], stddev=0.1))
b2 = tf.Variable(tf.zeros([1, 500]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([1, 10]))
prediction = tf.nn.sigmoid(tf.matmul(layer2, W3) + b3)

# 交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# GradientDescent
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 0.7})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 0.7})
        print("I:" + str(epoch) + " test_acc=" + str(test_acc) + " train_acc" + str(train_acc))


# I:0 test_acc=0.8739 train_acc0.8697091
# I:1 test_acc=0.9042 train_acc0.9010182
# I:2 test_acc=0.9153 train_acc0.9143091
# I:3 test_acc=0.9263 train_acc0.92547274
# I:4 test_acc=0.9355 train_acc0.93252724
# I:5 test_acc=0.9374 train_acc0.93694544
# I:6 test_acc=0.9398 train_acc0.9419091
# I:7 test_acc=0.9422 train_acc0.9448182
# I:8 test_acc=0.9449 train_acc0.94796365
# I:9 test_acc=0.947 train_acc0.9514727
# I:10 test_acc=0.9466 train_acc0.9521273
# I:11 test_acc=0.9509 train_acc0.95483637
# I:12 test_acc=0.9515 train_acc0.95814544
# I:13 test_acc=0.9539 train_acc0.9588364
# I:14 test_acc=0.9562 train_acc0.95976365
# I:15 test_acc=0.9548 train_acc0.9625273
# I:16 test_acc=0.9565 train_acc0.9630727
# I:17 test_acc=0.9607 train_acc0.96483636
# I:18 test_acc=0.9593 train_acc0.9650546
# I:19 test_acc=0.9577 train_acc0.96674544
# I:20 test_acc=0.9597 train_acc0.9674909
