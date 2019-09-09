import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_vairable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])  # 5*5的采样窗口，32个卷积核从1个平面抽取特征
b_conv1 = bias_vairable([32])  # 每个卷积核一个偏置值

# 28*28*1 的图片卷积之后变为28*28*32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化之后变为 14*14*32
h_pool1 = max_pool_2x2(h_conv1)

# 第二次卷积之后变为 14*14*64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_vairable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 第二次池化之后变为 7*7*64
h_pool2 = max_pool_2x2(h_conv2)

# 第一个全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_vairable([1024])
# 7*7*64的图像变成1维向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第二个全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_vairable([10])
logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
prediction = tf.nn.sigmoid(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

prediction_2 = tf.nn.softmax(prediction)
correct_prediction = (tf.equal(tf.argmax(prediction_2, 1), tf.argmax(y, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("I:" + str(epoch) + " acc=" + str(acc))
        
        
# I:0 acc=0.9829
# I:1 acc=0.9853
# I:2 acc=0.9891
# I:3 acc=0.9898
# I:4 acc=0.9876
# I:5 acc=0.9912
# I:6 acc=0.9898
# I:7 acc=0.9891
# I:8 acc=0.9887
# I:9 acc=0.9907
# I:10 acc=0.9907
# I:11 acc=0.991
# I:12 acc=0.987
# I:13 acc=0.9918
# I:14 acc=0.9864
# I:15 acc=0.9902
# I:16 acc=0.9916
# I:17 acc=0.9899
# I:18 acc=0.9908
# I:19 acc=0.9745
# I:20 acc=0.9903
