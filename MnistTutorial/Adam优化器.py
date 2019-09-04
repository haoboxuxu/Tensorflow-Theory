import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 批次大小
batch_size = 128
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W) + b)

# cost fun
# loss = tf.reduce_mean(tf.square(y - prediction))

# 交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# GradientDescent
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("I:" + str(epoch) + " acc=" + str(acc))

# I:0 acc=0.8952
# I:1 acc=0.9098
# I:2 acc=0.9135
# I:3 acc=0.9176
# I:4 acc=0.9212
# I:5 acc=0.9229
# I:6 acc=0.924
# I:7 acc=0.9267
# I:8 acc=0.9276
# I:9 acc=0.9277
# I:10 acc=0.9276
# I:11 acc=0.9286
# I:12 acc=0.9292
# I:13 acc=0.9291
# I:14 acc=0.9302
# I:15 acc=0.9299
# I:16 acc=0.9305
# I:17 acc=0.9298
# I:18 acc=0.9314
# I:19 acc=0.9308
# I:20 acc=0.9314
