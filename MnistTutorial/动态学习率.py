import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 128
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(1, dtype=tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([1, 2000]))
layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 500], stddev=0.1))
b2 = tf.Variable(tf.zeros([1, 500]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([1, 10]))
prediction = tf.matmul(layer2, W3) + b3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

train_step = tf.train.AdadeltaOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()

prediction_2 = tf.nn.softmax(prediction)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction_2, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        sess.run(tf.assign(lr, 1 * (0.98 ** epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.8})
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("I:" + str(epoch) + " acc=" + str(acc) + " lr=" + str(learning_rate))
