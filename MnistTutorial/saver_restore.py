import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 128
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.matmul(x, W) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

model_dir = 'net'

with tf.Session() as sess:
    sess.run(init)
    acc1 = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    saver.restore(sess, model_dir + '/test_net.ckpt')
    acc2 = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("init acc=" + str(acc1))
    print("restore acc= " + str(acc2))


# init acc=0.098
# restore acc= 0.9288
