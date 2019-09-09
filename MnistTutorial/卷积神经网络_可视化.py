import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 128
n_batch = mnist.train.num_examples // batch_size
max_step = 1000
keep_ = 0.8
log_dir = '/Users/haoboxuxu/Documents/pycharm/testft/3/'


# 生成权重
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')


# 生成偏差
def bias_vairable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')


# 记录变量
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')


def conv_layer(input_tensor, weight_shape, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(weight_shape)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_vairable([weight_shape[-1]])
            variable_summaries(biases)
        with tf.name_scope('conv_comput'):
            preactivate = conv2d(input_tensor, weights) + biases
        with tf.name_scope('activate'):
            activations = act(preactivate)
        return activations


def linear_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_vairable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('linear_comput'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        with tf.name_scope('activate'):
            activations = act(preactivate)
        return activations


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Max_pool')


with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, 784], name='input_x')
    with tf.name_scope('Input_reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x-image')
        tf.summary.image('input', x_image, 10)
    y = tf.placeholder(tf.float32, [None, 10], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# 第一次卷积   28*28*1->28*28*32
conv_layer1 = conv_layer(x_image, [5, 5, 1, 32], 'conv_layer1')
# 池化之后变为 14*14*32
with tf.name_scope('Max_pool1'):
    h_pool1 = max_pool_2x2(conv_layer1)

# 第二次卷积 14*14*32->14*14*64
conv_layer2 = conv_layer(h_pool1, [5, 5, 32, 64], 'conv_layer2')
# 第二次池化之后变为 7*7*64
with tf.name_scope('Max_pool2'):
    h_pool2 = max_pool_2x2(conv_layer2)

with tf.name_scope('Flatten'):
    flatten_ = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# 第一个全连接层 7*7*64 - 1024
fc1 = linear_layer(flatten_, 7 * 7 * 64, 1024, 'FC1')

with tf.name_scope('Dropput'):
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

# 第二个全连接层 1024 - 10
logits = linear_layer(fc1_drop, 1024, 10, 'FC2', act=tf.nn.sigmoid)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.name_scope('accuracy'):
    prediction = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()


def get_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(batch_size)
        k = keep_
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y: ys, keep_prob: k}


with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    sess.run(tf.global_variables_initializer())

    for i in range(max_step):
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=get_dict(False))
            test_writer.add_summary(summary, i)
            print("I:" + str(i) + " acc=" + str(acc))
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=get_dict(True))
            train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()


# I:0 acc=0.1032
# I:10 acc=0.275
# I:20 acc=0.2059
# I:30 acc=0.2927
# I:40 acc=0.2734
# I:50 acc=0.2461
# I:60 acc=0.3609
# I:70 acc=0.3703
# I:80 acc=0.6112
# I:90 acc=0.8521
# I:100 acc=0.8737
# I:110 acc=0.9285
# I:120 acc=0.9406
# I:130 acc=0.9504
# I:140 acc=0.9522
# I:150 acc=0.9576
# I:160 acc=0.9602
# I:170 acc=0.9589
# I:180 acc=0.9663
# I:190 acc=0.9645
# I:200 acc=0.9679
# I:210 acc=0.9651
# I:220 acc=0.9683
# I:230 acc=0.97
# I:240 acc=0.9687
# I:250 acc=0.9708
# I:260 acc=0.9699
# I:270 acc=0.9706
# I:280 acc=0.9696
# I:290 acc=0.9751
# I:300 acc=0.9745
# I:310 acc=0.9749
# I:320 acc=0.9751
# I:330 acc=0.9746
# I:340 acc=0.977
# I:350 acc=0.9781
# I:360 acc=0.977
# I:370 acc=0.9777
# I:380 acc=0.9744
# I:390 acc=0.9758
# I:400 acc=0.9772
# I:410 acc=0.9759
# I:420 acc=0.9726
# I:430 acc=0.9767
# I:440 acc=0.9784
# I:450 acc=0.981
# I:460 acc=0.9807
# I:470 acc=0.9823
# I:480 acc=0.9818
# I:490 acc=0.9808
# I:500 acc=0.9804
# I:510 acc=0.9815
# I:520 acc=0.9794
# I:530 acc=0.9827
# I:540 acc=0.9826
# I:550 acc=0.9831
# I:560 acc=0.981
# I:570 acc=0.9811
# I:580 acc=0.9797
# I:590 acc=0.9821
# I:600 acc=0.9808
# I:610 acc=0.9836
# I:620 acc=0.9814
# I:630 acc=0.9846
# I:640 acc=0.9827
# I:650 acc=0.9797
# I:660 acc=0.983
# I:670 acc=0.984
# I:680 acc=0.9821
# I:690 acc=0.9848
# I:700 acc=0.9852
# I:710 acc=0.9848
# I:720 acc=0.9853
# I:730 acc=0.9846
# I:740 acc=0.9852
# I:750 acc=0.9837
# I:760 acc=0.9847
# I:770 acc=0.984
# I:780 acc=0.9839
# I:790 acc=0.9835
# I:800 acc=0.9828
# I:810 acc=0.9842
# I:820 acc=0.9846
# I:830 acc=0.9838
# I:840 acc=0.9859
# I:850 acc=0.9843
# I:860 acc=0.9854
# I:870 acc=0.9857
# I:880 acc=0.9827
# I:890 acc=0.9842
# I:900 acc=0.9862
# I:910 acc=0.9874
# I:920 acc=0.9852
# I:930 acc=0.9852
# I:940 acc=0.9866
# I:950 acc=0.9855
# I:960 acc=0.9856
# I:970 acc=0.9854
# I:980 acc=0.9855
# I:990 acc=0.9824
