import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(5.0)
input3 = tf.constant(2.0)

add = tf.add(input2,input3)
mul = tf.multiply(input1,add)

with tf.Session() as sess:
    res = sess.run([add,mul])
    print(res)

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

res = tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run(res,feed_dict={input1:[3.],input2:[4.]}))
