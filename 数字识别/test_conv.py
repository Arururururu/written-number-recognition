
#导入input_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

#创建两个占位符，x为输入网络的图像，y_为输入网络的图像类别
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

#权重初始化函数
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#偏置初始化函数
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#创建卷积op
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#创建池化op
#采用最大池化，也就是取窗口中的最大值作为结果
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#第1层，卷积层，卷积核大小为5*5  
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#把输入变成四维张量，#表示自动推测这个维度的size
x_image = tf.reshape(x, [-1,28,28,1])
#h_pool1的输出即为第一层网络输出
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#第2层，卷积层
#这层的输入和输出神经元通道数为32和64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#h_pool2即为第二层网络输出
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#第3层, 全连接层
#这层是拥有1024个神经元的全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#计算前需要把第2层的输出reshape成[batch, 7*7*64]的张量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
#Dropout层
#为了减少过拟合，在输出层前加入dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#输出层
#添加一个softmax层
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#计算交叉熵表示成本
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#train op, 使用ADAM优化器来做梯度下降。学习率为0.0001
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。
#因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#计算正确预测项的比例
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#开始训练模型，每次随机从训练集中抓取50幅图像
for i in range(10000):
  batch_x,batch_y = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = sess.run(accuracy,feed_dict={
        x:batch_x, y_: batch_y, keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
  sess.run(train_step,feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
  
accuracy_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
good = 0
total = 0
for i in range(10):
    testSet = mnist.test.next_batch(50)
    good += sess.run(accuracy_sum,feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0})
    total += testSet[0].shape[0]
print("test accuracy %g"%(good/total))

