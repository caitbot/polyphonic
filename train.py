import tensorflow as tf
import math

def conv1d(x, f):
    return tf.nn.conv1d(x, f, strides=[1,1,1,1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(p, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='SAME')

def inference(data):
    
    with tf.name_scope('conv1'):
        filter_weights = tf.Variable(tf.random_normal([3, 2, 3], stddev=1.0, name="filter_weights"))
        conv1 = tf.nn.relu(conv1d(data, f))
        
    with tf.name_scope('max1'):
        max1 = tf.nn.max_pool(tf.reshape(conv1, [-1, 1, 132299, 3]), [1, 2, 1, 1], [1, 2, 1, 1], 'SAME')
        
    with tf.name_scope('conv2'):
        filter_weights = tf.Variable(tf.random_normal([3, 2, 3], stddev=1.0, name="filter_weights"))
        conv2 = tf.nn.relu(conv1d(max1, f))
        
    with tf.name_scope('max2'):
        max2 = tf.nn.max_pool(tf.reshape(conv2, [-1, 1, 132299, 3]), [1, 2, 1, 1], [1, 2, 1, 1], 'SAME')
        
    with tf.name_scope('logits'):
        logits = tf.matmul(tf.reshape(m, [-1, 132299 * 3]), max2)
        
    return logits

def init_graph():
    len([1])

def train(sounds, labels):
    tf.reset_default_graph()
    a = tf.placeholder(tf.float32, [None, 132299, 2])
    l = tf.placeholder(tf.float32, [None, 11])
    f = tf.Variable(tf.random_normal([3, 2, 3], dtype=tf.float32))
    w = tf.Variable(tf.random_normal([132299 * 3, 11]), dtype=tf.float32)
    
    t = tf.nn.conv1d(a, f, 1, 'SAME')
    r = tf.nn.relu(t)
    p = tf.reshape(r, [10, 1, -1, 3])
    m = tf.nn.max_pool(p, [1, 2, 1, 1], [1, 2, 1, 1], 'SAME')
    print m
    mul = tf.matmul(tf.reshape(m, [-1, 132299 * 3]), w)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(mul, l))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    print mul
    correct_prediction = tf.equal(tf.argmax(mul,1), tf.argmax(l,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    for i in range(20):
        batch = get_batch(10)
        print sess.run(f)
        print sess.run(train_step, feed_dict={a:batch[0], l:batch[1]})
        print sess.run(accuracy, feed_dict={a:batch[0], l:batch[1]})
