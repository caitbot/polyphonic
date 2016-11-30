import tensorflow as tf

def train(sounds, labels):
    a = tf.placeholder(tf.float32, [None, 132299, 2])
    l = tf.placeholder(tf.float32, [None, 11])
    f = tf.Variable(tf.random_normal([3, 2, 3], dtype=tf.float32))
    w = tf.Variable(tf.random_normal([132299 * 3, 11]), dtype=tf.float32)
    
    t = tf.nn.conv1d(a, f, 1, 'SAME')
    r = tf.nn.relu(t)
    p = tf.reshape(r, [10, 1, -1, 3])
    m = tf.nn.max_pool(p, [1, 2, 1, 1], [1, 2, 1, 1], 'SAME')
    
    mul = tf.matmul(tf.reshape(m, [-1, 132299 * 3]), w)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(mul, labels))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    print mul
    print labels.shape
    correct_prediction = tf.equal(tf.argmax(mul,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess = tf.Session()
    
    sess.run(tf.initialize_all_variables())
    print sess.run(accuracy, feed_dict={a:sounds, l:labels}).shape
