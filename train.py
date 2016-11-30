import tensorflow as tf

def train(sounds):
    a = tf.placeholder(tf.float32, [None, 132299, 2])
    f = tf.Variable(tf.random_normal([3, 2, 3], dtype=tf.float32))
    
    t = tf.nn.conv1d(a, f, 1, 'SAME')
    
    sess = tf.Session()
    
    sess.run(t, feed_dict={a:sounds})
