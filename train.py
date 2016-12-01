import tensorflow as tf
import math
import time

def conv1d(x, f):
    return tf.nn.conv1d(x, f, stride=1, padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME')

def inference(data):
    
    with tf.name_scope('conv1'):
        filter_weights = tf.Variable(tf.random_normal([3, 2, 10], name='filter_weights'))
        conv1 = tf.nn.relu(conv1d(data, filter_weights))
        
    with tf.name_scope('max1'):
        max1 = max_pool(tf.reshape(conv1, [-1, 1, 132299, 10]))
        
    with tf.name_scope('conv2'):
        filter_weights = tf.Variable(tf.random_normal([3, 10, 20], name='filter_weights'))
        conv2 = tf.nn.relu(conv1d(tf.reshape(max1, [-1, 44100, 10]), filter_weights))
        
    with tf.name_scope('max2'):
        max2 = max_pool(tf.reshape(conv2, [-1, 1, 44100, 20]))
        
    with tf.name_scope('logits'):
        weights = tf.Variable(tf.random_normal([14700 * 20, 11]), name='weights')
        logits = tf.matmul(tf.reshape(max2, [-1, 14700 * 20]), weights, name='logits')
        
    return logits
    
def loss(logits, labels):

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
        
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        
    return loss
        
def training(loss):
    
    tf.scalar_summary('loss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    with tf.name_scope('training'):
        optimizer = tf.train.GradientDescentOptimizer(0.05)
        
        train_op = optimizer.minimize(loss, global_step=global_step)
        
    return train_op
    
def evaluation(logits, labels):
    
    with tf.name_scope('evaluation'):
        correct = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    
    return tf.reduce_sum(tf.cast(correct, tf.int32))
    
def placeholder_inputs():
    
    data_placeholder = tf.placeholder(tf.float32, shape=[None, 132299, 2], name='data')
    
    labels_placeholder = tf.placeholder(tf.int32, shape=[None, 11], name='labels')

    return data_placeholder, labels_placeholder
    
def fill_feed_dict(batch_size, data_pl, labels_pl, eval=False):
    
    batch = []
    
    if eval:
        batch = eval_get_batch(batch_size)
    else:
        batch = get_batch(batch_size)
    
    feed_dict = {data_pl: batch[0], labels_pl: batch[1]}
    
    return feed_dict
    
def do_eval(sess, eval_correct, data_pl, labels_pl, batch_size):
    
    true_count = 0
    
    for step in range(30):
        feed_dict = fill_feed_dict(batch_size, data_pl, labels_pl, True)
    
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
        
    precision = true_count / 300
    
    print 'Num examples: 300 Num correct: {} Precision: {}'.format(true_count, precision)
    
def run_training():
    
    with tf.Graph().as_default():
    
        data_placeholder, labels_placeholder = placeholder_inputs()
        
        logits = inference(data_placeholder)
        
        loss_op = loss(logits, labels_placeholder)
        
        train_op = training(loss_op)
        
        eval_correct = evaluation(logits, labels_placeholder)
        
        summary = tf.merge_all_summaries()
        
        init = tf.initialize_all_variables()
        
        sess = tf.Session()
        
        summary_writer = tf.train.SummaryWriter('.\summ', sess.graph)
        
        sess.run(init)

        
        for step in range(10):
            start_time = time.time()
            
            feed_dict = fill_feed_dict(10, data_placeholder, labels_placeholder)
            
            _, loss_value = sess.run([train_op, loss_op], feed_dict=feed_dict)
            
            duration = time.time() - start_time
            
            print 'Step {}: loss = {} ({} sec)'.format(step, loss_value, duration)
            
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
            
            print 'Training Data Eval:'
            if (step + 1) % 5 == 0:
                do_eval(sess, eval_correct, data_placeholder, labels_placeholder, 10)


