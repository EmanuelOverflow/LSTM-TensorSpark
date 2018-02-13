from pyspark import SparkConf, SparkContext
from pyspark.rdd import Partitioner
import tensorflow as tf
import numpy as np
from models.recurrent import rnn
import sys
import time
import os
import argparse
# import argparse
import csv
# import json
import multiprocessing  # works with python 2.6+
from tqdm import trange

# flags = tf.app.flags

# Spark configuration
# flags.DEFINE_string("master", "local", "Host or master node location (can be node name)")
# flags.DEFINE_string("spark_exec_memory", '4g', "Spark executor memory")
# flags.DEFINE_integer("partitions", 4, "Number of distributed partitions")
#
# # Network parameters
# flags.DEFINE_integer("epoch", 5, "Number of epochs")
# flags.DEFINE_integer("hidden_units", 128, "Number of hidden units")
# flags.DEFINE_integer("batch_size", 10, "Mini batch size")
# flags.DEFINE_integer("num_classes", 3, "Number of classes in dataset")
#
# flags.DEFINE_integer("evaluate_every", 10, "Numbers of steps for each evaluation")
#
# # Hyper-parameters
# flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
#
# # Dataset values
# flags.DEFINE_string("training_path", 'train', "Path to training set")
# flags.DEFINE_string("labels_path", "train_labels", "Path to training_labels")
# flags.DEFINE_string("output_path", "output_path", "Path for store network state")
#
# # Restore options
# flags.DEFINE_boolean("load_pickle", False, "Load weights from a pickle file")
# flags.DEFINE_string("load_op", "reduce", "Operation to execute after load")
#
# flags.DEFINE_string("checkpoint_path", "train_dir", "Directory where to save network model and logs")

# FLAGS = flags.FLAGS
# FLAGS._parse_flags()
params_str = ""
# print("Parameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     params_str += "{} = {}\n".format(attr.upper(), value)
#     print("{} = {}".format(attr.upper(), value))
# print("")


def compute_loss(labels, logits, sparse=True):
    if not sparse:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    cross_entropy_mean = tf.reduce_mean(
        cross_entropy
    )

    tf.summary.scalar(
        'cross_entropy',
        cross_entropy_mean
    )

    weight_decay_loss = tf.get_collection("weight_decay")

    if len(weight_decay_loss) > 0:
        tf.summary.scalar('weight_decay_loss', tf.reduce_mean(weight_decay_loss))

        # Calculate the total loss for the current tower.
        total_loss = cross_entropy_mean + weight_decay_loss
        tf.summary.scalar('total_loss', tf.reduce_mean(total_loss))
    else:
        total_loss = cross_entropy_mean

    return total_loss


def compute_accuracy(labels, logits, sparse=True):
    if not sparse:
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    else:
        correct_pred = tf.equal(tf.argmax(logits, 1), labels)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy


def min_max_normalizer(x):
    x = np.array(x)
    mmax = np.amax(x)
    mmin = np.amin(x)
    rng = mmax - mmin
    d = 1. - (((1. - 0.) * (mmax - x)) / rng)
    return d.tolist()


def csv_to_partitions(line, num_partitions, shuffle=True):
    # Spark is unable to create partitions with same size on default
    lines = csv.reader(line)
    # return data if len(data) > 0 else None
    data = []
    mini_batch = []

    for d in lines:
        if len(d) > 0:
            data.append(d)
            
    if shuffle:
        np.random.shuffle(data)

    total_lines = len(data)
    bs = int(total_lines / num_partitions)

    rdd = []
    key = 0
    for d in data:
        mini_batch.append(d)
        if len(mini_batch) == bs:
            rdd.append((key, mini_batch))
            key += 1
            mini_batch = []
    else:
        if len(mini_batch) > 0:
            rdd.append((key, mini_batch))

    return rdd


def text_to_rdd(sc, path, num_partitions):
    # If minPartitions csv_to_partitions is not necessary
    return sc.textFile(path, minPartitions=1).mapPartitions(lambda line: csv_to_partitions(line, num_partitions))


def process_batch(train_xy, normalize=False):
    train_x = []
    train_y = []
    if len(train_xy) <= 1:
        return train_xy

    for xy in train_xy:
        if len(xy) <= 1:
            continue

        x, y = map(float, xy[:-1]), xy[-1]
        train_x.append(x)
        train_y.append(y)

    if normalize:
        train_x = min_max_normalizer(train_x)

    return np.array(train_x), np.array(train_y)


def next_batch(train_x, train_y, batch_size=10, shuffle=True):
    total_iteration = int(train_x.shape[0] / batch_size)

    while True:
        if shuffle:
            p = np.random.permutation(train_x.shape[0])
            train_x = train_x[p]
            train_y = train_y[p]

        for i, batch in enumerate(xrange(0, train_x.shape[0], batch_size)):

            if i == total_iteration:
                continue

            x = train_x[batch:batch + batch_size]
            y = train_y[batch:batch + batch_size]
            yield x, y


def train_rnn(partition, net_settings, FLAGS, train_optimizer=tf.train.AdamOptimizer):
    # FLAGS: just to preserve tensorflow notation

    partition = list(partition)

    if len(partition) == 0:
        print 'RNN-LSTM - ZERO SIZE'
        return partition

    partition_key = partition[0][0]

    print 'LSTM - Partition: {}'.format(partition_key)

    full_batch_size = len(partition[0][1])
    if not FLAGS.batch_size:
        batch_size = full_batch_size
        for i in xrange(net_settings):
            net_settings[i]['batch_size'] = batch_size
    else:
        batch_size = FLAGS.batch_size

    num_hidden_last = net_settings[-1]['num_hidden']

    input_placeholder = tf.placeholder(tf.float32,
                                       shape=[batch_size, net_settings[0]['dim_size']],
                                       name="input_placeholder")
    labels_placeholder = tf.placeholder(tf.int64, shape=[batch_size], name="labels_placeholder")
    optimizer = train_optimizer(FLAGS.learning_rate)

    rnn_model = rnn.RNN(net_settings)

    with tf.name_scope("LSTM"):
        net = rnn_model.fit_layers(input_placeholder)

    with tf.variable_scope("Dense1"):
        dense = tf.reshape(net, [batch_size, -1])
        weights = tf.get_variable(name="weights", shape=[num_hidden_last, FLAGS.num_classes],
                                  initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable(name="bias", shape=[FLAGS.num_classes],
                               initializer=tf.truncated_normal_initializer())

        logits = tf.matmul(dense, weights) + bias

    loss = compute_loss(logits=logits, labels=labels_placeholder)
    train_op = optimizer.minimize(loss)

    accuracy = compute_accuracy(logits=logits, labels=labels_placeholder)

    train_vars = rnn_model.map_data_by_key()

    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        current_exec = str(time.time())
        train_dir = FLAGS.checkpoint_path
        model_save_dir = os.path.join(train_dir, current_exec, str(partition_key))

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        model_filename = os.path.join(model_save_dir, "spark_lstm.model")

        with open(os.path.join(model_save_dir, "params_settings"), "w+") as f:
            f.write(params_str)

        if os.path.isfile(model_filename) and FLAGS.use_pretrained_model:
            saver.restore(sess, model_filename)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(model_save_dir, "train"), sess.graph)
        # test_writer = tf.summary.FileWriter(os.path.join(model_save_dir, "test"), sess.graph)

        train_x, train_y = process_batch(partition[0][1])
        batches = next_batch(train_x, train_y, batch_size=batch_size)

        max_steps = FLAGS.epochs * batch_size
        total_steps = trange(max_steps)

        start = time.time()
        t_acc, v_acc, t_loss, v_loss = 0., 0., 0., 0.
        for step in total_steps:
            train_input, train_labels = batches.next()

            _, t_loss = sess.run([train_op, loss], feed_dict={
                input_placeholder: train_input,
                labels_placeholder: train_labels
            })

            t_loss = np.mean(t_loss)
            total_steps.set_description('Loss: {:.4f}/{:.4f} - t_acc {:.3f} - v_acc {:.3f}'
                                 .format(t_loss, v_loss, t_acc, v_acc))

            if step % FLAGS.evaluate_every == 0 or (step + 1) == max_steps:
                saver.save(sess, os.path.join(model_save_dir, 'spark_lstm'), global_step=step)

                summary, t_loss, t_acc = sess.run([merged, loss, accuracy], feed_dict={
                    input_placeholder: train_input,
                    labels_placeholder: train_labels
                })
                train_writer.add_summary(summary, step)
                # t_loss = np.mean(t_loss)

                # val_input, val_labels = batches.next()
                # summary, v_loss, v_acc = sess.run([merged, loss, accuracy], feed_dict={
                #     input_placeholder: val_input,
                #     labels_placeholder: val_labels
                # })
                # test_writer.add_summary(summary, step)
                # v_loss = np.mean(v_loss)
                #
                total_steps.set_description('Loss: {:.4f} - t_acc {:.3f}'
                                            .format(t_loss, t_acc))

        trainable_variables = sess.run(train_vars)
    end_time = time.time() - start
    print 'RNN-LSTM - Partition: {} - Time: {}s'.format(partition[0][0], end_time)
    return trainable_variables


def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='RNN-LSTM built on Tensorflow')

    parser.add_argument("--master", default='local', type=str, help="Host or master node location (can be node name)")
    parser.add_argument("--spark_exec_memory", default='4g', type=str, help="Spark executor memory")
    parser.add_argument("--partitions", default=4, type=int, help="Number of distributed partitions")

    # Network parameters
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("--hidden_units", default='128,256', type=str, help="List of hidden units per layer")
    parser.add_argument("--batch_size", default=10, type=int, help="Mini batch size")
    parser.add_argument("--num_classes", default=3, type=int, help="Number of classes in dataset")
    parser.add_argument("--in_features", default=4, type=int, help="Number of input features")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")

    parser.add_argument("--evaluate_every", default=10, type=int, help="Numbers of steps for each evaluation")

    # Dataset values
    parser.add_argument("--training_path", default='train', type=str, help="Path to training set")
    parser.add_argument("--labels_path", default='train_labels', type=str, help="Path to training_labels")
    parser.add_argument("--output_path", default='output_path', type=str, help="Path for store network state")

    # Restore options
    parser.add_argument("--mode", default="train", help="Execution mode")

    parser.add_argument("--checkpoint_path", default='train_dir', type=str,
                        help="Directory where to save network model and logs")

    return parser.parse_known_args(argv)


def main(argv):
    FLAGS, _ = parse_args(argv)

    input_path = FLAGS.training_path
    output = FLAGS.output_path

    master_host = FLAGS.master
    sem = FLAGS.spark_exec_memory
    partitions = FLAGS.partitions

    hidden_units = FLAGS.hidden_units.split(',')

    mode = FLAGS.mode

    # Initialize spark
    # Substitute 4 with max supported
    workers = partitions if partitions == multiprocessing.cpu_count() else partitions % multiprocessing.cpu_count()
    workers_master = '[%d]' % workers
    conf = SparkConf().setMaster(master_host + workers_master).setAppName(
        "RNN-LSTM").set("spark.executor.memory", sem)

    print 'Total workers: ', workers_master
    print 'Spark executor memory: ', sem

    sc = SparkContext(conf=conf)
    quiet_logs(sc)

    # with open(target_path, 'rb') as t_f:
    #     target = json.load(t_f)

    # target = map_target(target)

    if mode == 'train':
        # Read dataset into RDD as csv
        training_rdd = text_to_rdd(sc, input_path, partitions)
        minibatch_rdd = training_rdd.partitionBy(partitions + 1)  # OK FOR NOW

        net_settings = []
        for i, hidden in enumerate(hidden_units):
            if i == 0:
                dim_size = FLAGS.in_features
            else:
                dim_size = int(hidden_units[i - 1])

            net_settings.append({
                'layer_name': "LSTMLayer{}".format(i),
                'dim_size': dim_size,
                'num_hidden': int(hidden),
                'batch_size': FLAGS.batch_size,
                'normalize': True
            })

        start = time.time()

        weights_rdd = minibatch_rdd.mapPartitions(
            lambda x: train_rnn(x, net_settings, FLAGS), True)

        # Filter out empty partition
        weights_rdd = weights_rdd.filter(lambda x: len(x) == 2)
        out = weights_rdd.filter(lambda x: len(x) == 2)

        def mean_weights(x):
            average = []
            for el in x:
                average.append(np.mean(el, 0))
            return average

        weights_mean_rdd = out.reduceByKey(mean_weights)
        wm = weights_mean_rdd.collect()
        # Do something with wm

        print 'RNN-LSTM - Total Processing Time {}s'.format(time.time() - start)


if __name__ == '__main__':
    main(sys.argv)
