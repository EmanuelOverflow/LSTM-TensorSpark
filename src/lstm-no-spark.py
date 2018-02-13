import tensorflow as tf
import numpy as np
from models.recurrent import rnn
import time
import os
import csv
from tqdm import trange

flags = tf.app.flags

# Network parameters
flags.DEFINE_integer("epochs", 5, "Number of epochs")
flags.DEFINE_integer("batch_size", 10, "Mini batch size")
flags.DEFINE_integer("num_classes", 3, "Number of classes")
flags.DEFINE_integer("in_features", 4, "Number of input features")
flags.DEFINE_string("hidden_units", '128,256', "List of hidden units per layer (seprated by comma)")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")

flags.DEFINE_integer("evaluate_every", 10, "Numbers of steps for each evaluation")

# Dataset values
flags.DEFINE_string("training_path", 'train', "Path to training set")
flags.DEFINE_string("labels_path", "train_labels", "Path to training_labels")
flags.DEFINE_string("output_path", "output_path", "Path for store network state")

# Other options
flags.DEFINE_string("mode", 'train', "Execution mode")

flags.DEFINE_string("checkpoint_path", "train_dir", "Directory where to save network model and logs")

FLAGS = flags.FLAGS
FLAGS._parse_flags()
params_str = ""
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    params_str += "{} = {}\n".format(attr.upper(), value)
    print("{} = {}".format(attr.upper(), value))
print("")


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


def csv_to_batch(lines, split=0.8, shuffle=True):
    # Spark is unable to create partitions with same size on default
    lines = csv.reader(lines)
    # return data if len(data) > 0 else None
    data = []

    for line in lines:
        if len(line) > 0:
            data.append(line)

    # data = np.array(data)
    # if shuffle:
    #     perm = np.random.permutation(data.shape[0])
    #     data = data[perm]
    #
    # if split:
    #     train_split_idx = data.shape[0] * split
    #     train_data = data[:train_split_idx]
    #     val_data = data[train_split_idx:]
    #
    #     return train_data, val_data
    # else:
    return data


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


def train_rnn(dataset, net_settings, train_optimizer=tf.train.AdamOptimizer):
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

    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        current_exec = str(time.time())
        train_dir = FLAGS.checkpoint_path
        model_save_dir = os.path.join(train_dir, current_exec)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        model_filename = os.path.join(model_save_dir, "lstm_no_spark.model")

        with open(os.path.join(model_save_dir, "params_settings"), "w+") as f:
            f.write(params_str)

        if os.path.isfile(model_filename) and FLAGS.use_pretrained_model:
            saver.restore(sess, model_filename)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(model_save_dir, "train"), sess.graph)
        # test_writer = tf.summary.FileWriter(os.path.join(model_save_dir, "test"), sess.graph)

        train_x, train_y = process_batch(dataset)
        # val_x, val_y = process_batch(dataset[1])

        train_batches = next_batch(train_x, train_y, batch_size=FLAGS.batch_size)
        # val_batches = next_batch(val_x, val_y, batch_size=FLAGS.batch_size)

        batch_size = FLAGS.batch_size if FLAGS.batch_size else 1
        max_steps = FLAGS.epochs * batch_size
        total_steps = trange(max_steps)

        start = time.time()
        t_acc, v_acc, t_loss, v_loss = 0., 0., 0., 0.
        for step in total_steps:
            train_input, train_labels = train_batches.next()

            _, t_loss = sess.run([train_op, loss], feed_dict={
                input_placeholder: train_input,
                labels_placeholder: train_labels
            })

            t_loss = np.mean(t_loss)
            total_steps.set_description('Loss: {:.4f} - t_acc {:.3f}'
                                        .format(t_loss, t_acc))

            if step % FLAGS.evaluate_every == 0 or (step + 1) == max_steps:
                saver.save(sess, os.path.join(model_save_dir, 'lstm_no_spark'), global_step=step)

                summary, t_loss, t_acc = sess.run([merged, loss, accuracy], feed_dict={
                    input_placeholder: train_input,
                    labels_placeholder: train_labels
                })
                train_writer.add_summary(summary, step)
                # t_loss = np.mean(t_loss)

                # val_input, val_labels = val_batches.next()
                # summary, v_loss, v_acc = sess.run([merged, loss, accuracy], feed_dict={
                #     input_placeholder: val_input,
                #     labels_placeholder: val_labels
                # })
                # test_writer.add_summary(summary, step)
                # v_loss = np.mean(v_loss)

                total_steps.set_description('Loss: {:.4f} - t_acc {:.3f}'
                                            .format(t_loss, t_acc))

    print 'RNN-LSTM - Time: {}'.format(time.time() - start)
    return []


def read_dataset_from_path(path):
    with open(path, 'r') as df:
        lines = df.read().splitlines()

    return csv_to_batch(lines)


def main(_):
    input_path = FLAGS.training_path
    hidden_units = FLAGS.hidden_units.split(',')

    mode = FLAGS.mode
    if mode == 'train':
        dataset = read_dataset_from_path(input_path)

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

        train_rnn(dataset, net_settings)


if __name__ == '__main__':
    tf.app.run()
