'''
* Dopo aver costruito il modello batch con spark provare a fonderlo con lo streaming
in questo modo si dovrebbe essere in grado di effettuare il fit infinito della rete (buon modello?
overfitting? dove prendo l'etichetta?) utilizzando un modello lambda (parlare con nec e ferone)
Se non e' attuabile si potrebbe attaccare lo streaming per la classificazione.

* Altra idea...se processo tutto in streaming? In questo modo dovrei ottenere automaticamente dei min-batch
(Spark lavora in modalita' mini-batch quando utilizza lo streaming)
'''

# from pyspark import SparkConf, SparkContext
# from pyspark.rdd import Partitioner
import tensorflow as tf
import numpy as np
import sys
import time
from random import Random


##########################################################################
##########################################################################

NEURON_WEIGHT = 0
HIDDEN_NEURON = 1

class OutputLayer():

    ACTIVATON_FUNCTIONS = [
        tf.nn.softmax
    ]

    SOFTMAX = 0

    def __init__(self, num_hidden, num_class, batch_size, activation = 0):
        self.weight = tf.Variable(tf.random_normal([num_hidden, num_class]), name="output_layer_weight")
        self.bias = tf.Variable(tf.ones([batch_size, num_class]), name="output_layer_bias")

        self.activation = self.ACTIVATON_FUNCTIONS[activation]

        self.cross_entropy = None #maybe local
        self.error = None # maybe local

    def predict(self, h_input, s=None):
        if s:
            print 'TEST_DATA: ', s.run([h_input, tf.shape(h_input), tf.shape(self.weight)])
        return self.activation(tf.matmul(h_input, self.weight) + self.bias)

    def compute_loss(self, h_input, target, s=None):
        prediction = self.predict(h_input, s)
        return -tf.reduce_sum(target * tf.log(prediction)) # could be another function

    # I'm unuseful for now
    #def optimize(self, h_input, learning_rate=0.01):
    #    return self.optimizer(learning_rate).minimize(compute_loss(h_input))

    def evaluate(self, test_state, target, s=None):
        return self.predict(test_state, s)

    def compute_error(self, test_state, target, s=None):
        prediction = self.predict(test_state, s)
        incorrects = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1)) # always 1?
        return tf.reduce_mean(tf.cast(incorrects, tf.float32)) # what is reduce_mean?

class RnnLayer():

    def __init__(self, ltype, num_hidden, dim_size, batch_size=1):
        self.ltype = ltype
        self.shape = [num_hidden, dim_size]
        self.batch_size = batch_size

        self.output_layer = OutputLayer(num_hidden, 3, batch_size)

        self.node_id = Random().getrandbits(64)
        self.state = []

        # ATTENTION: non si possono passare tutti gli input insieme, perche deve aggiornare gli stati con ogni nuovo input


        # weight_forget size = num_hidden , dim_size * 2
        # dim_size * 2 = because it is the horizontal concatenation of two weight matrix Wf=[Wx|Wht]
        # This size is shared by all weights for the concatenation needed by LSTM
        self.weight_forget = tf.Variable(tf.random_normal([self.shape[0], self.shape[1] + self.shape[0]]), name="Wf_%d" % self.node_id)
        self.biases_forget = tf.Variable(tf.ones([self.shape[0], 1]), name="bf_%d" % self.node_id)

        # Dovrebbero essere inizializzati random nell'intervallo [-1/sqrt(n),
        # 1/sqrt(n)], n = numero di connessioni in ingresso (XAVIER)
        self.weight_input = tf.Variable(tf.random_normal([self.shape[0], self.shape[1] + self.shape[0]]), name="Wi_%d" % self.node_id)
        self.biases_input = tf.Variable(tf.ones([self.shape[0], 1]), name="bi_%d" % self.node_id)

        self.weight_C = tf.Variable(tf.random_normal([self.shape[0], self.shape[1] + self.shape[0]]), name="WC_%d" % self.node_id)
        self.biases_C = tf.Variable(tf.ones([self.shape[0], 1]), name="bC_%d" % self.node_id)

        self.weight_output = tf.Variable(tf.random_normal([self.shape[0], self.shape[1] + self.shape[0]]), name="Wo_%d" % self.node_id)
        self.biases_output = tf.Variable(tf.ones([self.shape[0], 1]), name="bo_%d" % self.node_id)

        #self.ht = tf.Variable(tf.ones([self.shape[1], self.shape[0]]), trainable=False, name="h_t_%d" % self.node_id)
        self.ht = None
        #self.ht = tf.Variable(tf.zeros([self.shape[1], self.shape[0]]), trainable=False, name="ht_%d" % self.node_id)
        self.Ct = None
        #        self.Ct = tf.Variable(tf.ones([self.shape[0], 1]), trainable=False, name="Ct_%d" % self.node_id)
        #self.i_t = tf.Variable(tf.ones(self.shape), trainable=False, name="it_%d" % self.node_id)
        #self.Cta = None
        #self.Cta = tf.Variable(tf.ones(self.shape), trainable=False, name="Cat_%d" % self.node_id)


        #self.i_t = tf.Variable(tf.ones([self.shape[0], 1]), trainable=False, name="it_%d" % self.node_id)
        #self.Cta = tf.Variable(tf.ones([self.shape[0], 1]), trainable=False, name="Cat_%d" % self.node_id)

    def layer_step(self, weight, bias, input_data, func, name , s=None):
        data_concat = tf.concat(0, [self.ht, input_data])
        W = tf.matmul(weight, data_concat, name='%s_W' % name)
        return func(W + bias, name='%s_%s' % (name, func.__name__))

    def forget_gate_layer(self, input_data):
        self.ft.assign(self.layer_step(
            self.weight_forget, self.biases_forget, input_data, tf.sigmoid,
            'ft_%d' % self.node_id))

    def input_gate_layer(self, input_data):
        self.it.assign(self.layer_step(
            self.weight_input, self.biases_input, input_data,
            tf.sigmoid, 'it_%d' % self.node_id))
        self.Cta.assign(self.layer_step(
            self.weight_C, self.biases_C, input_data, tf.tanh,
            'Cat_%d' % self.node_id))

    def update_old_cell_state_layer(self):
        #self.Ct.assign(ft.add(self.ft * self.Ct, self.it * self.Cta, 'Ct_%d' % self.node_id))
        self.Ct.assign(tf.add(tf.mul(self.ft, self.Ct),
                          tf.mul(self.it, self.Cta), 'Ct_%d' % self.node_id))

    def to_output_layer(self, input_data, s):
        ot = self.layer_step(self.weight_output,
                              self.biases_output, input_data, tf.sigmoid, 'Ot_%d' % self.node_id, s)
        self.ht.assign(tf.mul(ot, tf.tanh(self.Ct), 'ht_%d' % self.node_id))

    def train_layer(self, input_data_T, s):
        # self.ft etc can be local
        with tf.name_scope('forget_gate_layer'):
            self.forget_gate_layer(input_data_T)
        with tf.name_scope('input_gate_layer'):
            self.input_gate_layer(input_data_T)
        with tf.name_scope('update_old_cell_state_layer'):
            self.update_old_cell_state_layer()
        with tf.name_scope('to_output_layer'):
            self.to_output_layer(input_data_T, s)

    def compute_loss(self, input_data, label):  # set choose loss function
        return self.output_layer.compute_loss(self.ht, label)

    def restore_state(self):
        self.ht = self.state[-1][0]
        self.Ct = self.state[-1][1]

    def fit_next(self, data, s, last_state=True, train=True):  # set choose optimizer
        with tf.name_scope('optimizer'):
            input_data_T = tf.transpose([data], name="input_data_T")

            if not self.ht:
                # Init h_t
                self.ht = tf.Variable(tf.random_normal([self.shape[0], 1]), trainable=False, name="ht_%d" % self.node_id)
                # Init C_t
                self.Ct = tf.Variable(tf.ones([self.shape[0], 1]), trainable=False, name="Ct_%d" % self.node_id)
                #self.Cta = tf.Variable(tf.ones([self.shape[0] + input_length, self.shape[0]]), trainable=False, name="Cat_%d" % self.node_id)

                # Init layers variables
                self.ft = tf.Variable(tf.ones([self.shape[0], 1]), trainable=False, name="ft_%d" % self.node_id)
                self.it = tf.Variable(tf.ones([self.shape[0], 1]), trainable=False, name="it_%d" % self.node_id)
                self.Cta = tf.Variable(tf.ones([self.shape[0], 1]), trainable=False, name="Cta_%d" % self.node_id)

                s.run(tf.initialize_variables([self.ht, self.Ct, self.ft, self.it, self.Cta]))

            with tf.name_scope('train_layer'):
                self.train_layer(input_data_T, s)
                if train:
                    self.state.append((self.ht, self.Ct)) # store the state of each step
                    ret = self.state[-1] if last_state else self.state
                else:
                    ret = (self.ht, self.Ct)
                    self.restore_state()
        return ret


    def minimize(self, data, t_label, s, optimizer):
        label = tf.Variable(t_label, name="label", trainable=False, dtype=tf.float32)
        s.run(tf.initialize_variables([label]))
        with tf.name_scope('cost_function'):
            cost = self.output_layer.compute_loss(tf.transpose(self.ht), label)
        with tf.name_scope('minimization'):
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            return optimizer.minimize(cost, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)


    def evaluate(self, t_data, t_label, s):
        state = self.fit_next(t_data, s, train=False)
        label = tf.Variable(t_label, name="label", trainable=False, dtype=tf.float32)
        s.run(tf.initialize_variables([label]))
        with tf.name_scope('evaluate'):
            return self.output_layer.evaluate(tf.transpose(state[0]), label)


        # decay_fn = tf.train.exponential_decay
        # Tutta sta roba da aggiornare???
        # loss = tf.argmax(self.ht, 1)
        # learning_rate_decay_fn=decay_fn
        # optimization = tf.contrib.layers.optimize_loss(self.ht, global_step=tf.Variable([1, 1]), optimizer=optimizer,
        #                                                learning_rate=0.01,
        #                                                variables=[self.weight_forget, self.weight_input, self.weight_output,
        #                                                           self.weight_C, self.biases_forget, self.biases_input,
        #                                                           self.biases_C, self.biases_output])
        # opt_op = optimizer.minimize(loss, var_list=[self.weight_forget, self.weight_input, self.weight_output,
        # self.weight_C, self.biases_forget, self.biases_input, self.biases_C,
        # self.biases_output])

##########################################################################
##########################################################################
class RNN():

    OPTIMIZERS = [
        tf.train.AdamOptimizer,
        tf.train.GradientDescentOptimizer
    ]

    ADAM_OPTIMIZER = 0
    GRADIENT_DESCEND_OPTIMIZER = 1

    # FIXME: Put hidden > 1 and epoch > 1 as default in final version
    def __init__(self, props, target, epoch=100, layers=1, optimizer=0):
        self.rnn_type = props[0]['layer_type']
        self.target = target
        self.epoch = epoch

        #self.layers = [RnnLayer(self.rnn_type, props[0].num_hidden, props[0].dim_size)] * layers
        self.layer = RnnLayer(self.rnn_type, props[0]['num_hidden'], props[0]['dim_size'])  # * hidden

        self.optimizer = self.OPTIMIZERS[optimizer]()

    @property
    def num_layers(self):
        return len(self.layers)

    def map_data_by_key(self, s):
        weight_forget = s.run(self.layer.weight_forget)
        weight_input = s.run(self.layer.weight_input)
        weight_output = s.run(self.layer.weight_output)
        weight_C = s.run(self.layer.weight_C)
        biases_forget = s.run(self.layer.biases_forget)
        biases_input = s.run(self.layer.biases_input)
        biases_C = s.run(self.layer.biases_C)
        biases_output = s.run(self.layer.biases_output)
        return [
            ("wf", weight_forget),
            ("wi", weight_input),
            ("wo", weight_output),
            ("wc", weight_C),
            ("bf", biases_forget),
            ("bi", biases_input),
            ("bc", biases_C),
            ("bo", biases_output),
        ]

    def fit_layer(self, datas, s, partition=-1, minibatch=None, normalize=True):
        #global COUNT_RUN
        #print 'TOTAL_COUNT: ', COUNT_RUN
        #COUNT_RUN=+1

        data = []
        labels = []
        if len(datas) <= 1:
            return datas

        for d in datas:
            if len(d) <= 1:
                continue

            dd, ll = ([float(dt) for dt in d[:-1]], self.target['mapping'][d[-1]])
            data.append(dd)
            labels.append(ll)

        if normalize:
            data = min_max_normalizer(data)

        if not minibatch:
            minibatch = len(data) - 1

        batch_weights = None
        #input_data = tf.placeholder(
            #tf.float32, [1, len(data[0])], name="input_data")
        #    tf.float32, [len(data), len(data[0])], name="input_data")
        var_initialized = False

        for e in xrange(self.epoch):
            e_time = time.time()
            print 'Partition: %d - Epoch: %d' % (partition, (e + 1))
            print

            for i, (in_data, label) in enumerate(zip(data, labels)):
                # print "Partition: %d - Input: %d/%d\r" % (partition, (i + 1), len(data))
                #s.run(self.layer.fit_next(in_data, s))#, feed_dict={input_data: data})
                self.layer.fit_next(in_data, s)
                if i%minibatch == 0:
                    #print 'Minibatch run ', i
                    minimizer = self.layer.minimize(in_data, label, s, self.optimizer)
                    if not var_initialized:
                        tf.initialize_all_variables().run()
                        var_initialized = True
                    s.run(minimizer)
                    #s.run(self.layer.minimize(in_data, label, s, self.optimizer))
            #print
            #print 'Time for an epoch: ', time.time() - e_time
        #s.run(minimizer)
        # summary_writer = tf.train.SummaryWriter("./tmp/graph_3", s.graph)
        batch_weights = self.map_data_by_key(s)

        return batch_weights
        #return data
        # s.run(tf.initialize_all_variables())
        # maybe self is not accepted
        # print np.transpose(input_data)
        # self.layers[0].run(train_step, feed_dict={
        #                    self.input_layer: np.transpose(input_data)})
        # print 'Last weight forget'
        # print self.layers[0].run(self.layers[0].weight_forget)

    def evaluate(self, data, label, s):
        print 'EVALUATION: ', s.run(self.layer.evaluate(data, label, s)), label


###########################################################
###########################################################

# DA NON USARE PER ORA
class Dataset():
    def __init__(self, data, label, mapping={}):
        self.data = data
        self.label = label
        self.label_mapping = mapping

    def putData(self, data, label):
        '''
        if (type(label) == 'string'):
            if label in labels_mapping:
                label = labels_mapping[label].label
            else:
                # could be more efficient
                self.num_labels += 1
                for l_name in labels_mapping:
                    labels_mapping[l_name].data.append(0)
                    labels_mapping[l_name].index.append(self.num_labels)
                labels_mapping[label] = namedtuple(label, 'name,data,index')
                labels_mapping[label].data = [0] * len(labels_mapping.keys)
                labels_mapping[label].data.append(1)
        else:
            pass  # map class
        '''
        self.data.append(data)
        self.label.append(label)


import argparse
import csv
import json
import multiprocessing # works with python 2.6+

def min_max_normalizer(data, last_str=False):
    #     if last_str:
    #         mmax = np.amax(d[:,:-1])
    #         mmin = np.amin(d[:,:-1])
    #         rng = mmax - mmin
    #         d[:,:-1] = 1. - (((1. - 0.) * (mmax - d[:,:-1])) / rng)
    # else:
    d = np.array(data)
    mmax = np.amax(d)
    mmin = np.amin(d)
    rng = mmax - mmin
    d = 1. - (((1. - 0.) * (mmax - d)) / rng)
    return d.tolist()

def CSVtoMiniBatchData(line, num_partitions, shuffle=True):
    '''
    Enhance me
    '''

    # Spark is unable to create partitions with same size on default
    with open(line, 'rb') as cf:
        lines = csv.reader(cf)
        # return data if len(data) > 0 else None

        data = []
        minibatch = []

        for d in lines:
            if len(d) > 0:
                data.append(d)
                #dd, ll = float(dd) if i < len(d)-1 for i, dd in enumerate(d)], d[:-1]

        if shuffle:
            np.random.shuffle(data)

        total_lines = len(data)
        bs = int(total_lines / num_partitions)

        # print total_lines
        # print bs

        rdd = []
        key = 0
        for d in data:
            minibatch.append(d)
            if len(minibatch) == bs:
                rdd.append((key, minibatch))
                # rdd.append(minibatch)
                key += 1
                minibatch = []
        else:
            if len(minibatch) > 0:
                rdd.append((key, minibatch))
                # rdd.append(minibatch)
        return rdd


def textToRDDCsv(sc, path, num_partitions):
    # If minPartitions CSVtoMiniBatchData is not necessary
    return sc.textFile(path, minPartitions=1).mapPartitions(lambda line: CSVtoMiniBatchData(line, num_partitions))
    # return sc.textFile(path,
    # minPartitions=num_partitions).mapPartitions(lambda line:
    # csv.reader(line))


# , hidden, epoch, batch_size, dim_size=4):
def train_rnn(partition, multilayer_props, target, epoch=100, minibatch=None):

    partition = list(partition)

    if len(partition) == 0:
        print 'RNN-LSTM - ZERO SIZE'
        return partition

    print 'RNN-LSTM - Partition: %s' % partition[0][0]
    #print '\tTotal epoch: %d' % epoch
    #'''
    multilayer_props[0]['dim_size'] = len(partition[0][1][0]) - 1
    rnn = RNN(multilayer_props, epoch=epoch, target=target)
    start = time.time()
    with tf.Session() as s:
        ret = rnn.fit_layer(partition[0][1], s, partition=int(partition[0][0]), minibatch=minibatch)
        # test_data = [5.0,3.3,1.4,0.2]
        # if multilayer_props[0]['normalize']:
        #     test_data = min_max_normalizer(test_data)
        # rnn.evaluate(test_data, [1.0, 0.0, 0.0], s)
    print 'RNN-LSTM - Partition: %s - Time: %f' % (partition[0][0], time.time() - start)
    return ret
    #'''
    #return partition

def map_target(target):
    target_out = dict(mapping={}, target=[])
    clazz = 0
    if type(target) == dict:
        classes = np.eye(target['num_class'], dtype=np.uint8).tolist()
        for t in target['classes']:
            target_out['mapping'][t] = classes[clazz]
            clazz += 1
        target_out['target'].extend(classes)
    return target_out

def quiet_logs( sc ):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org").setLevel( logger.Level.ERROR )
  logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='RNN-LSTM built on Tensorflow')
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='Path to dataset')
    parser.add_argument('-t', '--target', type=str,
                        required=True, help='Path to target classes')
    parser.add_argument('-m', '--master', type=str,
                        help='host of master node', default='local')
    parser.add_argument('-sem', '--sparkexecmemory', type=str,
                        help='Spark executor memory', default='4g')
    parser.add_argument('-p', '--partitions', type=int,
                        help='Number of partitions', default=4)
    parser.add_argument('-mb', '--minibatch', type=int,
                        help='Number of minibatch per partition', default=None)
    parser.add_argument('-mbp', '--minibatchperc', type=int,
                        help='Number of minibatch percentage per partition', default=None)
    parser.add_argument('-hl', '--numHidden', type=int,
                        help='Number of hidden layers', default=1)
    parser.add_argument('-e', '--epoch', type=int,
                        help='Number of training epoch', default=1)
    parser.add_argument('-o', '--output', type=str,
                        help='Output path', default='temp')
    parser.add_argument('-lp', '--loadPickle', type=bool,
                        help='Load weights from a pickle file', default=False)
    parser.add_argument('-lo', '--loadOp', type=str,
                        help='Operation to execute after load', default='reduce')
    parser.add_argument('-gp', '--graphPath', type=str,
                        help='Graph path', default='tmp/graph_default')

    args = vars(parser.parse_args())
    input_path = args['input']
    target_path = args['target']
    master_host = args['master']
    sem = args['sparkexecmemory']
    partitions = args['partitions']
    hidden = args['numHidden']
    epoch = args['epoch']
    output = args['output']
    load = args['loadPickle']
    load_op = args['loadOp'].split('|')
    graphPath = args['graphPath']
    minibatch = args['minibatch']
    minibatchp = args['minibatchperc']

    # Initialize spark
    # Substitute 4 with max supported
    # workers = partitions % multiprocessing.cpu_count()
    # workers_master = '[%d]' % workers
    # conf = SparkConf().setMaster(master_host + workers_master).setAppName(
    #     "RNN-LSTM").set("spark.executor.memory", sem)

    # print 'Total workers: ', workers_master
    # print 'Spark executor memory: ', sem

    # sc = SparkContext(conf=conf)
    # quiet_logs(sc)

    print 'Initialize application'

    with open(target_path, 'rb') as t_f:
        print 'Read target config'
        target = json.load(t_f)

    target = map_target(target)

    if not load:
        print 'Read Input Data'
        # Read dataset into RDD as csv
        #training_rdd = textToRDDCsv(sc, input_path, partitions)
        training_rdd = CSVtoMiniBatchData(input_path, 1)

        if minibatchp:
            minibatch = (len(training_rdd[0][1]) / 100) * minibatchp

        # training_rdd.partitionBy(partitions + 1)  # .sortByKey(
        #minibatch_rdd = training_rdd.partitionBy(partitions + 1)  # FOR NOW OK
        # lambda x: x[0], numPartitions=partitions)
        # minibatch_rdd = training_rdd.repartition(partitions)

        # It is simple to extend multilayer lstm to support different settings
        # on multiple layers
        multilayer_props = [dict(
            layer_name='1',
            layer_type='lstm',
            dim_size=-1,
            num_hidden=hidden,
            normalize=True
        )]
        print 'Start training'
        start = time.time()

        weights_rdd = train_rnn(training_rdd, multilayer_props, epoch=epoch, target=target, minibatch=minibatch)

        # Return weights and average them
        #weights_rdd = weights_rdd.filter(lambda x: len(x) == 2)
        #weights = weights_rdd.saveAsPickleFile(output + '_raw')

        #print 'WEIGHTS'
        #print weights

        # Without spark no empty partition exists

        #out = weights_rdd.filter(lambda x: len(x) == 2)
        # Mean row by row

        def weight_averaging(x):
            return lambda x: sum(x) / float(len(x))

        weights_rdd = weight_averaging(weights_rdd)

        # weights_mean_rdd = out.groupByKey().mapValues(lambda x: sum(
        #     x) / float(len(x)))  #
        # if (output == 'temp'):
        #     print 'No output directory defined using temp'
        # weights_mean_rdd.collect()
        print 'RNN-LSTM - Total Processing Time (with weight averaging): %f' % (time.time() - start)
        print 'RNN-LSTM - Total Processing Time (with repartition) %f' % (time.time() - start)


        #weights_mean_rdd.saveAsPickleFile(output + '_mean')
        #print weights_mean_rdd.collect()
        # rnn.fit(minibatch_data_rdd)

        # input_data = [[1.1, 2.1, 3.1, 4.1], [3.1, 2.1, 1.1, 4.1], [4.1, 2.1, 1.1, 3.1], [2.1, 3.1, 1.1, 4.1]]
        # rnn.fit_normal(input_data)
    else:
        # print sc.parallelize([("a", 1), ("b", 1), ("a",
        # 1)]).reduceByKey(mean).collect()
        if 'reduce' in load_op :
            print 'REDUCING'
            weights_rdd = sc.pickleFile(input_path + '_raw', partitions)
            print weights_rdd.collect()
            out = weights_rdd.filter(lambda x: len(x) == 2)

            # Mean row by row
            c = out.groupByKey().mapValues(lambda x: sum(
                x) / float(len(x))).collect()  #
            #c = out.reduceByKey(mean).collect()
            for i, d in enumerate(c):
                print i, " - ", d

                print
                print
            if 'save' in load_op:
                if (output == 'temp'):
                    print 'No output directory defined using temp'
                weights_mean_rdd.saveAsPickleFile(output + '_mean')
                print weights_mean_rdd.collect()
        # RIPRENDERE DA QUI
    sys.exit(0)


main()
