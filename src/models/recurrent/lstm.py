import tensorflow as tf


def create_variable(name, shape, dtype, initializer=tf.truncated_normal_initializer,
                    weight_decay=None, loss=tf.nn.l2_loss):
    with tf.device("/cpu:0"):
        var = tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer())

    if weight_decay:
        wd = loss(var) * weight_decay
        tf.add_to_collection("weight_decay", wd)

    return var


class LSTMLayer:
    def __init__(self, name, num_hidden, dim_size, batch_size):
        self.shape = [batch_size, num_hidden, dim_size]
        self.batch_size = batch_size

        self.node_name = name
        self.state = []

        self.WEIGHT_STATE = 0
        self.WEIGHT_INPUT = 1

        with tf.variable_scope(name):
            self.weight_forget = [
                create_variable(name="weights_forget_h",
                                shape=[num_hidden, num_hidden],
                                dtype=tf.float32),
                create_variable(name="weights_forget_x",
                                shape=[dim_size, num_hidden],
                                dtype=tf.float32)
            ]
            self.biases_forget = create_variable(name="bias_forget",
                                                 shape=[num_hidden],
                                                 dtype=tf.float32)

            self.weight_input = [
                create_variable(name="weights_input_h",
                                shape=[num_hidden, num_hidden],
                                dtype=tf.float32),
                create_variable(name="weights_input_x",
                                shape=[dim_size, num_hidden],
                                dtype=tf.float32)
            ]
            self.biases_input = create_variable(name="bias_input",
                                                shape=[num_hidden],
                                                dtype=tf.float32)

            self.weight_C = [
                create_variable(name="weights_C_h",
                                shape=[num_hidden, num_hidden],
                                dtype=tf.float32),
                create_variable(name="weights_C_x",
                                shape=[dim_size, num_hidden],
                                dtype=tf.float32)
                ]
            self.biases_C = create_variable(name="bias_C",
                                            shape=[num_hidden],
                                            dtype=tf.float32)

            self.weight_output = [
                create_variable(name="weights_output_h",
                                shape=[num_hidden, num_hidden],
                                dtype=tf.float32),
                create_variable(name="weights_output_x",
                                shape=[dim_size, num_hidden],
                                dtype=tf.float32)
                ]
            self.biases_output = create_variable(name="bias_output",
                                                 shape=[num_hidden],
                                                 dtype=tf.float32)

            self.ht = create_variable(name="state",
                                      shape=[batch_size, num_hidden],
                                      dtype=tf.float32)
            self.Ct = create_variable(name="context_state",
                                      shape=[batch_size, num_hidden],
                                      dtype=tf.float32)

            # PEP8 wants class members declared in __init__, first usage in layer computation
            self.ft = None
            self.it = None
            self.c_ta = None

    def layer_step(self, weight, bias, input_data, activation, name):
        out = tf.matmul(self.ht, weight[self.WEIGHT_STATE], name="MATMUL_STATE") + \
              tf.matmul(input_data, weight[self.WEIGHT_INPUT], name="MATMUL_INPUT") + bias
        return activation(out, name='{}_{}'.format(name, activation.__name__))

    def forget_gate_layer(self, input_data):
        self.ft = self.layer_step(self.weight_forget, self.biases_forget, input_data,
                               tf.sigmoid, 'ft_{}'.format(self.node_name))

    def input_gate_layer(self, input_data):
        self.it = self.layer_step(self.weight_input, self.biases_input, input_data,
                                  tf.sigmoid, 'it_{}'.format(self.node_name))

        self.c_ta = self.layer_step(self.weight_C, self.biases_C, input_data, tf.tanh, 'Cat_{}'.format(self.node_name))

    def update_old_cell_state_layer(self):
        self.Ct = tf.add(self.ft * self.Ct, self.it * self.c_ta, name='Ct_{}'.format(self.node_name))

    def to_output_layer(self, input_data):
        ot = self.layer_step(self.weight_output, self.biases_output, input_data,
                             tf.sigmoid, 'Ot_{}'.format(self.node_name))
        self.ht = tf.multiply(ot, tf.tanh(self.Ct), 'ht_{}'.format(self.node_name))

    def train_layer(self, input_data):
        with tf.name_scope('forget_gate_layer'):
            self.forget_gate_layer(input_data)

        with tf.name_scope('input_gate_layer'):
            self.input_gate_layer(input_data)

        with tf.name_scope('update_old_cell_state_layer'):
            self.update_old_cell_state_layer()

        with tf.name_scope('to_output_layer'):
            self.to_output_layer(input_data)

    def restore_state(self):
        self.ht = self.state[-1][0]
        self.Ct = self.state[-1][1]

    def fit_next(self, data, train=True):
        # input_data_t = tf.transpose([data], name="input_data")
        self.train_layer(data)

        if train:
            self.state.append((self.ht, self.Ct))  # store the state of each step
        else:
            self.restore_state()
        return self.ht
