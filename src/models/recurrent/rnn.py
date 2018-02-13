import tensorflow as tf
import lstm


class RNN:
    def __init__(self, settings):
        self.layers = []
        for setting in settings:
            self.layers.append(
                lstm.LSTMLayer(name=setting['layer_name'], num_hidden=setting['num_hidden'],
                               dim_size=setting['dim_size'], batch_size=setting['batch_size'])
            )

    def map_data_by_key(self):
        weight_forget, weight_input, weight_output, weight_c = [], [], [], []
        biases_forget, biases_input, biases_output, biases_c = [], [], [], []
        for layer in self.layers:
            weight_forget.append(layer.weight_forget)
            weight_input.append(layer.weight_input)
            weight_output.append(layer.weight_output)
            weight_c.append(layer.weight_C)
            biases_forget.append(layer.biases_forget)
            biases_input.append(layer.biases_input)
            biases_c.append(layer.biases_C)
            biases_output.append(layer.biases_output)

        return [
            (tf.convert_to_tensor("wf", tf.string), weight_forget),
            (tf.convert_to_tensor("wi", tf.string), weight_input),
            (tf.convert_to_tensor("wo", tf.string), weight_output),
            (tf.convert_to_tensor("wc", tf.string), weight_c),
            (tf.convert_to_tensor("bf", tf.string), biases_forget),
            (tf.convert_to_tensor("bi", tf.string), biases_input),
            (tf.convert_to_tensor("bc", tf.string), biases_c),
            (tf.convert_to_tensor("bo", tf.string), biases_output),
        ]

    def fit_layers(self, input_data):
        state = input_data
        for layer in self.layers:
            state = layer.fit_next(state)
        return state

    def add_layer(self, setting):
        self.layers.append(lstm.LSTMLayer(name=setting['layer_name'], num_hidden=setting['num_hidden'],
                                          dim_size=setting['dim_size'], batch_size=setting['batch_size']))

    def add_layers(self, settings):
        for setting in settings:
            self.layers.append(
                lstm.LSTMLayer(name=setting['layer_name'], num_hidden=setting['num_hidden'],
                               dim_size=setting['dim_size'], batch_size=setting['batch_size'])
            )
