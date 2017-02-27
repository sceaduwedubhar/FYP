import tensorflow as tf
import tensorflow.contrib as rnn
import numpy as np

keep_prob = 0.5

class encoder(object):
    def __init__(self, num_sentword, num_sent, size, num_layers, batch_size):
        self.num_sentword = num_sentword
        self.num_sent = num_sent
        self.size = size
        self.num_layers = num_layers
        self.batch_size = batch_size

        def multicell(self):
            lstm_cell = rnn.rnn.DropoutWrapper(
               rnn.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True))
            return rnn.rnn.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

        def sentencoder(self):
            cell = multicell(self)
            self._initial_state = cell.zero_state(self.batch_size, tf.float32)

