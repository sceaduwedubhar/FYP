import tensorflow as tf
import tensorflow.contrib as rnn

class HirerachicalEncoder(object):
    lstm_cell = rnn.rnn.BasicLSTMCell(size,forget_bias=0.0,state_is_tuple=True)

    def __int__(self,num_sentword,num_sent,size,num_layers,batch_size):
        self.num_sentword = num_sentword
        self.numsent = num_sent
        self.size = size
        self.num_layers = num_layers
        self.batch_size = batch_size

    def lstm_cell():
        lstm = rnn.rnn.BasicLSTMCell(size)
        return rnn.rnn.MultiRNNCell([lstm for _ in range(num_layers)])


    def sentEncoder(self):

        return

    def docEncoder(self):
        return
