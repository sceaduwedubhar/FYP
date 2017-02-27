import tensorflow as tf
import tensorflow.contrib as rnn

class HirerachicalEncoder(object):

    def __int__(self,num_sentword,num_sent,size,num_layers,batch_size):
        self.num_sentword = num_sentword
        self.numsent = num_sent
        self.size = size
        self.num_layers = num_layers
        self.batch_size = batch_size

    def lstm_cell(self):
        lstm = rnn.rnn.BasicLSTMCell(self.size,state_is_tuple=True)
        return rnn.rnn.MultiRNNCell([lstm for _ in range(self.num_layers)])


    def sentEncoder(self):

        return

    def docEncoder(self):
        return
