import tensorflow as tf
import tensorflow.contrib as rnn
import tensorflow.contrib as seq2seq
from six.moves import xrange

vocab_size = 3000000
size = 256
num_layers = 4
bucket = 30
learning_rate = 0.001
batch_size = 100
buckets = [(120, 30), (200, 35), (300, 40), (400, 40), (500, 40)]

class Model():

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 num_samples=512,
                 dtype=tf.float32):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=tf.float32)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=tf.float32)
            output_projection = (w, b)

        def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            # We need to compute the sampled_softmax_loss using 32bit floats to
            # avoid numerical instabilities.
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(inputs, tf.float32)
            return tf.cast(tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,num_samples, self.target_vocab_size), tf.float32)

        softmax_loss_function = sampled_loss

        single_cell = rnn.rnn.GRUCell(size)
        cell = rnn.rnn.MultiRNNCell([single_cell]*num_layers)
        initial_state = cell.zero_state(batch_size,dtype=tf.float32)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs):
            return seq2seq.legacy_seq2seq.embedding_attention_decoder(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=True,
                dtype=tf.float32)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                  name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                  name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[batch_size],
                                                  name="weight{0}".format(i)))
        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                for i in xrange(len(self.decoder_inputs) - 1)]

        self.outputs, self.losses = seq2seq.legacy_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets, lambda x, y: seq2seq_f(x, y),
            softmax_loss_function=softmax_loss_function)