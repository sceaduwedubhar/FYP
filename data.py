import numpy as np
import tensorflow as tf
import gensim

w2vmodel = gensim.models.Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)



