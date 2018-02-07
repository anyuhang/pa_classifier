import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
lines = ['Some title', 
         'A longer title', 
         'An even longer title', 
         'This is longer than doc length']

MAX_DOCUMENT_LENGTH = 5  
PADWORD = 'ZZZZ'

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
vocab_processor.fit(lines)
with gfile.Open('vocab.tsv', 'wb') as f:
   f.write("{}\n".format(PADWORD))
   for word, index in vocab_processor.vocabulary_._mapping.iteritems():
     f.write("{}\n".format(word))

N_WORDS = len(vocab_processor.vocabulary_)   
table = lookup.index_table_from_file(
  vocabulary_file='vocab.tsv', num_oov_buckets=1, vocab_size=None, default_value=-1)
numbers = table.lookup(tf.constant('Some title'.split()))
with tf.Session() as sess:
  tf.tables_initializer().run()
  print "{} --> {}".format(lines[0], numbers.eval())
  
titles = tf.constant(lines)
words = tf.string_split(titles)        
densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
numbers = table.lookup(densewords)  
padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
padded = tf.pad(numbers, padding)
sliced = tf.slice(padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])
