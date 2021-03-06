import tensorflow as tf
import numpy as np
import zipfile
### 1
# X, W_xh = tf.random.normal(shape=(3, 1)), tf.random.normal(shape=(1,4))
# H, W_hh = tf.random.normal(shape=(3, 4)), tf.random.normal(shape=(4,4))
# print(tf.matmul(X, W_xh)+tf.matmul(H, W_hh))
# print(tf.matmul(tf.concat([X,H],axis=1),tf.concat([W_xh,W_hh],axis=0)))
# print(tf.__version__)

### 2
with zipfile.ZipFile('jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
# print(str(corpus_chars[:40]))
corpus_chars = corpus_chars.replace('\n',' ').replace('\r',' ')
corpus_chars = corpus_chars[:10000]
# print(corpus_chars)
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i,char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
# print(vocab_size)
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)