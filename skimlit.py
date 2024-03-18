# Skim Lit -> classify medical papers into categories (background, results, conclusions, etc)
    # USE takes forever on CPU 

# https://www.kaggle.com/code/bebekjk/skimlit

# BOOM GOT IT TO WORK, Excellent scores 85% accuracy

# Combined Model -> use + bi lstm + ann

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score

import string

categories_int = [0, 1, 2, 3, 4]
categories = ['OBJ', 'METH', 'BACK', 'CONC', 'RES']

with open('Machine Learning 3/pubmed-rct-master/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/train.txt', 'r') as f:
    content = f.readlines()

MAIN_LIST = []
list_ = []
line_number = -1
for each in content:
    dict_ = {}
    if each == '\n':
        MAIN_LIST.append(list_)
        list_ = []
        line_number = -1
    elif 'OBJECTIVE' in each:
        line_number += 1 
        dict_['line_number'] = line_number
        dict_['text'] = each.replace('OBJECTIVE', '').replace('\t', '').replace('\n', '')
        dict_['target'] = 0
        list_.append(dict_)
    elif 'METHODS' in each:
        line_number += 1 
        dict_['line_number'] = line_number
        dict_['text'] = each.replace('METHODS', '').replace('\t', '').replace('\n', '')
        dict_['target'] = 1
        list_.append(dict_)
    elif 'BACKGROUND' in each:
        line_number += 1 
        dict_['line_number'] = line_number
        dict_['text'] = each.replace('BACKGROUND', '').replace('\t', '').replace('\n', '')
        dict_['target'] = 2
        list_.append(dict_)
    elif 'CONCLUSIONS' in each:
        line_number += 1 
        dict_['line_number'] = line_number
        dict_['text'] = each.replace('CONCLUSIONS', '').replace('\t', '').replace('\n', '')
        dict_['target'] = 3
        list_.append(dict_)
    elif 'RESULTS' in each:
        line_number += 1 
        dict_['line_number'] = line_number
        dict_['text'] = each.replace('RESULTS', '').replace('\t', '').replace('\n', '')
        dict_['target'] = 4  
        list_.append(dict_)


line_numbers = []
texts = []
chars = []
targets = []

for each in MAIN_LIST:
    for e in each:
        texts.append(e['text'])
        targets.append(e['target'])
        line_numbers.append(e['line_number'])

# reshape so that each value is a list by itself since OneHotEncoder class demands it
line_numbers = [[each] for each in line_numbers]
# onehotencode list -> line_numbers
onehot = OneHotEncoder(sparse_output=False)
line_numbers = onehot.fit_transform(line_numbers)

# calculate avg, 95th percentiles for words, characters
char_lens = []
word_lens = []
for l in MAIN_LIST:
    for each in l:
        string_ = ''
        char_lens.append(len(each['text']))
        word_lens.append(len(each['text'].split()))
        for c in each['text']:
            string_ += f' {c} '
        chars.append(string_)
print(np.mean(word_lens)) # 26
print(np.percentile(word_lens, 95)) # 55
print(np.mean(char_lens)) # 149
print(np.percentile(char_lens, 95)) # 290

# # split data, manual split since in modelling we have a single output layer which has to correspond to their respective X train vals
split_range = int(0.8 * len(texts))

X_train_texts = texts[:split_range]
X_test_texts = texts[split_range:]

X_train_chars = chars[:split_range]
X_test_chars = chars[split_range:]

X_train_lines = line_numbers[:split_range]
X_test_lines = line_numbers[split_range:]

y_train = targets[:split_range]
y_test = targets[split_range:]

# preprocessing
text_vectorization = tf.keras.layers.experimental.preprocessing.TextVectorization(
    ngrams=1, 
    max_tokens=68000,
    output_mode='int',
    output_sequence_length=55,
    pad_to_max_tokens=True
)
text_vectorization.adapt(texts)
text_embedding = tf.keras.layers.Embedding(
    input_dim=text_vectorization._max_tokens,
    input_length=text_vectorization._output_sequence_length,
    output_dim=128
)

char_vectorization = tf.keras.layers.experimental.preprocessing.TextVectorization(
    ngrams=1, 
    output_mode='int', 
    output_sequence_length=290,
    max_tokens=len(string.ascii_lowercase) + len(string.punctuation) + 2,  # whitespace and <UNK> # no digits on this one its @ symbol
    pad_to_max_tokens=True
)
char_vectorization.adapt(chars)
char_embedding = tf.keras.layers.Embedding(
    input_dim=char_vectorization._max_tokens,
    input_length=char_vectorization._output_sequence_length, 
    output_dim=25
)

# modelling : combined models

# USE (text)
use_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[], dtype='string'),
    hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', trainable=False),
    tf.keras.layers.Dense(units=24, activation='relu')
])
# BI LSTM (chars)
bi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, ), dtype='string'),
    char_vectorization,
    char_embedding,
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, activation='tanh'))
])
# ANN (pos -> lines)
ann_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(31, ), dtype=tf.int32),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu')
])

concat_layer = tf.keras.layers.Concatenate()([use_model.output, bi_lstm_model.output, ann_model.output])
output_layer = tf.keras.layers.Dense(units=5, activation='softmax')(concat_layer)

combined_model = tf.keras.Model(
    inputs=[use_model.input, bi_lstm_model.input, ann_model.input],
    outputs=output_layer
)

combined_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
combined_model.fit(
    x=[np.array(X_train_texts), np.array(X_train_chars), np.array(X_train_lines)],
    y=np.array(y_train), # all y is same, no shuffle 
    verbose=2,
    epochs=15,
    shuffle=True,
    batch_size=128
)

preds = combined_model.predict([np.array(X_test_texts), np.array(X_test_chars), np.array(X_test_lines)])
preds_int = []
for pred in preds:
    preds_int.append(np.argmax(pred))

print(accuracy_score(y_pred=preds_int, y_true=y_test))
