# -*- coding: utf-8 -*-
import utils as utils

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    roc_auc_score, f1_score

import numpy as np
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# print('Using:')
# print('\t\u2022 TensorFlow version:', tf.__version__)
# print('\t\u2022 tf.keras version:', tf.keras.__version__)
# print('\t\u2022 Running on GPU' if tf.train-train.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

# basedir = os.path.join("dataset", "siamese")
basedir = r"D:\my_code_2\my_code\metacovid-siamese-neural-network-main\metacovid-siamese-neural-network-main\scripts\dataset\siamese"
epochs = 20
train_image_list, train_y_list = utils.load_images(basedir, 'train', (100, 100))
print("The train set contains", len(train_image_list))

valid_image_list, valid_y_list = utils.load_images(basedir, 'validation', (100, 100))
print("The valid set contains", len(valid_image_list))

test_image_list, test_y_list = utils.load_images(basedir, 'test', (100, 100))
print("The test set contains", len(test_image_list))

# make train pairs
pairs_train, labels_train, source_labels_train = utils.make_pairs(train_image_list, train_y_list)

# make validation pairs
pairs_val, labels_val, source_labels_val = utils.make_pairs(valid_image_list, valid_y_list)

# make validation pairs
pairs_test, labels_test, source_labels_test = utils.make_pairs(test_image_list, test_y_list)
# print(labels_test)
# pairs_test[x1,x2] label_test[1,0]
# pairs_train[[[x1,x2]],[[x1,x2]],[[x1,x2]]]
x_train_1 = pairs_train[:, 0]  # x1(如何给标签带上)
x_train_2 = pairs_train[:, 1]  # x2
print("number of pairs for train", np.shape(x_train_1)[0])

x_val_1 = pairs_val[:, 0]
x_val_2 = pairs_val[:, 1]
print("number of pairs for validation", np.shape(x_val_1)[0])

x_test_1 = pairs_test[:, 0]
x_test_2 = pairs_test[:, 1]
# print(x_test_1)
# '''
print("number of pairs for test", np.shape(x_test_1)[0])

# utils.visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=4)

tf.compat.v1.reset_default_graph()

SIAMESE_MODEL_FNAME = 'siamese_network.h5'
EMBEDDING_MODEL_FNAME = 'embedding_network.h5'

input_1 = Input((100, 100, 3))
input_2 = Input((100, 100, 3))

embedding_network = tf.keras.models.load_model(EMBEDDING_MODEL_FNAME)
embedding_network.trainable = False

model = tf.keras.Sequential()
for layer in embedding_network.layers:
    model.add(layer)

model.add(Flatten(name='flat'))
model.add(Dense(5120, name='den', activation='sigmoid', kernel_regularizer='l2'))

output_1 = model(input_1)
output_2 = model(input_2)

merge_layer = Lambda(utils.manhattan_distance)([output_1, output_2])
output_layer = Dense(1, activation="sigmoid")(merge_layer)
siamese = Model(inputs=[input_1, input_2], outputs=output_layer)
siamese.summary()

""" callbacks """

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=0.0001)

checkpointer = ModelCheckpoint(filepath='siamese_network.h5', verbose=1,
                               save_best_only=True)

""" train the model """

optimizer = Adam(learning_rate=0.0001)
siamese.compile(loss=utils.loss(1), optimizer=optimizer, metrics=["accuracy"])
# siamese.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

siamese.summary()
history = siamese.fit([x_train_1, x_train_2],
                      labels_train,
                      validation_data=([x_val_1, x_val_2], labels_val),
                      batch_size=1,
                      epochs=epochs,  # 175 for contrastive 100 for cross ent
                      callbacks=[checkpointer, early_stopping, reduce_lr]
                      )
# print()
# Plot the accuracy
# utils.plt_metric(history=history.history, metric="accuracy", title="Model accuracy")
#
# # Plot the constrastive loss
# utils.plt_metric(history=history.history, metric="loss", title="Constrastive Loss")

""" Test the model """
# x_test1的图像和X_test2图像是否匹配
results = siamese.evaluate([x_test_1, x_test_2], labels_test)
print("validation loss, validation acc:", results)

Y_pred = siamese.predict([x_test_1, x_test_2]).squeeze()
# 返回的是TRUE或FALSE,没有标签数据怎么知道他被分到哪儿个类中？
y_pred = Y_pred > 0.5
# x1,和x2是否匹配：匹配1，不匹配0
y_test = labels_test

print("\nEvaluate on validation data")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("ROC AUC:", roc_auc_score(y_test, y_pred, average='weighted'))
print("F1:", f1_score(y_test, y_pred, average='weighted'))
y_pred = [1 if i else 0 for i in y_pred]
pred_labels = []
for i in range(0, len(y_pred)):
    if y_pred[i] == y_test[i]:
        # 匹配
        pred_labels += [source_labels_test[i][0]]
        # print("匹配，类别为：", source_labels_test[i])
        pass
    else:
        # 不匹配
        pred_labels += [source_labels_test[i][1]]
        # print("不匹配，类别为：", source_labels_test[i + 1])
        pass
# print("y_pred", y_pred)
# print("y_test", y_test)

import pandas as pd

df = pd.DataFrame(pred_labels)
df.to_excel(
    "D:\my_code_2\my_code\mh-metacovid-siamese-neural-network\metacovid-siamese-neural-network-main\scripts\pred_resule_{}.xlsx".format(
        epochs))
print(len(y_pred))
print(len(pred_labels))
print(pred_labels)
# source_labels_test
# cm = confusion_matrix(y_test, y_pred)
# cm_display = ConfusionMatrixDisplay(cm, labels_test).plot()
#
# tn, fp, fn, tp = cm.ravel()
# specificity = tn / (tn + fp)
# print("Specificity:", specificity)
#
# tf.keras.backend.clear_session()
# '''
