from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List
import numpy as np
from os import path

from metaqnn.grammar.state_enumerator import State
from metaqnn.attack import utils
from metaqnn.training.one_cycle_lr import OneCycleLR

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn import preprocessing


class TensorFlowRunner(object):
    def __init__(self, state_space_parameters, hyper_parameters):
        self.ssp = state_space_parameters
        self.hp = hyper_parameters
        self.key = self.hp.KEY
        self.features = self.hp.TRAIN_TRACES
        self.labels = self.hp.TRAIN_LABELS
        self.attack_features = self.hp.ATTACK_TRACES
        # Only used for some experiments to use part of attack set as validation set
        self.attack_labels = getattr(self.hp, 'ATTACK_LABELS', np.ndarray((0, 0)))
        self.precomputed_byte_values = self.hp.ATTACK_PRECOMPUTED_BYTE_VALUES

        self.split_validation_from_attack = getattr(self.hp, 'VALIDATION_FROM_ATTACK_SET', False)

        scaler = preprocessing.StandardScaler()
        self.features = scaler.fit_transform(self.features)
        self.attack_features = scaler.transform(self.attack_features)

        self.features = self.features.reshape((self.features.shape[0], self.features.shape[1], 1))
        self.attack_features = self.attack_features.reshape(
            (self.attack_features.shape[0], self.attack_features.shape[1], 1)
        )

    @staticmethod
    def compile_model(state_list: List[State], loss, metric_list):
        _optimizer = Adam()  # Learning rate will be handled by OneCycleLR policy
        if len(state_list) < 1:
            raise Exception("Illegal neural net")  # TODO create clearer/better exception (class)

        model = tf.keras.Sequential()
        for state in state_list:
            model.add(state.to_tensorflow_layer())
        model.compile(optimizer=_optimizer, loss=loss, metrics=metric_list)
        return model

    @staticmethod
    def clear_session():
        K.clear_session()

    @staticmethod
    def count_trainable_params(model):
        return np.sum([K.count_params(w) for w in model.trainable_weights])

    @staticmethod
    def get_strategy():
        return tf.distribute.MirroredStrategy()

    def train_and_predict(self, model, parallel_no=1):
        features, labels = utils.shuffle_arrays_together(self.features, self.labels)
        training_features = features[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN]
        training_labels = to_categorical(
            labels[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN], num_classes=self.hp.NUM_CLASSES
        )

        if self.split_validation_from_attack:
            validation_features, validation_labels = utils.shuffle_arrays_together(
                self.attack_features, self.attack_labels
            )
            validation_features = validation_features[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL]
            validation_labels = to_categorical(
                self.attack_labels[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL],
                num_classes=self.hp.NUM_CLASSES
            )
        else:
            validation_features = features[self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN:]
            validation_labels = to_categorical(
                labels[self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN:], num_classes=self.hp.NUM_CLASSES
            )

        model.fit(
            x=training_features, y=training_labels, epochs=self.hp.MAX_EPOCHS,
            batch_size=self.hp.TRAIN_BATCH_SIZE * parallel_no,
            validation_data=(validation_features, validation_labels), shuffle=True, callbacks=[
                OneCycleLR(
                    max_lr=self.hp.MAX_LR * parallel_no, end_percentage=0.2, scale_percentage=0.1,
                    maximum_momentum=None,
                    minimum_momentum=None, verbose=True
                )
            ]
        )

        return (
            model.predict(self.attack_features),
            model.evaluate(x=validation_features, y=validation_labels, batch_size=self.hp.EVAL_BATCH_SIZE)
        )

    def perform_attacks(self, predictions, save_graph: bool = False, filename: str = None, folder: str = None):
        return utils.perform_attacks_precomputed_byte_n(
            self.hp.TRACES_PER_ATTACK, predictions, self.hp.NUM_ATTACKS, self.precomputed_byte_values, self.key,
            self.hp.ATTACK_KEY_BYTE, shuffle=True, save_graph=save_graph, filename=filename, folder=folder
        )

    def perform_attacks_parallel(self, predictions, save_graph: bool = False, filename: str = None, folder: str = None):
        return utils.perform_attacks_precomputed_byte_n_parallel(
            self.hp.TRACES_PER_ATTACK, predictions, self.hp.NUM_ATTACKS, self.precomputed_byte_values, self.key,
            self.hp.ATTACK_KEY_BYTE, shuffle=True, save_graph=save_graph, filename=filename, folder=folder
        )

