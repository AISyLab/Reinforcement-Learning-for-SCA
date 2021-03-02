import math

import tensorflow as tf


class State(object):
    def __init__(self,
                 layer_type=None,  # String -- conv, pool, fc, softmax
                 layer_depth=None,  # Current depth of network
                 filter_depth=0,  # Used for conv, 0 when not conv
                 filter_size=0,  # Used for conv and pool, 0 otherwise
                 stride=0,  # Used for conv and pool, 0 otherwise
                 feature_size=None,  # Used for convolutional blocks, 0 otherwise
                 fc_size=0,  # Used for fc and softmax -- number of neurons in layer
                 dropout_ratio=0.0,  # Used for dropout layer -- percentage of weights to zero
                 terminate=False):  # can be constructed from a list instead, list takes precedent
        self.layer_type = layer_type
        self.layer_depth = layer_depth
        self.filter_depth = filter_depth
        self.filter_size = filter_size
        self.stride = stride
        self.feature_size = feature_size
        self.fc_size = fc_size
        self.dropout_ratio = dropout_ratio
        self.terminate = terminate

    def __str__(self):
        return str(self.as_tuple())

    def as_tuple(self):
        return (self.layer_type,
                self.layer_depth,
                self.filter_depth,
                self.filter_size,
                self.stride,
                self.feature_size,
                self.fc_size,
                self.dropout_ratio,
                self.terminate)

    def as_list(self):
        return list(self.as_tuple())

    def to_tensorflow_layer(self):  # TODO Make activation configurable
        if self.terminate:
            return tf.keras.layers.Dense(self.fc_size, kernel_initializer='glorot_uniform', activation='softmax')
        elif self.layer_type == 'start':
            return tf.keras.layers.InputLayer(input_shape=(self.feature_size, 1))
        elif self.layer_type == 'conv':
            return tf.keras.layers.Convolution1D(filters=self.filter_depth, kernel_size=self.filter_size,
                                                 strides=self.stride, kernel_initializer='he_uniform',
                                                 activation='selu', padding='same')
        elif self.layer_type == 'gap':
            return tf.keras.layers.GlobalAveragePooling1D()
        elif self.layer_type == 'pool':
            return tf.keras.layers.AveragePooling1D(pool_size=self.filter_size, strides=self.stride)
        elif self.layer_type == 'flatten':
            return tf.keras.layers.Flatten()
        elif self.layer_type == 'max_pool':
            return tf.keras.layers.MaxPooling1D(pool_size=self.filter_size, strides=self.stride)
        elif self.layer_type == 'fc':
            return tf.keras.layers.Dense(self.fc_size, kernel_initializer='he_uniform', activation='selu')
        elif self.layer_type == 'batch_norm':
            return tf.keras.layers.BatchNormalization()
        elif self.layer_type == 'dropout':
            return tf.keras.layers.Dropout(self.dropout_ratio)
        return None

    def copy(self):
        return State(self.layer_type,
                     self.layer_depth,
                     self.filter_depth,
                     self.filter_size,
                     self.stride,
                     self.feature_size,
                     self.fc_size,
                     self.dropout_ratio,
                     self.terminate)


class StateEnumerator(object):
    """Class that deals with:
            Enumerating States (defining their possible transitions)
    """

    def __init__(self, state_space_parameters):
        # Limits
        self.ssp = state_space_parameters
        self.layer_limit = state_space_parameters.layer_limit

        self.output_states = state_space_parameters.output_states

    def enumerate_state(self, state: State, q_values):
        """Defines all state transitions, populates q_values where actions are valid
        Legal Transitions:
           conv         -> pool                         (IF state.layer_depth < layer_limit)
           conv         -> batch_norm, softmax, gap     (Always)
           batch_norm   ->
           pool         -> conv,                        (If state.layer_depth < layer_limit)
           pool         -> fc,                          (If state.layer_depth < layer_limit)
           pool         -> softmax, gap                 (Always)
           fc           -> fc                           (If state.layer_depth < layer_limit AND state.filter_depth < 3)
           fc           -> softmax                      (Always)
           gap          -> softmax                      (Always)
        Updates: q_values and returns q_values
        """
        actions = []

        if not state.terminate:
            # If we are at the layer limit, we can only go to softmax
            if state.layer_type in ['flatten', 'fc']:
                actions += [
                    State(
                        layer_type='softmax',
                        layer_depth=state.layer_depth + 1,
                        filter_depth=state.filter_depth,
                        filter_size=state.filter_size,
                        stride=state.stride,
                        feature_size=0,
                        fc_size=self.output_states,
                        dropout_ratio=state.dropout_ratio,
                        terminate=True
                    )
                ]

            if state.layer_depth < self.layer_limit:

                # Conv states -- iterate through all possible depths, filter sizes, and strides
                if state.layer_type in ['start', 'pool']:
                    for depth in self.ssp.possible_conv_depths:
                        for filter_size in self._possible_conv_sizes(state.feature_size):
                            actions += [
                                State(
                                    layer_type='conv',
                                    layer_depth=state.layer_depth + 1,
                                    filter_depth=depth,
                                    filter_size=filter_size,
                                    stride=1,
                                    feature_size=state.feature_size if self.ssp.conv_padding == 'SAME'
                                    else self.calc_new_feature_size(state.feature_size, filter_size, 1),
                                    fc_size=0
                                )
                            ]

                if state.layer_type == 'conv':
                    actions += [
                        State(
                            layer_type='batch_norm',
                            layer_depth=state.layer_depth + 1,
                            feature_size=state.feature_size,
                            stride=0
                        )
                    ]

                # Global Average Pooling States
                if state.layer_type in ['conv', 'pool']:
                    actions += [
                        State(
                            layer_type='gap',
                            layer_depth=state.layer_depth + 1,
                            filter_depth=0,
                            filter_size=0,
                            stride=0,
                            feature_size=state.feature_size if state.layer_type != 'conv' else state.filter_depth,
                            fc_size=0
                        )
                    ]

                # pool states -- iterate through all possible filter sizes and strides
                if (state.layer_type in ['conv', 'batch_norm'] or
                        (state.layer_type == 'pool' and self.ssp.allow_consecutive_pooling) or
                        (state.layer_type == 'start' and self.ssp.allow_initial_pooling)):
                    for filter_size in self._possible_pool_sizes(state.feature_size):
                        for stride in self._possible_pool_strides(filter_size):
                            actions += [
                                State(
                                    layer_type='pool',
                                    layer_depth=state.layer_depth + 1,
                                    filter_depth=0,
                                    filter_size=filter_size,
                                    stride=stride,
                                    feature_size=self.calc_new_feature_size(state.feature_size, filter_size, stride),
                                    fc_size=0
                                )
                            ]

                if state.layer_type in ['pool']:
                    actions += [
                        State(
                            layer_type='flatten',
                            layer_depth=state.layer_depth + 1,
                            filter_depth=0,
                            filter_size=0,
                            stride=0,
                            feature_size=state.feature_size * state.stride,
                            fc_size=0
                        )
                    ]

                # FC States -- iterate through all possible fc sizes
                if state.layer_type in ['flatten', 'gap']:
                    for fc_size in self._possible_fc_size(state):
                        actions += [
                            State(
                                layer_type='fc',
                                layer_depth=state.layer_depth + 1,
                                filter_depth=0,
                                filter_size=0,
                                stride=0,
                                feature_size=0,
                                fc_size=fc_size
                            )
                        ]

                # FC -> FC States
                if state.layer_type == 'fc' and state.filter_depth < self.ssp.max_fc - 1:
                    for fc_size in self._possible_fc_size(state):
                        actions += [
                            State(
                                layer_type='fc',
                                layer_depth=state.layer_depth + 1,
                                filter_depth=state.filter_depth + 1,
                                filter_size=0,
                                stride=0,
                                feature_size=0,
                                fc_size=fc_size
                            )
                        ]

            if len(actions) == 0:
                actions += [
                    State(
                        layer_type='flatten',
                        layer_depth=state.layer_depth + 1,
                        filter_depth=0,
                        filter_size=0,
                        stride=0,
                        feature_size=state.feature_size,
                        fc_size=0
                    )
                ]

        # Add states to transition and q_value dictionary
        q_values[state.as_tuple()] = {
            'actions': [to_state.as_tuple() for to_state in actions],
            'utilities': [self.ssp.init_utility for _ in range(len(actions))]
        }
        return q_values

    @staticmethod
    def calc_new_feature_size(feature_size, filter_size, stride):
        """Returns new image size given previous image size and filter parameters"""
        new_size = int(math.ceil(float(feature_size - filter_size + 1) / float(stride)))
        return new_size

    def _possible_conv_sizes(self, feature_size):
        return [conv for conv in self.ssp.possible_conv_sizes if conv < feature_size]

    def _possible_pool_sizes(self, feature_size):
        return [pool for pool in self.ssp.possible_pool_sizes if pool < feature_size]

    def _possible_pool_strides(self, filter_size):
        return [stride for stride in self.ssp.possible_pool_strides if stride <= filter_size]

    def _possible_fc_size(self, state):
        """Return a list of possible FC sizes given the current state"""
        if state.layer_type == 'fc':
            return [i for i in self.ssp.possible_fc_sizes if i <= state.fc_size]
        return self.ssp.possible_fc_sizes
