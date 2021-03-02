import math
import os

import numpy as np
import pandas as pd

from . import cnn
from .state_enumerator import State, StateEnumerator
from .state_string_utils import StateStringUtils


class QValues(object):
    """ Stores Q_values with helper functions."""

    def __init__(self):
        self.q = {}

    def load_q_values(self, q_csv_path):
        self.q = {}
        q_csv = pd.read_csv(q_csv_path)
        for row in zip(*[q_csv[col].values.tolist() for col in ['start_layer_type',
                                                                'start_layer_depth',
                                                                'start_filter_depth',
                                                                'start_filter_size',
                                                                'start_stride',
                                                                'start_feature_size',
                                                                'start_fc_size',
                                                                'start_terminate',
                                                                'end_layer_type',
                                                                'end_layer_depth',
                                                                'end_filter_depth',
                                                                'end_filter_size',
                                                                'end_stride',
                                                                'end_feature_size',
                                                                'end_fc_size',
                                                                'end_terminate',
                                                                'utility']]):
            start_state = State(layer_type=row[0],
                                layer_depth=row[1],
                                filter_depth=row[2],
                                filter_size=row[3],
                                stride=row[4],
                                feature_size=row[5],
                                fc_size=row[6],
                                terminate=row[7]).as_tuple()
            end_state = State(layer_type=row[8],
                              layer_depth=row[9],
                              filter_depth=row[10],
                              filter_size=row[11],
                              stride=row[12],
                              feature_size=row[13],
                              fc_size=row[14],
                              terminate=row[15]).as_tuple()
            utility = row[16]

            if start_state not in self.q:
                self.q[start_state] = {'actions': [end_state], 'utilities': [utility]}
            else:
                self.q[start_state]['actions'].append(end_state)
                self.q[start_state]['utilities'].append(utility)

    def to_dataframe(self) -> pd.DataFrame:
        start_layer_type = []
        start_layer_depth = []
        start_filter_depth = []
        start_filter_size = []
        start_stride = []
        start_feature_size = []
        start_fc_size = []
        start_terminate = []
        end_layer_type = []
        end_layer_depth = []
        end_filter_depth = []
        end_filter_size = []
        end_stride = []
        end_feature_size = []
        end_fc_size = []
        end_terminate = []
        utility = []
        for start_state_list in self.q.keys():
            start_state = State(*start_state_list)
            for to_state_ix in range(len(self.q[start_state_list]['actions'])):
                to_state = State(*self.q[start_state_list]['actions'][to_state_ix])
                utility.append(self.q[start_state_list]['utilities'][to_state_ix])
                start_layer_type.append(start_state.layer_type)
                start_layer_depth.append(start_state.layer_depth)
                start_filter_depth.append(start_state.filter_depth)
                start_filter_size.append(start_state.filter_size)
                start_stride.append(start_state.stride)
                start_feature_size.append(start_state.feature_size)
                start_fc_size.append(start_state.fc_size)
                start_terminate.append(start_state.terminate)
                end_layer_type.append(to_state.layer_type)
                end_layer_depth.append(to_state.layer_depth)
                end_filter_depth.append(to_state.filter_depth)
                end_filter_size.append(to_state.filter_size)
                end_stride.append(to_state.stride)
                end_feature_size.append(to_state.feature_size)
                end_fc_size.append(to_state.fc_size)
                end_terminate.append(to_state.terminate)

        return pd.DataFrame({'start_layer_type': start_layer_type,
                             'start_layer_depth': start_layer_depth,
                             'start_filter_depth': start_filter_depth,
                             'start_filter_size': start_filter_size,
                             'start_stride': start_stride,
                             'start_feature_size': start_feature_size,
                             'start_fc_size': start_fc_size,
                             'start_terminate': start_terminate,
                             'end_layer_type': end_layer_type,
                             'end_layer_depth': end_layer_depth,
                             'end_filter_depth': end_filter_depth,
                             'end_filter_size': end_filter_size,
                             'end_stride': end_stride,
                             'end_feature_size': end_feature_size,
                             'end_fc_size': end_fc_size,
                             'end_terminate': end_terminate,
                             'utility': utility})

    def save_to_csv(self, q_csv_path):
        self.to_dataframe().to_csv(q_csv_path, index=False)


class QLearner:
    """ All Q-Learning updates and policy generator
        Args
            state: The starting state for the QLearning Agent
            q_values: A dictionary of q_values --
                            keys: State tuples (State.as_tuple())
                            values: [state list, qvalue list]
            replay_dictionary: A pandas dataframe with columns: 'net' for net strings, and 'accuracy_best_val' for
            best accuracy
                                        and 'accuracy_last_val' for last accuracy achieved
            output_number : number of output neurons
    """

    def __init__(self, hyper_parameters, state_space_parameters, epsilon, state=None, qstore=None,
                 replay_dictionary=pd.DataFrame(columns=[
                     'net', 'accuracy', 'guessing_entropy_at_10_percent', 'guessing_entropy_at_50_percent',
                     'guessing_entropy_no_to_0', 'trainable_parameters', 'ix_q_value_update', 'epsilon', 'time_finished'
                 ]), reward_small=False):

        self.state_list = []

        self.hyper_parameters = hyper_parameters
        self.state_space_parameters = state_space_parameters

        # Class that will expand states for us
        self.enum = StateEnumerator(state_space_parameters)
        self.stringutils = StateStringUtils(state_space_parameters)

        # Starting State
        self.state = State('start', 0, 1, 0, 0, state_space_parameters.input_size, 0, 0, False) if not state else state

        # Cached Q-Values -- used for q learning update and transition
        self.qstore = QValues() if not qstore else qstore
        self.replay_dictionary = replay_dictionary

        self.epsilon = epsilon  # epsilon: parameter for epsilon greedy strategy
        self.reward_small = reward_small

    def update_replay_database(self, new_replay_dic):
        self.replay_dictionary = new_replay_dic

    def generate_net(self):
        # Have Q-Learning agent sample current policy to generate a network and convert network to string format
        self._reset_for_new_walk()
        state_list = self._run_agent()
        net_string = self.stringutils.state_list_to_string(state_list)

        # Check if we have already trained this model
        if net_string in self.replay_dictionary['net'].values:
            (accuracy, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent,
             guessing_entropy_no_to_0, trainable_params) = self.get_metrics_from_replay(net_string)
        else:
            accuracy = 0.0
            guessing_entropy_at_10_percent = 128
            guessing_entropy_at_50_percent = 128
            guessing_entropy_no_to_0 = 255
            trainable_params = 0

        return (
            net_string, state_list,
            accuracy, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent, guessing_entropy_no_to_0,
            trainable_params
        )

    def save_q(self, q_path):
        self.qstore.save_to_csv(os.path.join(q_path, 'q_values.csv'))

    def _reset_for_new_walk(self):
        """Reset the state for a new random walk"""
        # Starting State
        self.state = State('start', 0, 1, 0, 0, self.state_space_parameters.input_size, 0, 0, False)

        # Architecture String
        self.state_list = [self.state.copy()]

    def _run_agent(self):
        """Have Q-Learning agent sample current policy to generate a network"""
        while not self.state.terminate:
            self._transition_q_learning()

        return self.state_list

    def _transition_q_learning(self):
        """Updates self.state according to an epsilon-greedy strategy"""
        if self.state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(self.state, self.qstore.q)

        action_values = self.qstore.q[self.state.as_tuple()]
        # epsilon greedy choice
        if np.random.random() < self.epsilon:
            action = State(*action_values['actions'][np.random.randint(len(action_values['actions']))])
        else:
            max_q_value = max(action_values['utilities'])
            max_q_indexes = [i for i in range(len(action_values['actions'])) if
                             action_values['utilities'][i] == max_q_value]
            max_actions = [action_values['actions'][i] for i in max_q_indexes]
            action = State(*max_actions[np.random.randint(len(max_actions))])

        self.state = action.copy()

        self._post_transition_updates()

    def _post_transition_updates(self):
        # State to go in state list
        state = self.state.copy()

        self.state_list.append(state)

    def sample_replay_for_update(self, iteration):
        # Experience replay to update Q-Values
        for i in range(self.state_space_parameters.replay_number):
            net = np.random.choice(self.replay_dictionary['net'])
            (accuracy, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent, guessing_entropy_no_to_0,
             trainable_params) = self.get_metrics_from_replay(net)
            state_list = self.stringutils.convert_model_string_to_states(cnn.parse('net', net))

            state_list = self.stringutils.remove_drop_out_states(state_list)

            # Convert States so they are bucketed
            state_list = [state.copy() for state in state_list]

            self.update_q_value_sequence(state_list, self.metrics_to_reward(
                accuracy, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent,
                guessing_entropy_no_to_0, trainable_params
            ), iteration)

    def get_metrics_from_replay(self, net):
        net_replay = self.replay_dictionary[self.replay_dictionary['net'] == net]
        accuracy = net_replay['accuracy'].values[0]
        guessing_entropy_at_10_percent = net_replay['guessing_entropy_at_10_percent'].values[0]
        guessing_entropy_at_50_percent = net_replay['guessing_entropy_at_50_percent'].values[0]
        guessing_entropy_no_to_0 = net_replay['guessing_entropy_no_to_0'].values[0]
        trainable_params = net_replay['trainable_parameters'].values[0]
        return (
            accuracy, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent,
            guessing_entropy_no_to_0, trainable_params
        )

    def metrics_to_reward(self, accuracy, ge_at_10_percent, ge_at_50_percent, ge_no_to_0, trainable_params):
        """How to define reward from network (performance) metrics"""
        max_reward = 3  # 2 from below line, 1 from ge_no_to_0

        # R = 0-0.5 + 0-1 + 0-0.5
        reward = (
                accuracy * .5  # 0-0.5
                + (128 - min(ge_at_10_percent, 128)) / 128  # 0-1
                + (128 - min(ge_at_50_percent, 128)) / (128 * 2)  # 0-0.5
        )

        # The network was successful in the key recovery within the set amount of traces
        if ge_no_to_0 is not None and not math.isnan(ge_no_to_0):
            traces_per_attack = self.hyper_parameters.TRACES_PER_ATTACK + 1  # also reward ge of 0 in the max |traces|
            reward += (traces_per_attack - ge_no_to_0) / traces_per_attack  # R += 0-1

        if self.reward_small:
            max_trainable_params = getattr(self.hyper_parameters, 'MAX_TRAINABLE_PARAMS_FOR_REWARD', 20_000_000)

            reward += max(0, (max_trainable_params - trainable_params) / max_trainable_params)  # R += 0-1
            max_reward += 1

        return reward / max_reward

    def update_q_value_sequence(self, states, termination_reward, iteration):
        """Update all Q-Values for a sequence."""
        self._update_q_value(states[-2], states[-1], termination_reward, iteration)
        for i in reversed(range(len(states) - 2)):
            self._update_q_value(states[i], states[i + 1], 0, iteration)

    def _update_q_value(self, start_state, to_state, reward, iteration):
        """ Update a single Q-Value for start_state given the state we transitioned to and the reward. """
        if start_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(start_state, self.qstore.q)
        if to_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(to_state, self.qstore.q)

        actions = self.qstore.q[start_state.as_tuple()]['actions']
        values = self.qstore.q[start_state.as_tuple()]['utilities']

        max_over_next_states = max(self.qstore.q[to_state.as_tuple()]['utilities']) if to_state.terminate != 1 else 0

        action_between_states = to_state.as_tuple()

        action_index = actions.index(action_between_states)
        learning_rate_alpha = 1 / (iteration ** self.state_space_parameters.learning_rate_omega)

        # Q_Learning update rule
        values[action_index] = (  # Q_t+1(s_i,𝑢) =
                values[action_index] +  # Q_t(s_i,𝑢)
                learning_rate_alpha * (  # α
                        reward  # r_t
                        + self.state_space_parameters.discount_factor  # γ
                        * max_over_next_states  # max_{𝑢'∈ 𝒰(s_j)} Q_t(s_j,𝑢')
                        - values[action_index]  # -Q_t(s_i,𝑢)
                )
        )

        self.qstore.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}
