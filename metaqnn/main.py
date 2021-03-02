import argparse
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
from os import path

import cloudpickle
import numpy as np
import pandas as pd

from metaqnn.grammar import q_learner
from metaqnn.training.tensorflow_runner import TensorFlowRunner


class TermColors(object):
    HEADER = '\033[95m'
    YELLOW = '\033[93m'
    OKBLUE = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class QCoordinator(object):
    def __init__(self,
                 list_path,
                 state_space_parameters,
                 hyper_parameters,
                 epsilon=None,
                 number_models=None,
                 reward_small=False):

        print("\n\nRun started at: {}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))

        self.replay_columns = [
            'net',  # Net String
            'accuracy',
            'guessing_entropy_at_10_percent',
            'guessing_entropy_at_50_percent',
            'guessing_entropy_no_to_0',
            'trainable_parameters',  # Amount of trainable params of the network
            'ix_q_value_update',  # Iteration for q value update
            'epsilon',  # For epsilon greedy
            'time_finished'  # UNIX time
        ]

        self.list_path = list_path

        self.replay_dictionary_path = os.path.join(list_path, 'replay_database.csv')
        self.replay_dictionary, self.q_training_step = self.load_replay()

        self.schedule_or_single = False if epsilon else True
        if self.schedule_or_single:
            self.epsilon = state_space_parameters.epsilon_schedule[0][0]
            self.number_models = state_space_parameters.epsilon_schedule[0][1]
        else:
            self.epsilon = epsilon
            self.number_models = number_models if number_models else 10000000000
        self.state_space_parameters = state_space_parameters
        self.hyper_parameters = hyper_parameters

        self.number_q_updates_per_train = 100
        self.reward_small = reward_small

        self.list_path = list_path
        self.qlearner = self.load_qlearner()
        self.tf_runner = TensorFlowRunner(self.state_space_parameters, self.hyper_parameters)
        self.ten_percent_index = self.hyper_parameters.TRACES_PER_ATTACK // 10 - 1
        self.fifty_percent_index = self.hyper_parameters.TRACES_PER_ATTACK // 2 - 1

        while not self.check_reached_limit():
            self.train_new_net()

        print('{}{}Experiment Complete{}'.format(TermColors.BOLD, TermColors.OKGREEN, TermColors.RESET))

    def train_new_net(self):
        net, net_to_run, iteration = self.generate_new_network()
        print('{}Training net:\n{}\nIteration {:d}, Epsilon {:f}: [Network {:d}/{:d}]{}'.format(
            TermColors.OKBLUE, net_to_run, iteration, self.epsilon, self.number_trained_unique(self.epsilon),
            self.number_models, TermColors.RESET
        ))

        parent, child = mp.Pipe(duplex=False)
        process = mp.Process(target=self._train_and_predict, args=(
            cloudpickle.dumps(self.tf_runner),
            net,
            self.hyper_parameters.MODEL_NAME,
            iteration,
            child
        ))

        process.start()
        (predictions, (test_loss, test_accuracy)), trainable_params = cloudpickle.loads(parent.recv())
        process.join()

        guessing_entropy = self.tf_runner.perform_attacks_parallel(
            predictions, save_graph=True, filename=f"{self.hyper_parameters.MODEL_NAME}_{iteration:04}",
            folder=f"{self.hyper_parameters.BULK_ROOT}/graphs"
        )

        ge_no_to_0 = np.where(guessing_entropy <= 0)[0]

        self.incorporate_trained_net(
            net_to_run, float(test_accuracy), guessing_entropy[self.ten_percent_index],
            guessing_entropy[self.fifty_percent_index], ge_no_to_0[0] if len(ge_no_to_0) > 0 else None,
            trainable_params, float(self.epsilon), [iteration]
        )

    @staticmethod
    def _train_and_predict(tf_runner, net, model_name, iteration, return_pipe):
        tf_runner = cloudpickle.loads(tf_runner)
        strategy = tf_runner.get_strategy()
        parallel_no = strategy.num_replicas_in_sync
        if parallel_no is None:
            parallel_no = 1

        with strategy.scope():
            model = tf_runner.compile_model(net, loss='categorical_crossentropy', metric_list=['accuracy'])
            model.summary()
            trainable_params = tf_runner.count_trainable_params(model)

            return_pipe.send(cloudpickle.dumps((
                tf_runner.train_and_predict(model, parallel_no),
                trainable_params
            )))

        model.save(path.normpath(f"{tf_runner.hp.TRAINED_MODEL_DIR}/{model_name}_{iteration:04}.h5"))

    def load_replay(self):
        if os.path.isfile(self.replay_dictionary_path):
            print('Found replay dictionary')
            replay_dic = pd.read_csv(self.replay_dictionary_path)
            q_training_step = max(replay_dic.ix_q_value_update)
        else:
            replay_dic = pd.DataFrame(columns=self.replay_columns)
            q_training_step = 0
        return replay_dic, q_training_step

    def load_qlearner(self):
        # Load previous q_values
        if os.path.isfile(os.path.join(self.list_path, 'q_values.csv')):
            print('Found q values')
            qstore = q_learner.QValues()
            qstore.load_q_values(os.path.join(self.list_path, 'q_values.csv'))
        else:
            qstore = None

        ql = q_learner.QLearner(self.hyper_parameters,
                                self.state_space_parameters,
                                self.epsilon,
                                qstore=qstore,
                                replay_dictionary=self.replay_dictionary,
                                reward_small=self.reward_small)

        return ql

    @staticmethod
    def filter_replay_for_first_run(replay):
        """ Order replay by iteration, then remove duplicate nets keeping the first"""
        temp = replay.sort_values(['ix_q_value_update']).reset_index(drop=True).copy()
        return temp.drop_duplicates(['net'])

    def number_trained_unique(self, epsilon=None):
        """Epsilon defaults to the minimum"""
        replay_unique = self.filter_replay_for_first_run(self.replay_dictionary)
        eps = epsilon if epsilon else min(replay_unique.epsilon.values)
        replay_unique = replay_unique[replay_unique.epsilon == eps]
        return len(replay_unique)

    def check_reached_limit(self):
        """ Returns True if the experiment is complete"""
        if len(self.replay_dictionary):
            completed_current = self.number_trained_unique(self.epsilon) >= self.number_models

            if completed_current:
                if self.schedule_or_single:
                    # Loop through epsilon schedule, If we find an epsilon that isn't trained, start using that.
                    completed_experiment = True
                    for epsilon, num_models in self.state_space_parameters.epsilon_schedule:
                        if self.number_trained_unique(epsilon) < num_models:
                            self.epsilon = epsilon
                            self.number_models = num_models
                            self.qlearner = self.load_qlearner()
                            completed_experiment = False

                            break

                else:
                    completed_experiment = True

                return completed_experiment

            else:
                return False

    def generate_new_network(self):
        try:
            (net_string, net, accuracy, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent,
             guessing_entropy_no_to_0, trainable_params) = self.qlearner.generate_net()

            # We have already trained this net
            if net_string in self.replay_dictionary.net.values:
                self.q_training_step += 1
                self.incorporate_trained_net(
                    net_string,
                    accuracy,
                    guessing_entropy_at_10_percent,
                    guessing_entropy_at_50_percent,
                    guessing_entropy_no_to_0,
                    trainable_params,
                    self.epsilon,
                    [self.q_training_step]
                )
                return self.generate_new_network()
            else:
                self.q_training_step += 1
                return net, net_string, self.q_training_step

        except Exception:
            print(traceback.print_exc())
            sys.exit(1)

    def incorporate_trained_net(self, net_string, accuracy, ge_at_10_percent, ge_at_50_percent, ge_no_to_0,
                                trainable_params, epsilon, iterations):

        try:
            # If we sampled the same net many times, we should add them each into the replay database
            for train_iter in iterations:
                self.replay_dictionary = pd.concat([
                    self.replay_dictionary,
                    pd.DataFrame({
                        'net': [net_string],
                        'accuracy': [accuracy],
                        'guessing_entropy_at_10_percent': [ge_at_10_percent],
                        'guessing_entropy_at_50_percent': [ge_at_50_percent],
                        'guessing_entropy_no_to_0': [ge_no_to_0],
                        'trainable_parameters': [trainable_params],
                        'ix_q_value_update': [train_iter],
                        'epsilon': [epsilon],
                        'time_finished': [time.time()]
                    })
                ])
                self.replay_dictionary.to_csv(self.replay_dictionary_path, index=False, columns=self.replay_columns)

            self.qlearner.update_replay_database(self.replay_dictionary)
            for train_iter in iterations:
                self.qlearner.sample_replay_for_update(train_iter)
            self.qlearner.save_q(self.list_path)
            if ge_no_to_0 is None or math.isnan(ge_no_to_0):
                ge_no_to_0 = "∞"

            print('{}Incorporated net, acc: {:f}, t_GE <= 0: {}, net(trainable_params={}):\n{}{}'.format(
                TermColors.YELLOW, accuracy, ge_no_to_0, trainable_params, net_string, TermColors.RESET
            ))
        except Exception:
            print(traceback.print_exc())


def main():
    parser = argparse.ArgumentParser()

    model_pkgpath = 'models'
    model_choices = next(os.walk(model_pkgpath))[1]

    parser.add_argument(
        'model',
        help='Model package name. Package should have a hyper_parameters.py and a state_space_parameters.py file.',
        choices=model_choices
    )

    parser.add_argument(
        '--reward-small',
        help='Reward having a network with little trainable parameters',
        action='store_true'
    )
    parser.add_argument('-eps', '--epsilon', help='For Epsilon Greedy Strategy', type=float)
    parser.add_argument('-nmt', '--number_models_to_train', type=int,
                        help='How many models for this epsilon do you want to train.')

    args = parser.parse_args()

    _model = __import__(
        'models.' + args.model,
        globals(),
        locals(),
        ['state_space_parameters', 'hyper_parameters'],
        0
    )

    factory = QCoordinator(
        path.normpath(path.join(_model.hyper_parameters.BULK_ROOT, "qlearner_logs")),
        _model.state_space_parameters,
        _model.hyper_parameters,
        args.epsilon,
        args.number_models_to_train,
        args.reward_small
    )


if __name__ == '__main__':
    main()
