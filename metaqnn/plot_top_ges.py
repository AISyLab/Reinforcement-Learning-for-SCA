import argparse
import os

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

from .grammar import q_learner
from .training import tensorflow_runner
from .training.one_cycle_lr import OneCycleLR


def get_ge_data(model, results_dir, model_prefix):
    _model = __import__(
        'models.' + model,
        globals(),
        locals(),
        ['state_space_parameters', 'hyper_parameters'],
        0
    )
    hp = _model.hyper_parameters
    ssp = _model.state_space_parameters

    reward_small = results_dir.rstrip('/ ').endswith("_rs")
    qstore = q_learner.QValues()
    qstore.load_q_values(os.path.join(results_dir, 'qlearner_logs', 'q_values.csv'))
    replay_dic = pd.read_csv(os.path.join(results_dir, 'qlearner_logs', 'replay_database.csv'))
    ql = q_learner.QLearner(hp, ssp, 0.0, qstore=qstore, replay_dictionary=replay_dic, reward_small=reward_small)
    tf_runner = tensorflow_runner.TensorFlowRunner(hyper_parameters=hp, state_space_parameters=ssp)

    replay_dic['reward'] = replay_dic.apply(
        lambda row: ql.metrics_to_reward(*ql.get_metrics_from_replay(row['net'])),
        axis='columns'
    )

    deduplicated = replay_dic.drop_duplicates(subset='net', keep='first')
    results_sorted = deduplicated.sort_values(by=['reward'], ascending=False)
    top_networks = results_sorted.head(1)['ix_q_value_update'].values.tolist()
    top_model = keras.models.load_model(
        os.path.join(results_dir, 'trained_models', f'{model_prefix}_{top_networks[0]:04}.h5')
    )
    return tf_runner.perform_attacks_parallel(top_model.predict(tf_runner.attack_features))


def plot_top_network_ges(model, results_dir, model_prefix, top, ensemble, retrain, combine):
    if combine:
        traces_per_attack = 2000
        attack_amount = 100
        plt.style.use(os.path.dirname(__file__) + '/ge_plot.mplstyle')
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.ylim(-5, 200)
        # plt.ylim(-1, 40)
        plt.xlim(0, traces_per_attack + 1)
        plt.grid(True)

        plt.xlabel('Number of traces')
        plt.ylabel('Mean rank of correct key guess')

        for ext, pre_ext, name in [('hw', 'HW', 'HW Model'), ('hw_rs', 'HW_RS', 'HW Model (RS)'),
                                   ('value', 'Value', 'ID Model'), ('value_rs', 'Value_RS', 'ID Model (RS)')]:
            rk_avg = get_ge_data(
                f"{model}_{ext}",
                os.path.join(results_dir, f"experiment_{ext}"),
                f"{model_prefix}_{pre_ext}"
            )
            if rk_avg.shape[0] < traces_per_attack:
                rk_avg = np.pad(rk_avg, (0, traces_per_attack - rk_avg.shape[0]), 'edge')
            plt.plot(range(1, traces_per_attack + 1), rk_avg, '-', label=name)

        plt.legend()
        plt.savefig(
            os.path.normpath(
                os.path.join(results_dir, f'{model_prefix}_{traces_per_attack:d}trs_{attack_amount:d}att.svg')
            ),
            format='svg', dpi=1200, bbox_inches='tight'
        )
        plt.close()

    else:
        _model = __import__(
            'models.' + model,
            globals(),
            locals(),
            ['state_space_parameters', 'hyper_parameters'],
            0
        )
        hp = _model.hyper_parameters
        ssp = _model.state_space_parameters
        reward_small = results_dir.rstrip('/ ').endswith("_rs")
        qstore = q_learner.QValues()
        qstore.load_q_values(os.path.join(results_dir, 'qlearner_logs', 'q_values.csv'))
        replay_dic = pd.read_csv(os.path.join(results_dir, 'qlearner_logs', 'replay_database.csv'))
        ql = q_learner.QLearner(hp, ssp, 0.0, qstore=qstore, replay_dictionary=replay_dic, reward_small=reward_small)
        tf_runner = tensorflow_runner.TensorFlowRunner(hyper_parameters=hp, state_space_parameters=ssp)

        replay_dic['reward'] = replay_dic.apply(
            lambda row: ql.metrics_to_reward(*ql.get_metrics_from_replay(row['net'])),
            axis='columns'
        )

        deduplicated = replay_dic.drop_duplicates(subset='net', keep='first')
        results_sorted = deduplicated.sort_values(by=['reward'], ascending=False)
        top_networks = results_sorted.head(top)['ix_q_value_update'].values.tolist()
        top_models = [
            keras.models.load_model(os.path.join(results_dir, 'trained_models', f'{model_prefix}_{network:04}.h5'))
            for network in top_networks
        ]

        if retrain:
            for model in top_models:
                reset_weights(model)
                _, _ = tf_runner.train_and_predict(model)

        if ensemble:
            rk_key_avgs = [tf_runner.perform_attacks_parallel(
                np.average(
                    np.array([model.predict(tf_runner.attack_features) for model in top_models]),
                    axis=0,  # We weight the averages by the reward the networks received individually
                    weights=results_sorted.head(top)['reward']
                )
            )]  # Top Weighted Average Ensemble
        else:
            def _annotate_perform_attack(predictions, i):
                print(f"CNN {i + 1}/{top}")
                return tf_runner.perform_attacks_parallel(predictions)

            rk_key_avgs = [
                _annotate_perform_attack(model.predict(tf_runner.attack_features), i)
                for i, model in enumerate(top_models)
            ]  # Top Individual attacks

        traces_per_attack = hp.TRACES_PER_ATTACK
        attack_amount = hp.NUM_ATTACKS

        plt.style.use(os.path.dirname(__file__) + '/ge_plot.mplstyle')
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.ylim(-5, 200)
        # plt.ylim(-1, 40)
        plt.xlim(0, traces_per_attack + 1)
        plt.grid(True)

        plt.xlabel('Number of traces')
        plt.ylabel('Mean rank of correct key guess')

        for rk_avg in rk_key_avgs:
            plt.plot(range(1, traces_per_attack + 1), rk_avg, '-')

        title = f'Guessing Entropy for Ensemble of top 20 CNNs\n' if ensemble else f'Guessing Entropy for top 20 CNNs\n'
        plt.title(
            title +
            f'Up to {traces_per_attack:d} traces averaged over {attack_amount:d} attacks',
            loc='center'
        )

        plt.savefig(
            os.path.normpath(
                os.path.join(results_dir, f'{model_prefix}_{traces_per_attack:d}trs_{attack_amount:d}att.svg')
            ),
            format='svg', dpi=1200, bbox_inches='tight'
        )
        plt.close()

        if ensemble or top == 1:
            print("\n t_GE[0] = ")
            print(np.array2string(
                rk_key_avgs[0],
                formatter={"float_kind": lambda x: f"{x:g}"},
                threshold=hp.TRACES_PER_ATTACK + 1  # Always print full array instead of summary
            ))
            ge_no_to_0 = np.where(rk_key_avgs[0] <= 0)
            print("\n mean GE == 0 with #traces: {}".format(ge_no_to_0[0][0] + 1 if len(ge_no_to_0[0] > 0) else "∞"))


def top_network_bagging_ensemble(hp, ssp, results_dir, model_prefix, top):
    reward_small = results_dir.rstrip('/ ').endswith("_rs")
    qstore = q_learner.QValues()
    qstore.load_q_values(os.path.join(results_dir, 'qlearner_logs', 'q_values.csv'))
    replay_dic = pd.read_csv(os.path.join(results_dir, 'qlearner_logs', 'replay_database.csv'))
    ql = q_learner.QLearner(hp, ssp, 0.0, qstore=qstore, replay_dictionary=replay_dic, reward_small=reward_small)
    tf_runner = tensorflow_runner.TensorFlowRunner(hyper_parameters=hp, state_space_parameters=ssp)

    replay_dic['reward'] = replay_dic.apply(
        lambda row: ql.metrics_to_reward(*ql.get_metrics_from_replay(row['net'])),
        axis='columns'
    )

    deduplicated = replay_dic.drop_duplicates(subset='net', keep='first')
    results_sorted = deduplicated.sort_values(by=['reward'], ascending=False)
    top_networks = results_sorted.head(top)['ix_q_value_update'].values.tolist()
    top_models = [
        keras.models.load_model(os.path.join(results_dir, 'trained_models', f'{model_prefix}_{network:04}.h5'))
        for network in top_networks
    ]
    for model in top_models:
        reset_weights(model)

    rk_key_avg = tf_runner.perform_attacks_parallel(
        np.average(
            np.array([_train_and_predict(tf_runner, model, hp, 9, i + 1, top) for i, model in enumerate(top_models)]),
            axis=0,  # We weight the averages by the reward the networks received individually
            weights=results_sorted.head(top)['reward']
        )
    )  # Top Weighted Average Ensemble

    traces_per_attack = hp.TRACES_PER_ATTACK
    attack_amount = hp.NUM_ATTACKS

    plt.style.use(os.path.dirname(__file__) + '/ge_plot.mplstyle')
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.ylim(-5, 200)
    plt.xlim(0, traces_per_attack + 1)
    plt.grid(True)

    plt.xlabel('Number of traces')
    plt.ylabel('Mean rank of correct key guess')

    plt.plot(range(1, traces_per_attack + 1), rk_key_avg, '-')

    title = f'Guessing Entropy for Ensemble of top 20 CNNs'
    plt.title(
        title +
        f'Up to {traces_per_attack:d} traces averaged over {attack_amount:d} attacks',
        loc='center'
    )

    plt.savefig(
        os.path.normpath(
            os.path.join(results_dir, f'{model_prefix}_{traces_per_attack:d}trs_{attack_amount:d}att.svg')
        ),
        format='svg', dpi=1200, bbox_inches='tight'
    )
    plt.close()

    print("\n t_GE[0] = ")
    print(np.array2string(
        rk_key_avg,
        formatter={"float_kind": lambda x: f"{x:g}"},
        threshold=hp.TRACES_PER_ATTACK + 1  # Always print full array instead of summary
    ))
    ge_no_to_0 = np.where(rk_key_avg <= 0)
    print("\n mean GE == 0 with #traces: {}".format(ge_no_to_0[0][0] + 1 if len(ge_no_to_0[0] > 0) else "∞"))


def _train_and_predict(tf_runner, model, hp, parallel_no=1, model_no=None, total=None):
    print(f"================================== Training Model {model_no} of {total} ==================================")
    choices = np.random.choice(
        np.array(range(tf_runner.attack_features.shape[0])),
        size=hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + hp.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
        replace=True
    )
    features = np.take(tf_runner.features, choices, axis=0)
    labels = np.take(tf_runner.labels, choices, axis=0)

    # features, labels = utils.shuffle_arrays_together(features, labels)

    num_train_traces = hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    training_features = features[:num_train_traces]
    training_labels = to_categorical(
        labels[:num_train_traces], num_classes=hp.NUM_CLASSES
    )

    validation_features = features[num_train_traces:]
    validation_labels = to_categorical(
        labels[num_train_traces:], num_classes=hp.NUM_CLASSES
    )

    model.fit(
        x=training_features, y=training_labels, epochs=hp.MAX_EPOCHS,
        batch_size=hp.TRAIN_BATCH_SIZE * parallel_no,
        validation_data=(validation_features, validation_labels), shuffle=True, callbacks=[
            OneCycleLR(
                max_lr=hp.MAX_LR * parallel_no, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None,
                minimum_momentum=None, verbose=True
            ),
        ]
    )

    return model.predict(tf_runner.attack_features)


def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            if var is not None:
                var.assign(initializer(var.shape, var.dtype))


def main():
    parser = argparse.ArgumentParser()

    # model_pkgpath = os.path.join(os.path.dirname(__file__), 'models')
    model_pkgpath = 'models'
    model_choices = next(os.walk(model_pkgpath))[1]

    parser.add_argument(
        'model',
        help='Model package name. Package should have a hyper_parameters.py and a state_space_parameters.py file.',
        # choices=model_choices
    )

    parser.add_argument(
        'results_dir',
        help='The directory where the results are stored'
    )

    parser.add_argument(
        'trained_model_prefix',
        help='The prefix for the trained model filename'
    )

    parser.add_argument(
        '-t', '--top',
        required=False,
        default=20,
        type=int,
        help='The number of top networks to consider'
    )

    parser.add_argument(
        '-e', '--ensemble',
        action='store_true',
        help='Plot GE of ensemble instead of separately'
    )

    parser.add_argument(
        '-r', '--retrain',
        action='store_true',
        help='Retrain the neural network(s) from scratch'
    )

    parser.add_argument(
        '-c', '--combine',
        action='store_true',
        help='Combine the GE plots of the 4 experiments for 1 dataset'
    )

    args = parser.parse_args()

    plot_top_network_ges(args.model, args.results_dir,
                         args.trained_model_prefix, args.top, args.ensemble, args.retrain, args.combine)
    # top_network_bagging_ensemble(_model.hyper_parameters, _model.state_space_parameters, args.results_dir,
    #                              args.trained_model_prefix, args.top)


if __name__ == '__main__':
    main()
