import argparse
import os
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .grammar import q_learner


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def main(
        results_dir: str, input_size: int, output_size: int, traces_per_attack: int, max_trainable_reward: int,
        ref_trainable_parameters: int, ref_reward: float
):
    qstore = q_learner.QValues()
    qstore.load_q_values(os.path.join(results_dir, 'qlearner_logs', 'q_values.csv'))
    replay_dic = pd.read_csv(os.path.join(results_dir, 'qlearner_logs', 'replay_database.csv'))
    ssp = AttrDict({
        'input_size': input_size,
        'output_states': output_size,
        'layer_limit': 14,
        'init_utility': 0.3,
        'possible_conv_depths': [2, 4, 8, 16, 32, 64, 128],
        'possible_conv_sizes': [1, 2, 3, 25, 50, 75, 100],
        'possible_pool_sizes': [2, 4, 7, 25, 50, 75, 100],
        'possible_pool_strides': [2, 4, 7, 25, 50],
        'max_fc': 3,
        'possible_fc_sizes': [2, 4, 10, 15, 20, 30],
        'conv_padding': 'SAME',
        'epsilon_schedule': [
            [1.0, 1500], [0.9, 100], [0.8, 100], [0.7, 100], [0.6, 150], [0.5, 150], [0.4, 150], [0.3, 150],
            [0.2, 150], [0.1, 150]
        ]
    })

    ql = q_learner.QLearner(
        AttrDict({
            'ssp': ssp,
            'TRACES_PER_ATTACK': traces_per_attack,
            'MAX_TRAINABLE_PARAMS_FOR_REWARD': max_trainable_reward
        }),
        ssp,
        0.0,
        qstore=qstore,
        replay_dictionary=replay_dic,
        reward_small=results_dir.rstrip('/ ').endswith("_rs")
    )

    metrics = ["accuracy", "GE at 10% traces", "GE at 50% traces", "GE #traces to 0", "trainable paramaters", "reward"]
    (net_string, state_list, accuracy, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent,
     guessing_entropy_no_to_0, trainable_params) = ql.generate_net()
    reward = ql.metrics_to_reward(accuracy, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent,
                                  guessing_entropy_no_to_0, trainable_params)
    iteration = replay_dic[replay_dic['net'] == net_string]
    iteration = iteration['ix_q_value_update'].values[0] if len(iteration.index) > 0 else None

    replay_dic['reward'] = replay_dic.apply(
        lambda row: ql.metrics_to_reward(*ql.get_metrics_from_replay(row['net'])),
        axis='columns'
    )
    deduplicated = replay_dic.drop_duplicates(subset='net', keep='first')
    results_sorted = deduplicated.sort_values(by=['reward'], ascending=False)
    title = os.path.join(*results_dir.split(os.path.sep)[-2:])

    with open(os.path.join(results_dir, 'results_overview.txt'), mode="w") as file:
        file.write(f"Results for {title}\n\n")

        file.write("Best network according to Q-Learning:\n")
        file.write(f"{net_string}\n")
        file.write(f"First found at iteration: {iteration}\n")
        file.write("Metrics:\n")

        file.writelines(
            iterable_as_list(metrics, [
                accuracy, guessing_entropy_at_10_percent, guessing_entropy_at_50_percent, guessing_entropy_no_to_0,
                trainable_params, reward
            ])
        )

        q_values = ql.qstore.to_dataframe()
        file.write(f"\n\nAverage q_value: {q_values['utility'].mean()}\n")
        file.write(f"Average (filtered) q_value: {q_values[q_values['utility'] != ssp.init_utility]['utility'].mean()}")

        file.write("\n\nTop 20 total reward networks:\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000,
                               'display.max_colwidth', 100):
            file.write(str(results_sorted.head(20)))

    rolling = replay_dic.rolling(50).mean()[50:][[
        'reward', 'accuracy', 'guessing_entropy_at_10_percent',
        'guessing_entropy_at_50_percent', 'trainable_parameters'
    ]]
    rolling.index = rolling.index + 1  # Properly align index to iterations (iterations start at 1, index started at 0)
    rolling.to_csv(os.path.join(results_dir, 'rolling.csv'), index_label='iteration')
    epsilon_avg = replay_dic.groupby(by=['epsilon']).mean()[[
        'reward', 'accuracy', 'guessing_entropy_at_10_percent',
        'guessing_entropy_at_50_percent', 'trainable_parameters'
    ]].sort_index(ascending=False)
    epsilon_avg.to_csv(os.path.join(results_dir, 'epsilon.csv'))

    epsilon_max_it = replay_dic[
        ['epsilon', 'ix_q_value_update']
    ].groupby(by=['epsilon']).max().sort_index(ascending=False)
    epsilon_avg['max_ix_q_value'] = epsilon_max_it['ix_q_value_update']

    with open(os.path.join(results_dir, 'epsilon-avg-plot.csv'), 'w') as file:
        prev_it = 0
        file.write('iteration,y\n')
        for epsilon in epsilon_avg.iterrows():
            file.write(f'{prev_it},0\n')
            file.write(f'{prev_it},{epsilon[1]["reward"]}\n')
            prev_it = int(epsilon[1]["max_ix_q_value"])
            file.write(f'{prev_it},{epsilon[1]["reward"]}\n')

        file.write(f'{prev_it},0\n')

    plt.style.use(os.path.dirname(__file__) + '/scatter_plot.mplstyle')
    deduplicated.plot.scatter(x='reward', y='trainable_parameters', c='epsilon', colormap='viridis',
                              figsize=(10, 9), logy=True, xlim=(-0.01, 1), ylim=(1, 80_000_000))
    plt.xlabel('Q-Learning reward')
    plt.ylabel('Number of Trainable Parameters in CNN')
    ax = plt.gca()
    ax.figure.axes[-1].set_ylabel('Epsilon When First Generated')

    if ref_trainable_parameters != 0:
        ax.figure.axes[0].axhline(ref_trainable_parameters, color='red')
    if ref_reward != 0.0:
        ax.figure.axes[0].axvline(ref_reward, color='red')

    # ========================================== Zaid ID Model Values ==========================================
    # ax.figure.axes[0].axhline(16960, color='red')  # Gabzai ASCAD_F Trainable Params
    # ax.figure.axes[0].axvline(0.740807917082917, color='red')  # Gabzai ASCAD_F Value Reward (RS=False)
    # ax.figure.axes[0].axvline(0.5556059378121878, color='red')  # Gabzai ASCAD_F Value Reward (RS=True)

    # ax.figure.axes[0].axhline(87279, color='red')  # Gabzai ASCAD_50 Trainable Params
    # ax.figure.axes[0].axvline(0.7354888684232433, color='red')  # Gabzai ASCAD_50 Value Reward (RS=False)
    # ax.figure.axes[0].axvline(0.5516166513174325, color='red')  # Gabzai ASCAD_50 Value Reward (RS=True)

    # ========================================== Zaid HW Model Values ==========================================
    # ax.figure.axes[0].axhline(14235, color='red')  # Gabzai ASCAD_F HW Trainable Params
    # ax.figure.axes[0].axvline(0.6124074231113610, color='red')  # Gabzai ASCAD_F HW Reward (RS=False)
    # ax.figure.axes[0].axvline(0.6043895885599359, color='red')  # Gabzai ASCAD_F HW Reward (RS=True)

    # ax.figure.axes[0].axhline(82879, color='red')  # Gabzai ASCAD_50 HW Trainable Params
    # ax.figure.axes[0].axvline(0.45632343750000004, color='red')  # Gabzai ASCAD_50 HW Reward (RS=False)
    # ax.figure.axes[0].axvline(0.4735442085286481, color='red')  # Gabzai ASCAD_50 HW Reward (RS=True)

    plt.savefig(
        os.path.join(results_dir, f'{title.replace(os.path.sep, "_")}_scatter.svg'),
        format='svg', dpi=150, bbox_inches='tight'
    )
    plt.close()

    top = results_sorted.head(20)
    top_gb = top.groupby(['epsilon']).count()['net']

    totals = pd.Series(dict(map(lambda x: (x[0], x[1]), ssp['epsilon_schedule'])))
    weighted_gb = top_gb.divide(totals)
    plt.style.use(os.path.dirname(__file__) + '/scatter_plot.mplstyle')
    weighted_gb.iloc[::-1].plot.bar(figsize=(15, 10))
    plt.xlabel('Epsilon')
    plt.ylabel('Percentage of networks generated during that epsilon')
    plt.title(f'Top 100 Networks Q-Learning Reward for {title}')

    plt.savefig(
        os.path.join(results_dir, f'{title.replace(os.path.sep, "_")}_epsilon_percentage.svg'),
        format='svg', dpi=150, bbox_inches='tight'
    )
    plt.close()

    epsilon_avg_plot = pd.read_csv(os.path.join(results_dir, 'epsilon-avg-plot.csv'))

    plt.style.use(os.path.dirname(__file__) + '/perf_plot.mplstyle')
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, max(epsilon_avg_plot['iteration']))
    ax.set_ylim(0, 1)
    ax.plot(
        epsilon_avg_plot['iteration'], epsilon_avg_plot['y'],
        label=None, color='black'
    )
    ax.fill_between(
        epsilon_avg_plot['iteration'], epsilon_avg_plot['y'],
        color='skyblue', alpha=0.4, label='Average Reward Per Epsilon'
    )
    ax.plot(rolling.index, rolling['reward'], color='blue', linewidth=1, label='Rolling Average Reward')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Reward')

    for i, epsilon in enumerate(epsilon_avg.index):
        start_index = 1 + 3 * i
        loc = (epsilon_avg_plot['iteration'][start_index] + epsilon_avg_plot['iteration'][start_index + 1]) / 2
        eps = f"{epsilon:.1f}"
        ax.text(loc, 0.01, f'{eps.lstrip("0")}' if epsilon != 1.0 else f'epsilon = {eps}', horizontalalignment='center')

    plt.legend()
    plt.savefig(
        os.path.join(results_dir, f'{title.replace(os.path.sep, "_")}_qlearning_performance.svg'),
        format='svg', dpi=150, bbox_inches='tight'
    )
    plt.close()

    return replay_dic


def iterable_as_list(descriptions: iter, dictionary: iter) -> Generator[str, str, None]:
    for description, el in zip(descriptions, dictionary):
        yield f"- {description}: {el}\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-t', '--trainable-params',
        help='The number of trainable parameters for the reference neural network',
        default=0,
        type=int
    )

    parser.add_argument(
        '-r', '--reward',
        help='The Q-learning reward for the reference neural network',
        default=0.0,
        type=float
    )

    parser.add_argument(
        'results_dir',
        help='Directory with results of an experiment'
    )
    parser.add_argument(
        'input_size',
        help='The input layer size',
        default=700,
        type=int
    )
    parser.add_argument(
        'output_size',
        help='The number of output classes',
        default=256,
        choices=[9, 256],
        type=int
    )
    parser.add_argument(
        'traces_per_attack',
        help='The number of traces used per attack',
        type=int
    )

    parser.add_argument(
        'max_trainable_reward',
        help='The maximum number of trainable parameters for a reward',
        type=int,
        default=20_000_000
    )

    args = parser.parse_args()

    subdirs = next(os.walk(args.results_dir))[1]
    if np.isin(subdirs, ["graphs", "trained_models", "qlearner_logs"]).all():
        results = main(args.results_dir, args.input_size, args.output_size,
                       args.traces_per_attack, args.max_trainable_reward,
                       args.trainable_params, args.reward)
    else:
        print(f"Results dir {args.results_dir} does not contain the required graphs, trained_models and qlearner_logs "
              "subfolders")
