"""
Given a logfile, plot a graph
"""
import matplotlib.pyplot as plt
import json
import argparse
import numpy as np


def plot_reward(logfile, title, min_y, max_y, max_x):
    """
    logfiles separated by : are combined
    logfiles separated by , go in separate plots
    (: binds tighter than ,)
    """
    logfiles = logfile
    split_logfiles = logfiles.split(',')
    for j, logfile_groups in enumerate(split_logfiles):
        epoch = []
        reward_both = []
        reward_A = []
        reward_B = []
        test_reward_both = []
        test_reward_A = []
        test_reward_B = []
        reward_names = {
            'train reward both': reward_both,
            'train reward A': reward_A,
            'train reward B': reward_B,
            'test reward both': test_reward_both,
            'test reward A': test_reward_A,
            'test reward B': test_reward_B,
        }

        for logfile in logfile_groups.split(':'):
            with open(logfile, 'r') as f:
                for n, line in enumerate(f):
                    if n == 0:
                        print(logfile, line)
                        continue  # skip first line
                    line = line.strip()
                    if line == '':
                        continue
                    d = json.loads(line)
                    if max_x is not None and d['episode'] > max_x:
                        continue
                    epoch.append(int(d['episode']))
                    reward_both.append(float(d['avg_reward_0']))
                    if 'avg_reward_A' in d:
                        reward_A.append(float(d['avg_reward_A']))
                    if 'avg_reward_B' in d:
                        reward_B.append(float(d['avg_reward_B']))
                    if 'test_reward' in d:
                        test_reward_both.append(d['test_reward'])
                    if 'test_reward_A' in d:
                        test_reward_A.append(float(d['test_reward_A']))
                    if 'test_reward_B' in d:
                        test_reward_B.append(float(d['test_reward_B']))
        print('epoch[0]', epoch[0], 'epochs[-1]', epoch[-1])
        if min_y is None:
            min_y = 0
        if max_y is not None:
            plt.ylim([min_y, max_y])
        suffix = ''
        if len(split_logfiles) > 0:
            suffix = ' %s' % (j + 1)

        for name, reward in reward_names.items():
            if reward:
                plt.plot(np.array(epoch) / 1000, reward, label=name + suffix)

    if title is not None:
        plt.title(title)
    plt.xlabel('Episodes of 128 games (thousands)')
    plt.ylabel('Normalized Reward')
    plt.legend()
    plt.show()
    # plt.savefig('/tmp/out-reward.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', type=str)
    parser.add_argument('--title', type=str)
    parser.add_argument('--max-x', type=int)
    parser.add_argument('--min-y', type=float)
    parser.add_argument('--max-y', type=float)

    args = parser.parse_args()
    plot_reward(**vars(args))
