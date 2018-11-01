"""
Given a logfile, plot a graph
"""
import matplotlib.pyplot as plt
import json
import argparse
import numpy as np
import glob


json_to_name = {
    'episode': 'epochs',
    'avg_reward_0': 'train reward both',
    'avg_reward_A': 'train reward A',
    'avg_reward_B': 'train reward B',
    'test_reward': 'test reward both',
    'test_reward_A': 'test reward A',
    'test_reward_B': 'test reward B',
    'utt_unmasked_A': 'unmasked A',
    'utt_unmasked_B': 'unmasked B',
}

def plot_one(logfile, title=None, min_y=None, max_y=None,
             show_train=False, show_both=True, show_unmasked=False):
    rewards = parse_logfile(logfile)
    epochs = rewards.pop('epochs')
    epochs = np.array(epochs) / 1000

    if min_y is None:
        min_y = 0
    if max_y is not None:
        plt.ylim([min_y, max_y])

    for name, values in rewards.items():
        if not values:
            continue
        elif not show_train and ('train' in name):
            continue
        elif not show_both and ('both' in name):
            continue
        elif not show_unmasked and ('unmasked' in name):
            continue
        else:
            plt.plot(epochs, values, label=name)

    if title:
        plt.title(title)

    plt.xlabel('Episodes of 128 games (thousands)')
    plt.ylabel('Normalized Reward')
    plt.legend()
    plt.show()
    # plt.savefig('/tmp/out-reward.png')


def plot_folder(logdir, **args):
    fileglob = '{}/*.log'.format(logdir)
    plot_glob(fileglob, **args)


def plot_glob(fileglob, **args):
    print('glob {}'.format(fileglob))
    logfiles = glob.glob(fileglob)
    if not logfiles:
        raise Exception('no file matches glob')

    for logfile in logfiles:
        plot_one(logfile, **args)


def plot_average(logdir, title=None, min_y=None, max_y=None,
                 show_train=False, show_both=True):
    logfiles = glob.glob('{}/*.log'.format(logdir))
    if not logfiles:
        raise Exception('no files in this folder')

    parsed = [parse_logfile(logfile) for logfile in logfiles]

    # cut off at minimum of all logs
    epochs = [log.pop('epochs') for log in parsed]
    length = min(len(epoch) for epoch in epochs)
    epochs = np.array(epochs[0][:length]) / 1000

    for name in parsed[0].keys():
        if not show_train and ('train' in name):
            continue
        elif not show_both and ('both' in name):
            continue

        values_arr = np.array([log[name][:length] for log in parsed])
        avg_values = values_arr.mean(axis=0)

        if not avg_values.any():
            continue

        plt.plot(epochs, avg_values, label=name)

    if title:
        plt.title(title)

    if min_y is None:
        min_y = 0
    if max_y is not None:
        plt.ylim([min_y, max_y])

    plt.xlabel('Episodes of 128 games (thousands)')
    plt.ylabel('Normalized Reward')
    plt.legend()
    plt.show()
    # plt.savefig('/tmp/out-reward.png')


def parse_logfile(logfile):
    rewards = {name: [] for name in json_to_name.values()}

    with open(logfile, 'r') as f:
        for n, line in enumerate(f):
            if n == 0:
                print(logfile, line)
                continue  # skip first line
            line = line.strip()
            if line == '':
                continue

            try:
                d = json.loads(line)
            except:
                continue

            # if not all(reward_name in d for reward_name in json_to_name):
                # continue

            for item, value in d.items():
                name = json_to_name.get(item)
                if name:
                    rewards[name].append(float(value))

    print(n)
    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str)
    parser.add_argument('--title', type=str)
    parser.add_argument('--min-y', type=float)
    parser.add_argument('--max-y', type=float)
    parser.add_argument('--show_train', action='store_true')
    parser.add_argument('--show_both', action='store_true')

    args = parser.parse_args()
    plot_folder(**vars(args))
