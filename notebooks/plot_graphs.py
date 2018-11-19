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
    'utt_unmasked_A': 'train unmasked utt A',
    'utt_unmasked_B': 'train unmasked utt B',
    'test_unmasked_A': 'test unmasked utt A',
    'test_unmasked_B': 'test unmasked utt B',
    'utt_unmasked_A_first': 'train unmasked utt first A',
    'utt_unmasked_B_first': 'train unmasked utt first B',
    'test_unmasked_A_first': 'test unmasked utt first A',
    'test_unmasked_B_first': 'test unmasked utt first B',
    'prop_unmasked_A': 'train prop unmasked A',
    'prop_unmasked_B': 'train prop unmasked B',
    'test_prop_unmasked_A': 'test prop unmasked A',
    'test_prop_unmasked_B': 'test prop unmasked B',
    'prop_unmasked_A_first': 'train prop unmasked first A',
    'prop_unmasked_B_first': 'train prop unmasked first B',
    'test_prop_unmasked_A_first': 'test prop unmasked first A',
    'test_prop_unmasked_B_first': 'test prop unmasked first B',
    'argmaxp_term': 'term argmax prob',
    'argmaxp_utt': 'utt argmax prob',
    'argmaxp_prop': 'prop argmax prob',
}

def plot_one(logfile, **args):
    rewards = parse_logfile(logfile)
    epochs = rewards.pop('epochs')
    epochs = np.array(epochs) / 1000
    plot(epochs, rewards, **args)


def plot(epochs, rewards, title=None, min_y=None, max_y=None,
         show_train=False, show_both=True, show_unmasked=False,
         show_argmax=False, rename={}, plot_args={}, legend_loc='best'):
    if min_y is None:
        min_y = 0
    if max_y is not None:
        plt.ylim([min_y, max_y])

    for name, values in sorted(rewards.items()):
        if not any(values):
            continue
        elif ('train' in name and 'reward' in name) and not show_train:
            continue
        elif ('both' in name) and not show_both:
            continue
        elif ('argmax' in name) and not show_argmax:
            continue
        elif 'unmasked' in name:
            if not show_unmasked:
                continue
            elif ('utt' in name) and not ('utt' in show_unmasked):
                continue
            elif ('prop' in name) and not ('prop' in show_unmasked):
                continue
            elif ('test' in name) and not ('test' in show_unmasked):
                continue
            elif ('train' in name) and not ('train' in show_unmasked):
                continue

            if ('first' in name) and ('first' in show_unmasked):
                if name in rename:
                    name = rename[name]
                extra_args = plot_args.get(name, None)
                if extra_args:
                    plt.plot(epochs, values, **extra_args, label=name)
                else:
                    plt.plot(epochs, values, label=name)
            elif ('avg' in show_unmasked):
                if name in rename:
                    name = rename[name]
                extra_args = plot_args.get(name, None)
                if extra_args:
                    plt.plot(epochs, values, **extra_args, label=name)
                else:
                    plt.plot(epochs, values, label=name)

        else:
            if name in rename:
                name = rename[name]
            extra_args = plot_args.get(name, None)
            if extra_args:
                plt.plot(epochs, values, **extra_args, label=name)
            else:
                plt.plot(epochs, values, label=name)
    if title:
        plt.title(title)

    plt.xlabel('Episodes of 128 games (thousands)')
    if show_unmasked: 
        plt.ylabel('Normalized Reward\nPercent Unmasked')
    else:
        plt.ylabel('Normalized Reward')
    plt.legend(loc=legend_loc)
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

    for logfile in sorted(logfiles):
        plot_one(logfile, **args)


def plot_average(logdir, **args):
    logfiles = glob.glob('{}/*.log'.format(logdir))
    if not logfiles:
        raise Exception('no files in this folder')

    parsed = [parse_logfile(logfile) for logfile in logfiles]

    # cut off at minimum of all logs
    epochs = [log.pop('epochs') for log in parsed]
    length = min(len(epoch) for epoch in epochs)
    epochs = np.array(epochs[0][:length]) / 1000
    avg_rewards = {}

    for reward_name in parsed[0].keys():
        reward_list = [log[reward_name][:length] for log in parsed]
        avg_rewards[reward_name] = np.array(reward_list).mean(axis=0)

    plot(epochs, avg_rewards, **args)


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
                print('failed on line {}'.format(n))
                continue


            for item, value in d.items():
                name = json_to_name.get(item)
                if name:
                    rewards[name].append(float(value))

    print(n)
    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', type=str)
    parser.add_argument('--title', type=str)
    parser.add_argument('--min-y', type=float)
    parser.add_argument('--max-y', type=float)
    parser.add_argument('--show_train', action='store_true')
    parser.add_argument('--show_both', action='store_true')
    parser.add_argument('--show_unmasked', type=str)

    args = parser.parse_args()
    plot_one(**vars(args))
