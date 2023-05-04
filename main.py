import datetime
import os

from absl import app
from absl import flags
from absl.flags import argparse_flags

from src.ecn import run

FLAGS = flags.FLAGS

flags.DEFINE_boolean('enable_cuda', True, 'DEPRECATED executing on gpu')
flags.DEFINE_enum('device', None, ['cuda', 'cpu'], 'device to execute on')

# game args
flags.DEFINE_integer('utt_max_length', 6, 'max length of an utterance')
flags.DEFINE_integer('utt_vocab_size', 11, 'size of utterance vocab')
flags.DEFINE_integer('item_max_quantity', 5, 'max quantity of pool item')
flags.DEFINE_integer('item_max_utility', 10, 'max utility of pool item')
flags.DEFINE_integer('item_num_types', 3, 'number of pool item types')
flags.DEFINE_integer('max_timesteps', 10, 'max number of timesteps')

# # model info
# flags.DEFINE_integer('batch_size', 128, 'number of negotiation games simultaneously')

# agents
flags.DEFINE_boolean('linguistic', True, 'whether agents can communicate along linguistic channel')
flags.DEFINE_boolean('proposal', True, 'whether agents can send propoposal along proposal channel')
flags.DEFINE_boolean('prosocial', True, 'whether agents share their rewards')

# experiments
flags.DEFINE_float('bias', None, 'level of agent competitiveness')
flags.DEFINE_boolean('randomize_first', False, 'whether agent 1 always goes first')
flags.DEFINE_boolean('utility_normalize', False, 'sum of an agent utilities is ~max_utility')
flags.DEFINE_boolean('utility_nonzero', False, 'force min utility of 1 for every object')
flags.DEFINE_enum('force_utility_comm', None, ['A', 'B', 'both'], 'force an agent to communicate its utilities')
flags.DEFINE_float('prosociality', 0, 'alpha of prosociality for selfish agents')
flags.DEFINE_boolean('force_masking_comm', False, 'force agents to communicate by masking their utilities')
flags.DEFINE_boolean('random_start', False, 'randomly choose which agent starts the negotiation')
flags.DEFINE_boolean('proposal_termination', False, 'term policy does nothing, to terminate agent must propose the same as the previous proposal')
flags.DEFINE_boolean('zero_sum_reward', False, 'R_A = r_A - r_B')
flags.DEFINE_boolean('normal_form', False, 'both agents act simultaneously')

def parse_flags(argv):
    parser = argparse_flags.ArgumentParser()
    # game args
    # game_args = parser.add_argument_group('game args')
    # game_args.add_argument('--utterance-max-length', type=int, default=6)
    # game_args.add_argument('--utterance-vocab-size', type=int, default=12)
    # game_args.add_argument('--item-max-quantity', type=int, default=6)
    # game_args.add_argument('--item-max-utility', type=int, default=11)
    # model info
    parser.add_argument('--seed', type=int, help='optional')
    parser.add_argument('--test-seed', type=int, default=123, help='used for generating test game set')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--episodes', type=int, default=5e5, help='total number of episodes to run')
    # hyperparams
    parser.add_argument('--term-entropy-reg', type=float, default=0.05)
    parser.add_argument('--utterance-entropy-reg', type=float, default=0.001)
    parser.add_argument('--proposal-entropy-reg', type=float, default=0.05)
    #game
    parser.add_argument('--render-every-episode', type=int, default=200)
    parser.add_argument('--save-every-episode', type=int, default=500)
    parser.add_argument('--testing', action='store_true', help='turn off learning; always pick argmax')
    parser.add_argument('--no-load', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--name',  help='used for logfile, model naming, and wandb')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--model_dir', type=str, default='model_saves')
    parser.add_argument('--savedir', type=str, default='.')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false')
    parser.add_argument('--wandb-offline', action='store_true')
    parser.add_argument('--wandb-group', type=str, help='used for grouping runs')
    args = parser.parse_args()

    args.logdir = f'{args.savedir}/{args.logdir}'
    args.model_dir = f'{args.savedir}/{args.model_dir}'

    job_id = os.environ.get('SLURM_ARRAY_JOB_ID', '')
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '')
    if task_id:
        slurm_id = '{}_{}'.format(job_id, task_id)
    else:
        slurm_id = os.environ.get('SLURM_JOB_ID', '')

    args.log_file = '{}/{}_{}_%Y%m%d.log'.format(args.logdir, args.name, slurm_id)
    args.log_file = datetime.datetime.strftime(datetime.datetime.now(), args.log_file)
    args.model_file = '{}/{}_{}_%Y%m%d.dat'.format(args.model_dir, args.name, slurm_id)
    args.model_file = datetime.datetime.strftime(datetime.datetime.now(), args.model_file)

    if ((FLAGS.force_masking_comm or FLAGS.force_utility_comm)
        and not FLAGS.linguistic):
        raise Exception('linguistic channel must be on to use making or utility comm')

    return args

if __name__ == '__main__':
    app.run(run, flags_parser=parse_flags)
