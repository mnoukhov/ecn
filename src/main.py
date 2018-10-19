import datetime

from absl import app
from absl import flags
from absl.flags import argparse_flags

from ecn import run

FLAGS = flags.FLAGS

# game args
flags.DEFINE_integer('utt_max_length', 6, 'max length of an utterance')
flags.DEFINE_integer('utt_vocab_size', 11, 'size of utterance vocab')
flags.DEFINE_integer('item_max_quantity', 6, 'max + 1 quantity of pool item')
flags.DEFINE_integer('item_max_utility', 6, 'max + 1 utility of pool item')
flags.DEFINE_integer('item_num_types', 3, 'number of pool item types')

# experiments
flags.DEFINE_boolean('utility_normalize', False, 'sum of both agents utilities is 0.5*max_utility*num_types')
flags.DEFINE_boolean('utility_nonzero', False, 'force min utility of 1 for every object')
flags.DEFINE_enum('force_utility_comm', None, ['A', 'B', 'both'], 'force an agent to communicate its utilities')


def parse_flags(argv):
    parser = argparse_flags.ArgumentParser()
    # game args
    # game_args = parser.add_argument_group('game args')
    # game_args.add_argument('--utterance-max-length', type=int, default=6)
    # game_args.add_argument('--utterance-vocab-size', type=int, default=11)
    # game_args.add_argument('--item-max-quantity', type=int, default=6)
    # game_args.add_argument('--item-max-utility', type=int, default=11)
    # model info
    parser.add_argument('--model-file', type=str, default='model_saves/model.dat')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, help='optional')
    parser.add_argument('--test-seed', type=int, default=123, help='used for generating test game set')
    parser.add_argument('--episodes', type=int, default=5e5, help='total number of episodes to run')
    # hyperparams
    parser.add_argument('--term-entropy-reg', type=float, default=0.05)
    parser.add_argument('--utterance-entropy-reg', type=float, default=0.001)
    parser.add_argument('--proposal-entropy-reg', type=float, default=0.05)
    #game
    parser.add_argument('--disable-proposal', action='store_true')
    parser.add_argument('--disable-comms', action='store_true')
    parser.add_argument('--disable-prosocial', action='store_true')
    parser.add_argument('--render-every-seconds', type=int, default=30)
    parser.add_argument('--save-every-seconds', type=int, default=30)
    parser.add_argument('--testing', action='store_true', help='turn off learning; always pick argmax')
    parser.add_argument('--enable-cuda', action='store_true')
    parser.add_argument('--no-load', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--name', type=str, default='', help='used for logfile naming')
    parser.add_argument('--logfile', type=str, default='logs/log_%Y%m%d_%H%M%S{name}.log')
    # experiments
    parser.add_argument('--opponent-utility-comms', type=int, default=None)
    parser.add_argument('--utility-normalization', action='store_true')
    args = parser.parse_args()
    args.enable_comms = not args.disable_comms
    args.enable_proposal = not args.disable_proposal
    args.prosocial = not args.disable_prosocial
    args.logfile = args.logfile.format(**args.__dict__)
    args.logfile = datetime.datetime.strftime(datetime.datetime.now(), args.logfile)
    del args.__dict__['disable_comms']
    del args.__dict__['disable_proposal']
    del args.__dict__['disable_prosocial']
    del args.__dict__['name']

    return args

if __name__ == '__main__':
    app.run(run, flags_parser=parse_flags)
