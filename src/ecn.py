#TODO
# test/save per episode not
# centralize args
# change everything from long to float

import argparse
import datetime
import json
import os
import time
from os import path

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from src.alive_sieve import AliveSieve, SievePlayback
from src.nets import AgentModel
from src.rewards_lib import calc_rewards
from src.sampling import (generate_test_batches,
                          generate_training_batch,
                          hash_batches)
from src.args import UTT_MAX_LEN

def render_action(t, s, prop, term):
    agent = t % 2
    speaker = 'A' if agent == 0 else 'B'
    utility = s.utilities[:, agent]
    print('  ', end='')
    if speaker == 'B':
        print('                                   ', end='')
    if term[0][0]:
        print(' ACC')
    else:
        print(' ' + ''.join([str(v) for v in s.m_prev[0].view(-1).tolist()]), end='')
        print(' %s:%s/%s %s:%s/%s %s:%s/%s' % (
            utility[0][0], prop[0][0], s.pool[0][0],
            utility[0][1], prop[0][1], s.pool[0][1],
            utility[0][2], prop[0][2], s.pool[0][2],
        ), end='')
        print('')
        if t + 1 == s.N[0]:
            print('  [out of time]')


def save_model(model_file, agent_models, agent_opts, start_time, episode):
    state = {}
    for i in range(2):
        state['agent%s' % i] = {}
        state['agent%s' % i]['model_state'] = agent_models[i].state_dict()
        state['agent%s' % i]['opt_state'] = agent_opts[i].state_dict()
    state['episode'] = episode
    state['elapsed_time'] = time.time() - start_time
    with open(model_file + '.tmp', 'wb') as f:
        torch.save(state, f)
    os.rename(model_file + '.tmp', model_file)


def load_model(model_file, agent_models, agent_opts):
    with open(model_file, 'rb') as f:
        state = torch.load(f)
    for i in range(2):
        agent_models[i].load_state_dict(state['agent%s' % i]['model_state'])
        agent_opts[i].load_state_dict(state['agent%s' % i]['opt_state'])
    episode = state['episode']
    # create a kind of 'virtual' start_time
    start_time = time.time() - state['elapsed_time']
    return episode, start_time


class State(object):
    def __init__(self, N, pool, utilities):
        batch_size = N.size()[0]
        self.N = N
        self.pool = pool
        self.utilities = torch.zeros(batch_size, 2, 3).long()
        self.utilities[:, 0] = utilities[0]
        self.utilities[:, 1] = utilities[1]

        self.last_proposal = torch.zeros(batch_size, 3).long()
        self.m_prev = torch.zeros(batch_size, UTT_MAX_LEN).long()

    def cuda(self):
        self.N = self.N.cuda()
        self.pool = self.pool.cuda()
        self.utilities = self.utilities.cuda()
        self.last_proposal = self.last_proposal.cuda()
        self.m_prev = self.m_prev.cuda()

    def sieve_(self, still_alive_idxes):
        self.N = self.N[still_alive_idxes]
        self.pool = self.pool[still_alive_idxes]
        self.utilities = self.utilities[still_alive_idxes]
        self.last_proposal = self.last_proposal[still_alive_idxes]
        self.m_prev = self.m_prev[still_alive_idxes]


def run_episode(
    batch,
    enable_cuda,
    enable_comms,
    enable_proposal,
    comms_opponent_utility,
    prosocial,
    agent_models,
    # batch_size,
    testing,
    render=False):
    """
    turning testing on means, we disable stochasticity: always pick the argmax
    """

    type_constr = torch.cuda if enable_cuda else torch
    batch_size = batch['N'].size()[0]
    s = State(**batch)
    if enable_cuda:
        s.cuda()

    sieve = AliveSieve(batch_size=batch_size, enable_cuda=enable_cuda)
    actions_by_timestep = []
    alive_masks = []

    # next two tensors wont be sieved, they will stay same size throughout
    # entire batch, we will update them using sieve.out_idxes[...]
    rewards = type_constr.FloatTensor(batch_size, 3).fill_(0)
    num_steps = type_constr.LongTensor(batch_size).fill_(10)
    term_matches_argmax_count = 0
    utt_matches_argmax_count = 0
    utt_stochastic_draws = 0
    num_policy_runs = 0
    prop_matches_argmax_count = 0
    prop_stochastic_draws = 0

    entropy_loss_by_agent = [
        Variable(type_constr.FloatTensor(1).fill_(0)),
        Variable(type_constr.FloatTensor(1).fill_(0))
    ]
    if render:
        print('  ')
    for t in range(10):
        agent = t % 2

        agent_model = agent_models[agent]
        if enable_comms:
            if comms_opponent_utility is None:
                _prev_message = s.m_prev
            elif ((comms_opponent_utility == 0 and agent == 0) or
                  (comms_opponent_utility == 1 and agent == 1) or
                  (comms_opponent_utility == 2)):
                # 0: agent 0 sees agent 1 utility
                # 1: agent 1 sees agent 0 utility
                # 2: both agents see each other's utility
                _prev_message = s.utilities[:, 1 - agent]
        else:
            # we dont strictly need to blank them, since they'll be all zeros anyway,
            # but defense in depth and all that :)
            _prev_message = type_constr.LongTensor(sieve.batch_size, 6).fill_(0)

        if enable_proposal:
            _prev_proposal = s.last_proposal
        else:
            # we do need to blank this one though :)
            _prev_proposal = type_constr.LongTensor(sieve.batch_size, 3).fill_(0)

        (nodes, term_a, s.m_prev, this_proposal, _entropy_loss,
         _term_matches_argmax_count, _utt_matches_argmax_count, _utt_stochastic_draws,
         _prop_matches_argmax_count, _prop_stochastic_draws) = agent_model(
             pool=Variable(s.pool),
             utility=Variable(s.utilities[:, agent]),
             m_prev=Variable(_prev_message),
             prev_proposal=Variable(_prev_proposal),
             testing=testing,
         )
        entropy_loss_by_agent[agent] += _entropy_loss
        actions_by_timestep.append(nodes)
        term_matches_argmax_count += _term_matches_argmax_count
        num_policy_runs += sieve.batch_size
        utt_matches_argmax_count += _utt_matches_argmax_count
        utt_stochastic_draws += _utt_stochastic_draws
        prop_matches_argmax_count += _prop_matches_argmax_count
        prop_stochastic_draws += _prop_stochastic_draws

        if render and sieve.out_idxes[0] == 0:
            render_action(
                t=t,
                s=s,
                term=term_a,
                prop=this_proposal
            )

        new_rewards = calc_rewards(
            t=t,
            s=s,
            term=term_a,
            enable_cuda=args.enable_cuda
        )
        rewards[sieve.out_idxes] = new_rewards
        s.last_proposal = this_proposal

        sieve.mark_dead(term_a)
        sieve.mark_dead(t + 1 >= s.N)
        alive_masks.append(sieve.alive_mask.clone())
        sieve.set_dead_global(num_steps, t + 1)
        if sieve.all_dead():
            break

        s.sieve_(sieve.alive_idxes)
        sieve.self_sieve_()

    if render:
        print('  r: %.2f' % rewards[0].mean())
        print('  ')

    return actions_by_timestep, rewards, num_steps, alive_masks, entropy_loss_by_agent, \
        term_matches_argmax_count, num_policy_runs, utt_matches_argmax_count, utt_stochastic_draws, \
        prop_matches_argmax_count, prop_stochastic_draws


def safe_div(a, b):
    """
    returns a / b, unless b is zero, in which case returns 0

    this is primarily for usage in cases where b might be systemtically zero, eg because comms are disabled or similar
    """
    return 0 if b == 0 else a / b


def run(args):
    """
    testing option will:
    - use argmax, ie disable stochastic draws
    - not run optimizers
    - not save model
    """
    type_constr = torch.cuda if args.enable_cuda else torch
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_r = np.random.RandomState(args.seed)
    else:
        train_r = np.random

    test_r = np.random.RandomState(args.test_seed)
    test_batches = generate_test_batches(batch_size=args.batch_size, num_batches=5, random_state=test_r)
    test_hashes = hash_batches(test_batches)

    episode = 0
    start_time = time.time()
    agent_models = []
    agent_opts = []
    for i in range(2):
        model = AgentModel(
            enable_comms=args.enable_comms,
            enable_proposal=args.enable_proposal,
            term_entropy_reg=args.term_entropy_reg,
            utterance_entropy_reg=args.utterance_entropy_reg,
            proposal_entropy_reg=args.proposal_entropy_reg
        )
        if args.enable_cuda:
            model = model.cuda()
        agent_models.append(model)
        agent_opts.append(optim.Adam(params=agent_models[i].parameters()))
    if path.isfile(args.model_file) and not args.no_load:
        episode, start_time = load_model(
            model_file=args.model_file,
            agent_models=agent_models,
            agent_opts=agent_opts)
        print('loaded model')
    elif args.testing:
        print('')
        print('ERROR: must have loadable model to use --testing option')
        print('')
        return
    last_print = time.time()
    rewards_sum = type_constr.FloatTensor(3).fill_(0)
    steps_sum = 0
    count_sum = 0
    for d in ['logs', 'model_saves']:
        if not path.isdir(d):
            os.makedirs(d)
    f_log = open(args.logfile, 'w')
    f_log.write('meta: %s\n' % json.dumps(args.__dict__))
    last_save = time.time()
    baseline = type_constr.FloatTensor(3).fill_(0)
    term_matches_argmax_count = 0
    num_policy_runs = 0
    utt_matches_argmax_count = 0
    utt_stochastic_draws = 0
    prop_matches_argmax_count = 0
    prop_stochastic_draws = 0
    while episode < args.episodes:
        render = time.time() - last_print >= args.render_every_seconds
        # render = True
        batch = generate_training_batch(batch_size=args.batch_size, test_hashes=test_hashes, random_state=train_r)
        (actions, rewards, steps, alive_masks, entropy_loss_by_agent,
         _term_matches_argmax_count, _num_policy_runs, _utt_matches_argmax_count, _utt_stochastic_draws,
         _prop_matches_argmax_count, _prop_stochastic_draws) = run_episode(
             batch=batch,
             enable_cuda=args.enable_cuda,
             enable_comms=args.enable_comms,
             enable_proposal=args.enable_proposal,
             comms_opponent_utility=args.comms_opponent_utility,
             agent_models=agent_models,
             prosocial=args.prosocial,
             # batch_size=batch_size,
             render=render,
             testing=args.testing)
        term_matches_argmax_count += _term_matches_argmax_count
        utt_matches_argmax_count += _utt_matches_argmax_count
        utt_stochastic_draws += _utt_stochastic_draws
        num_policy_runs += _num_policy_runs
        prop_matches_argmax_count += _prop_matches_argmax_count
        prop_stochastic_draws += _prop_stochastic_draws

        if not args.testing:
            for i in range(2):
                agent_opts[i].zero_grad()
            reward_loss_by_agent = [0, 0]
            baselined_rewards = rewards - baseline
            rewards_by_agent = []
            for i in range(2):
                if args.prosocial:
                    rewards_by_agent.append(baselined_rewards[:, 2])
                else:
                    rewards_by_agent.append(baselined_rewards[:, i])
            sieve_playback = SievePlayback(alive_masks, enable_cuda=args.enable_cuda)
            for t, global_idxes in sieve_playback:
                agent = t % 2
                if len(actions[t]) > 0:
                    for action in actions[t]:
                        _rewards = rewards_by_agent[agent]
                        _reward = _rewards[global_idxes].float().contiguous().view(
                            sieve_playback.batch_size, 1)
                        #TODO find overflow
                        _reward_loss = - (action * Variable(_reward))
                        _reward_loss = _reward_loss.sum()
                        reward_loss_by_agent[agent] += _reward_loss
            for i in range(2):
                loss = entropy_loss_by_agent[i] + reward_loss_by_agent[i]
                loss.backward()
                agent_opts[i].step()

        rewards_sum += rewards.sum(0)
        steps_sum += steps.sum()
        baseline = 0.7 * baseline + 0.3 * rewards.mean(0)
        count_sum += args.batch_size

        if render:
            """
            run the test batches, print the results
            """
            test_rewards_sum = np.zeros(3)
            test_count_sum = len(test_batches) * args.batch_size
            for test_batch in test_batches:
                (actions, test_rewards, steps, alive_masks, entropy_loss_by_agent,
                 _term_matches_argmax_count, _num_policy_runs, _utt_matches_argmax_count, _utt_stochastic_draws,
                 _prop_matches_argmax_count, _prop_stochastic_draws) = run_episode(
                     batch=test_batch,
                     enable_cuda=args.enable_cuda,
                     enable_comms=args.enable_comms,
                     enable_proposal=args.enable_proposal,
                     comms_opponent_utility=args.comms_opponent_utility,
                     agent_models=agent_models,
                     prosocial=args.prosocial,
                     render=True,
                     testing=True)
                test_rewards_sum += test_rewards.sum(0)
            print('test reward=%.3f' % (test_rewards_sum[2] / test_count_sum))

            time_since_last = time.time() - last_print
            if args.prosocial:
                baseline_str = '%.2f' % baseline[2]
                # rewards_str = '%.2f' % (rewards_sum[2] / count_sum)
            else:
                baseline_str = '%.2f,%.2f' % (baseline[0], baseline[1])
            rewards_str = '%.2f,%.2f,%.2f' % (rewards_sum[0] / count_sum, rewards_sum[1] / count_sum, rewards_sum[2] / count_sum)
            print('e=%s train=%s b=%s games/sec %s avg steps %.4f argmaxp term=%.4f utt=%.4f prop=%.4f' % (
                episode,
                rewards_str,
                baseline_str,
                int(count_sum / time_since_last),
                steps_sum / count_sum,
                term_matches_argmax_count / num_policy_runs,
                safe_div(utt_matches_argmax_count, utt_stochastic_draws),
                prop_matches_argmax_count / prop_stochastic_draws
            ))
            f_log.write(json.dumps({
                'episode': episode,
                'avg_reward_A': rewards_sum[0] / count_sum,
                'avg_reward_B': rewards_sum[1] / count_sum,
                'avg_reward_0': rewards_sum[2] / count_sum,
                'test_reward_A': test_rewards_sum[0] / test_count_sum,
                'test_reward_B': test_rewards_sum[1] / test_count_sum,
                'test_reward': test_rewards_sum[2] / test_count_sum,
                'avg_steps': steps_sum / count_sum,
                'games_sec': count_sum / time_since_last,
                'elapsed': time.time() - start_time,
                'argmaxp_term': (term_matches_argmax_count / num_policy_runs),
                'argmaxp_utt': safe_div(utt_matches_argmax_count, utt_stochastic_draws),
                'argmaxp_prop': (prop_matches_argmax_count / prop_stochastic_draws)
            }) + '\n')
            f_log.flush()
            last_print = time.time()
            steps_sum = 0
            rewards_sum.fill_(0)
            term_matches_argmax_count = 0
            num_policy_runs = 0
            utt_matches_argmax_count = 0
            utt_stochastic_draws = 0
            prop_matches_argmax_count = 0
            prop_stochastic_draws = 0
            count_sum = 0
        if (not args.testing and
            not args.no_save and
            time.time() - last_save >= args.save_every_seconds):
            save_model(
                model_file=args.model_file,
                agent_models=agent_models,
                agent_opts=agent_opts,
                start_time=start_time,
                episode=episode)
            print('saved model')
            last_save = time.time()

        episode += 1

    if (not args.no_save and
        not args.testing):
        save_model(
            model_file=args.model_file,
            agent_models=agent_models,
            agent_opts=agent_opts,
            start_time=start_time,
            episode=episode)
        print('saved model')
    f_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # game fixed args
    game_args = parser.add_argument_group('game args')
    game_args.add_argument('--utterance-max-length', type=int, default=6)
    game_args.add_argument('--utterance-vocab-size', type=int, default=11)
    game_args.add_argument('--item-max-quantity', type=int, default=6)
    game_args.add_argument('--item-max-utility', type=int, default=11)
    game_args.add_argument('--num-items', type=int, default=11)

    # model info
    parser.add_argument('--model-file', type=str, default='model_saves/model.dat')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, help='optional')
    parser.add_argument('--test-seed', type=int, default=123, help='used for generating test game set')
    parser.add_argument('--episodes', type=int, default=5e5, help='total number of episodes to run')
    parser.add_argument('--enable-cuda', action='store_true')

    # hyperparams
    parser.add_argument('--term-entropy-reg', type=float, default=0.05)
    parser.add_argument('--utterance-entropy-reg', type=float, default=0.001)
    parser.add_argument('--proposal-entropy-reg', type=float, default=0.05)

    # agents
    social = parser.add_mutually_exclusive_group(required=True)
    social.add_argument('--prosocial', dest='prosocial', action='store_true')
    social.add_argument('--selfish', dest='prosocial', action='store_false')
    proposal = parser.add_mutually_exclusive_group(required=True)
    proposal.add_argument('--enable-proposal', dest='enable_proposal', action='store_true')
    proposal.add_argument('--disable-proposal', dest='enable_proposal', action='store_false')
    comms = parser.add_mutually_exclusive_group(required=True)
    comms.add_argument('--enable-comms', dest='enable_comms', action='store_true')
    comms.add_argument('--disable-comms', dest='enable_comms', action='store_false')

    # loading / logging
    parser.add_argument('--render-every-seconds', type=int, default=30)
    parser.add_argument('--save-every-seconds', type=int, default=30)
    parser.add_argument('--testing', action='store_true', help='turn off learning; always pick argmax')
    parser.add_argument('--no-load', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--name', type=str, default='', help='used for logfile naming')
    parser.add_argument('--logfile', type=str, default='logs/log_%Y%m%d_%H%M%S{name}.log')

    # experiments
    parser.add_argument('--comms-opponent-utility', type=int, default=None)
    parser.add_argument('--utility-normalization', action='store_true')

    args = parser.parse_args()
    args.logfile = args.logfile.format(**args.__dict__)
    args.logfile = datetime.datetime.strftime(datetime.datetime.now(), args.logfile)
    del args.__dict__['name']
    run(args)
