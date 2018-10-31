# TODO
# test/save per episode not
# change everything from long to float

import argparse
import datetime
import json
import os
import time
from os import path

import numpy as np
import torch
from absl import flags
from torch import optim
from torch.autograd import Variable
from pprint import pprint

from src.alive_sieve import AliveSieve, SievePlayback
from src.nets import AgentModel
from src.rewards_lib import calc_rewards
from src.sampling import (generate_test_batches,
                          generate_training_batch,
                          hash_batches)

FLAGS = flags.FLAGS

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
        print(' %s/%s %s/%s %s/%s' % (
            prop[0][0].item(), s.pool[0][0].item(),
            prop[0][1].item(), s.pool[0][1].item(),
            prop[0][2].item(), s.pool[0][2].item(),
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
        self.m_prev = torch.zeros(batch_size, FLAGS.utt_max_length).long()

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
    agent_models,
    batch_size,
    testing,
    render=False):
    """
    turning testing on means, we disable stochasticity: always pick the argmax
    """

    type_constr = torch.cuda if enable_cuda else torch
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
        print('           ',
              '{}   {}   {}'.format(*s.utilities[0][0].tolist()),
              '      ',
              '{} {} {}'.format(*s.pool[0].tolist()),
              '          ',
              '{}   {}   {}'.format(*s.utilities[0][1].tolist()))
    for t in range(FLAGS.max_timesteps):
        if FLAGS.linguistic:
            _prev_message = s.m_prev
        else:
            _prev_message = type_constr.LongTensor(sieve.batch_size, 6).fill_(0)

        if FLAGS.proposal:
            _prev_proposal = s.last_proposal
        else:
            _prev_proposal = type_constr.LongTensor(sieve.batch_size, 3).fill_(0)

        agent = t % 2
        agent_model = agent_models[agent]
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
            enable_cuda=enable_cuda
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
        print(' rewards: {:2.2f} {:2.2f} {:2.2f}'.format(*rewards[0].tolist()))
        print('  ')

    return actions_by_timestep, rewards, num_steps, alive_masks, entropy_loss_by_agent, \
        term_matches_argmax_count, num_policy_runs, utt_matches_argmax_count, utt_stochastic_draws, \
        prop_matches_argmax_count, prop_stochastic_draws


def safe_div(a, b):
    """
    returns a / b, unless b is zero, in which case returns 0
    this is primarily for usage in cases where b might be systemtically zero, eg because comms are disabled or similar
    also accounts for a or b being tensors
    """
    if isinstance(a, torch.Tensor):
        a = a.item()
    if isinstance(b, torch.Tensor):
        b = b.item()
    return 0 if b == 0 else a / b


def run(args):
    """
    testing option will:
    - use argmax, ie disable stochastic draws
    - not run optimizers
    - not save model
    """
    flags_dict = {flag.name: flag.value for flag in FLAGS.flags_by_module_dict()['src/main.py']}
    args_dict = args.__dict__
    pprint(args_dict)
    pprint(flags_dict)

    type_constr = torch.cuda if FLAGS.enable_cuda else torch
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_r = np.random.RandomState(args.seed)
    else:
        train_r = np.random

    test_r = np.random.RandomState(args.test_seed)
    test_batches = generate_test_batches(batch_size=args.batch_size,
                                         num_batches=5,
                                         random_state=test_r)
    test_hashes = hash_batches(test_batches)

    episode = 0
    start_time = time.time()
    agent_models = []
    agent_opts = []
    agent_name = ['A', 'B']
    for i in range(2):
        model = AgentModel(
            name=agent_name[i],
            term_entropy_reg=args.term_entropy_reg,
            utterance_entropy_reg=args.utterance_entropy_reg,
            proposal_entropy_reg=args.proposal_entropy_reg
        )
        if FLAGS.enable_cuda:
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
    f_log = open(args.log_file, 'w')
    all_args = {**args_dict, **flags_dict}
    f_log.write('meta: %s\n' % json.dumps(all_args))
    last_save = time.time()
    baseline = type_constr.FloatTensor(3).fill_(0)
    term_matches_argmax_count = 0
    num_policy_runs = 0
    utt_matches_argmax_count = 0
    utt_stochastic_draws = 0
    prop_matches_argmax_count = 0
    prop_stochastic_draws = 0
    while episode < args.episodes:
        render = (episode % args.render_every_episode == 0)
        batch = generate_training_batch(batch_size=args.batch_size,
                                        test_hashes=test_hashes,
                                        random_state=train_r)
        (actions, rewards, steps, alive_masks, entropy_loss_by_agent,
         _term_matches_argmax_count, _num_policy_runs, _utt_matches_argmax_count, _utt_stochastic_draws,
         _prop_matches_argmax_count, _prop_stochastic_draws) = run_episode(
             batch=batch,
             enable_cuda=FLAGS.enable_cuda,
             agent_models=agent_models,
             batch_size=args.batch_size,
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
                if FLAGS.prosocial:
                    rewards_by_agent.append(baselined_rewards[:, 2])
                else:
                    rewards_by_agent.append(baselined_rewards[:, i])
            sieve_playback = SievePlayback(alive_masks, enable_cuda=FLAGS.enable_cuda)
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

        rewards_sum += rewards.detach().sum(0)
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
                     enable_cuda=FLAGS.enable_cuda,
                     agent_models=agent_models,
                     batch_size=args.batch_size,
                     render=True,
                     testing=True)
                test_rewards_sum += test_rewards.sum(0)

            time_since_last = time.time() - last_print
            rewards_str = '%.2f,%.2f,%.2f' % (rewards_sum[0] / count_sum,
                                              rewards_sum[1] / count_sum,
                                              rewards_sum[2] / count_sum)
            test_rewards_str = '%.2f,%.2f,%.2f' % (test_rewards_sum[0] / test_count_sum,
                                                   test_rewards_sum[1] / test_count_sum,
                                                   test_rewards_sum[2] / test_count_sum)
            baseline_str = '%.2f,%.2f,%.2f' % (baseline[0], baseline[1], baseline[2])
            print('test  {}'.format(test_rewards_str))
            print('train {}'.format(rewards_str))
            print('base  {}'.format(baseline_str))
            print('ep {}, {} games/sec, {:4.4f} avg steps, argmaxp term={:4.4f} utt={:4.4f} prop={:4.4f}'.format(
                episode,
                int(count_sum / time_since_last),
                steps_sum.item() / count_sum,
                term_matches_argmax_count / num_policy_runs,
                safe_div(utt_matches_argmax_count, utt_stochastic_draws),
                prop_matches_argmax_count / prop_stochastic_draws
            ))
            f_log.write(json.dumps({
                'episode': episode,
                'avg_reward_A': (rewards_sum[0] / count_sum).item(),
                'avg_reward_B': (rewards_sum[1] / count_sum).item(),
                'avg_reward_0': (rewards_sum[2] / count_sum).item(),
                'test_reward_A': (test_rewards_sum[0] / test_count_sum).item(),
                'test_reward_B': (test_rewards_sum[1] / test_count_sum).item(),
                'test_reward': (test_rewards_sum[2] / test_count_sum).item(),
                'avg_steps': (steps_sum / count_sum).item(),
                'games_sec': (count_sum / time_since_last),
                'elapsed': time.time() - start_time,
                'argmaxp_term': term_matches_argmax_count / num_policy_runs,
                'argmaxp_utt': safe_div(utt_matches_argmax_count, utt_stochastic_draws),
                'argmaxp_prop': prop_matches_argmax_count / prop_stochastic_draws,
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

        if (not args.testing
            and not args.no_save
            and episode > 0
            and episode % args.save_every_episode == 0):
            save_model(model_file=args.model_file,
                       agent_models=agent_models,
                       agent_opts=agent_opts,
                       start_time=start_time,
                       episode=episode)
            print('saved model')

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
