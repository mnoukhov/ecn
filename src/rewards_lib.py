#TODO
# measure selfish reward correctly
import torch

from absl import flags, logging

FLAGS = flags.FLAGS


def calc_rewards(t, s, term, enable_cuda):
    """ calcualate rewards for any games just finished
    it will calculate three reward values:
        - agent 1 (as % of its max),
        - agent 2 (as % of its max),
        - prosocial (as % of max agent 1 + 2)

    in the non-prosocial setting, we need all three:
        - first two for training
        - next one for evaluating Table 1, in the paper
    in the prosocial case, we'll skip calculating the individual agent rewards,
    possibly/probably
    """

    agent = t % 2
    batch_size = term.size()[0]
    type_constr = torch.cuda if enable_cuda else torch
    rewards_batch = type_constr.FloatTensor(batch_size, 3).fill_(0)
    if t == 0:
        # on first timestep theres no actual proposal yet, so score zero if terminate
        return rewards_batch

    reward_eligible_mask = term.view(batch_size).clone().byte()
    if reward_eligible_mask.max() == 0:
        # if none of them accepted proposal, by terminating
        return rewards_batch

    exceeded_pool, _ = ((s.last_proposal - s.pool) > 0).max(1)
    if exceeded_pool.max() > 0:
        reward_eligible_mask[exceeded_pool.nonzero().long().view(-1)] = 0
        if reward_eligible_mask.max() == 0:
            # all eligible ones exceeded pool
            return rewards_batch

    pool = s.pool.float()
    last_proposal = s.last_proposal.float()
    utilities = s.utilities.float()

    proposer = 1 - agent
    accepter = agent
    proposal = type_constr.FloatTensor(batch_size, 2, 3).fill_(0)
    proposal[:, proposer] = last_proposal
    proposal[:, accepter] = pool - last_proposal
    # max of all agents' utility for an item
    max_utility, _ = utilities.max(1)

    reward_eligible_idxes = reward_eligible_mask.nonzero().view(-1)
    raw_rewards = type_constr.FloatTensor(batch_size, 2).fill_(0)
    #TODO change this to be vector operation
    for b in reward_eligible_idxes:
        for i in range(2):
            raw_rewards[b][i] = torch.dot(utilities[b, i], proposal[b, i])

        available_prosocial = torch.dot(max_utility[b], pool[b])
        if available_prosocial == 0:
            logging.error('total available utility 0, utilities {}, pool {}'.format(utilities[b], pool[b]))
        else:
            actual_prosocial = raw_rewards[b].sum()
            rewards_batch[b][2] = actual_prosocial / available_prosocial

        max_agent = torch.matmul(utilities[b], pool[b])
        for i in range(2):
            if max_agent[i] == 0:
                logging.warning('agent {} available utility 0, utility {}, pool {}'.format(i, utilities[b,i], pool[b]))
            elif FLAGS.prosociality > 0:
                alpha = FLAGS.prosociality
                actual =  (1 - alpha) * raw_rewards[b][i] + alpha * raw_rewards[b][1-i]
                alpha_max_utility, _ = (utilities[b].t() * type_constr.FloatTensor([alpha, 1-alpha])).max(1)
                available = torch.matmul(alpha_max_utility, pool[b])
                rewards_batch[b][i] = actual / available
            else:
                rewards_batch[b][i] = raw_rewards[b][i] / max_agent[i]

    return rewards_batch
