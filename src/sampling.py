#TODO
# create a dataset of utilities + pools instead of generating on the fly?
# refactor by changing this to classes

import torch
import numpy as np
from absl import flags

FLAGS = flags.FLAGS


def sample_items(batch_size, max_quantity=FLAGS.item_max_quantity, num_items=FLAGS.item_num_types,
                 random_state=np.random):
    """
    max_quantity 6 will give possible values: 0,1,2,3,4,5
    """
    pool = np.zeros((batch_size, num_items), dtype=np.int64)
    zero_pool = [0]*num_items
    zero_idxs = np.arange(batch_size)
    num_zeros = batch_size
    #find batches with all 0s and regenerate
    while num_zeros > 0:
        pool[zero_idxs] = random_state.choice(max_quantity, (num_zeros, num_items), replace=True)
        zero_idxs = (pool == zero_pool)[:,0]
        num_zeros = np.count_nonzero(zero_idxs)

    return torch.from_numpy(pool)


def sample_utility(batch_size,
                   max_utility=FLAGS.item_max_utility,
                   num_items=FLAGS.item_num_types,
                   normalize=FLAGS.utility_normalize,
                   nonzero=FLAGS.utility_nonzero,
                   random_state=np.random):
    util = np.zeros((batch_size, num_items), dtype=np.int64)
    min_utility = 1 if nonzero else 0
    utility_range = np.arange(min_utility, max_utility)

    if not normalize:
        zero_util = [0]*num_items
        zero_idxs = np.arange(batch_size)
        num_zeros = batch_size
        #find batches with all 0s and regenerate
        while num_zeros > 0:
            util[zero_idxs] = random_state.choice(utility_range, (num_zeros, num_items), replace=True)
            zero_idxs = (util == zero_util)[:,0]
            num_zeros = np.count_nonzero(zero_idxs)
    else:
        norm_sum = int(0.5 * max_quantity * num_items)
        util_range = np.arange(1, max_quantity)
        util = random_state.choice(utility_range, (batch_size, num_items), replace=True)
        util = util * norm_sum // np.sum(util, axis=1)

    return torch.from_numpy(util)


def sample_N(batch_size, random_state=np.random):
    N = random_state.poisson(7, batch_size)
    N = np.clip(N, 4, 10)
    N = torch.from_numpy(N)
    return N


def generate_batch(batch_size, random_state=np.random):
    pool = sample_items(batch_size=batch_size,
                        random_state=random_state)
    utilities = []
    utilities.append(sample_utility(batch_size=batch_size,
                                    random_state=random_state))
    utilities.append(sample_utility(batch_size=batch_size,
                                    random_state=random_state))
    N = sample_N(batch_size=batch_size, random_state=random_state)
    return {
        'pool': pool,
        'utilities': utilities,
        'N': N
    }


def generate_test_batches(batch_size, num_batches, random_state):
    """
    so, we need:
    - pools
    - utilities (one set per agent)
    - N
    """
    # r = np.random.RandomState(seed)
    test_batches = []
    for i in range(num_batches):
        batch = generate_batch(batch_size=batch_size,
                               random_state=random_state)
        test_batches.append(batch)
    return test_batches


def hash_long_batch(int_batch, max_quantity):
    num_items = int_batch.size()[1]
    multiplier = torch.LongTensor(num_items)
    v = 1
    for i in range(num_items):
        multiplier[-i - 1] = v
        v *= max_quantity
    hashed_batch = (int_batch * multiplier).sum(1)
    return hashed_batch


def hash_batch(pool, utilities, N):
    v = N
    # use max_quantity=10, so human-readable
    v = v * 1000 + hash_long_batch(pool, max_quantity=10)
    v = v * 1000 + hash_long_batch(utilities[0], max_quantity=10)
    v = v * 1000 + hash_long_batch(utilities[1], max_quantity=10)
    return v


def hash_batches(test_batches):
    """
    we can store each game as a hash like:
    [N - 1]pppuuuuuu
    (where: [N - 1] is {4-10} - 1), ppp is the pool, like 442; and uuuuuu are the six utilities, like 354321
    so, this integer has 10 digits, which I guess we can just store as a normal python integer?
    """
    hashes = set()
    for batch in test_batches:
        hashed = hash_batch(**batch)
        hashes |= set(hashed.tolist())
        # for v in hashed:
        #     hashes.add(v)
    return hashes


def overlaps(test_hashes, batch):
    target_hashes = set(hash_batch(**batch).tolist())
    return bool(test_hashes & target_hashes)


def generate_training_batch(batch_size, test_hashes, random_state):
    batch = None
    while batch is None or overlaps(test_hashes, batch):
        batch = generate_batch(batch_size=batch_size,
                               random_state=random_state)
    return batch
