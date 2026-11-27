"""Synthetic tasks where input-dependent depth, revisiting a module and STOP should matter (docs/DOT_SYNTH_20261124.md).

Every task has a fixed-length prompt with the answer at the end, so config.prefix = answer start and the standard
token_losses (and the DOT rollouts) score only answer tokens. Difficulty varies WITHOUT changing the sequence length or
introducing unseen tokens, so models can be tested beyond the training difficulty.

  chain : N nodes form disjoint chains (node -> next node, last node -> END), listed as shuffled (node, successor)
          pairs. QUERIES queries per sequence: a start node x, answer = the last node of x's chain, k hops away. The
          graph is always CHAINS chains of CHAIN_LEN nodes, only the query distances change (chance = 1 / CHAINS).
          The model must follow pointers until END: repeated lookups (revisiting) and a data-dependent stop.
          Train k = 1..4, test k = 1..7. (s01 used one query per sequence and random distractor chains: every model
          stayed at a chance level that rose with k.)
  opseq : apply a listed sequence of string operations, placed in random slots among M_MAX (NOP elsewhere), to an
          8-digit string; output the result. Train m = 1..3 operations, test m = 1..6. (s01 left slots 4..6 always NOP,
          which made m >= 4 a positional shift rather than a longer composition.)
"""
import numpy as np
import torch

PAD, SEP, QUERY, NOP, END = 0, 1, 2, 3, 4
NODE0, CHAINS, CHAIN_LEN, QUERIES, SYMBOLS = 16, 4, 8, 8, 48  # node ids [16, 64)
N_NODES = CHAINS * CHAIN_LEN
DIGIT0 = 64  # digits 0..9
OP0 = 80  # operation tokens
OPS = ('rev', 'rotl', 'inc', 'swap')
STR_LEN, M_MAX = 8, 6
VOCAB = 128


def chain_spec():
    length = 2 * N_NODES + 1 + 2 * QUERIES  # pairs, SEP, (start, answer) x QUERIES
    return dict(length=length, prefix=2 * N_NODES + 1)


def chain_batch(n, ks, rng):
    """n sequences, QUERIES queries each; every query's hop count is drawn from ks. The graph is always CHAINS chains
    of CHAIN_LEN nodes (random labels, shuffled pair order), so only the query distances change between train and test
    and a 'guess a chain end' strategy scores 1 / CHAINS at every distance."""
    spec = chain_spec()
    out = np.zeros((n, spec['length']), dtype=np.int64)
    ks = np.asarray(ks)
    k_all = np.zeros((n, QUERIES), dtype=np.int64)
    for r in range(n):
        labels = (NODE0 + rng.permutation(SYMBOLS)[:N_NODES]).reshape(CHAINS, CHAIN_LEN)
        succ = {int(c): (int(chain[j + 1]) if j + 1 < CHAIN_LEN else END) for chain in labels for j, c in enumerate(chain)}
        keys = rng.permutation(np.fromiter(succ.keys(), dtype=np.int64))
        out[r, :2 * N_NODES] = np.stack([keys, [succ[int(a)] for a in keys]], 1).reshape(-1)
        out[r, 2 * N_NODES] = SEP
        for q in range(QUERIES):
            k = int(rng.choice(ks))
            chain = labels[int(rng.integers(CHAINS))]
            k_all[r, q] = k
            out[r, spec['prefix'] + 2 * q:spec['prefix'] + 2 * q + 2] = (chain[CHAIN_LEN - 1 - k], chain[-1])
    return out, k_all


def chain_answers():
    """Indices of answer tokens within the scored range (positions prefix..length-1); each answer is its own unit."""
    return [[2 * q + 1] for q in range(QUERIES)]


def _apply(op, s):
    if op == 'rev':
        return s[::-1]
    if op == 'rotl':
        return s[1:] + s[:1]
    if op == 'inc':
        return [(c + 1) % 10 for c in s]
    if op == 'swap':
        t = list(s)
        for i in range(0, len(t) - 1, 2):
            t[i], t[i + 1] = t[i + 1], t[i]
        return t
    raise ValueError(op)


def opseq_spec():
    length = M_MAX + 1 + STR_LEN + 1 + STR_LEN  # ops (NOP padded), SEP, string, QUERY, answer string
    return dict(length=length, prefix=length - STR_LEN)


def opseq_batch(n, ms, rng):
    """Operations sit in random slots among M_MAX (order kept, NOP elsewhere), so every slot carries operations in
    training; only the number of operations changes at test time."""
    spec = opseq_spec()
    out = np.zeros((n, spec['length']), dtype=np.int64)
    m = rng.choice(np.asarray(ms), size=n)
    for r in range(n):
        ops = [OPS[i] for i in rng.integers(0, len(OPS), size=int(m[r]))]
        s = [int(c) for c in rng.integers(0, 10, size=STR_LEN)]
        t = s
        for op in ops:
            t = _apply(op, t)
        slots = np.sort(rng.choice(M_MAX, size=len(ops), replace=False))
        op_tokens = [NOP] * M_MAX
        for slot, o in zip(slots, ops):
            op_tokens[slot] = OP0 + OPS.index(o)
        out[r] = op_tokens + [SEP] + [DIGIT0 + c for c in s] + [QUERY] + [DIGIT0 + c for c in t]
    return out, m


def opseq_answers():
    return [list(range(STR_LEN))]  # one unit: the whole output string


TASKS = {
    'chain': dict(spec=chain_spec, batch=chain_batch, answers=chain_answers, train=(1, 2, 3, 4),
                  test=(1, 2, 3, 4, 5, 6, 7)),
    'opseq': dict(spec=opseq_spec, batch=opseq_batch, answers=opseq_answers, train=(1, 2, 3), test=(1, 2, 3, 4, 5, 6)),
}


def to_device(tokens, device):
    return torch.as_tensor(tokens, dtype=torch.long, device=device)
