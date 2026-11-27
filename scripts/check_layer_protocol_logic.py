"""Analytical scalar counterexamples and publication arithmetic, no model loads.

This does not reproduce paper experiments or estimate an LLM effect size.
"""
import hashlib
import json
import math
from pathlib import Path


def binary_kl(a, b):
    """KL(softmax([a,0]) || softmax([b,0])) in float64 scalars."""
    logza = math.log1p(math.exp(a))
    logzb = math.log1p(math.exp(b))
    p = math.exp(a - logza)
    return p * (a - b) - logza + logzb


def check():
    # Distinct commuting residual maps can each make a large contribution.
    a = lambda x: 2*x
    b = lambda x: 3*x
    x = 1
    assert b(a(x)) == a(b(x)) == 6
    assert a(x) == 2 and b(x) == 3
    gb = lambda x: b(x)-x
    assert gb(a(x))-gb(x) == 2
    # At a given input, BOTH duplications can agree while the swap differs.
    c = lambda x: x+1
    d = lambda x: 2
    x = 0
    canonical, cc, dd, swapped = d(c(x)), c(c(x)), d(d(x)), c(d(x))
    assert (canonical, cc, dd, swapped) == (2, 2, 2, 3)
    repl = max(binary_kl(canonical, cc), binary_kl(canonical, dd))
    inter = binary_kl(canonical, swapped)
    assert repl == 0 and inter > 0
    # If A,B are nonnegative directional mean KLs on identical observations,
    # M=max(A,B) and S=(A+B)/2 necessarily satisfy M/2 <= S <= M.
    published_max, published_sym = .036, .00101
    required_minimum = published_max/2
    assert published_sym < required_minimum
    # Exact ordinary binomial sign-test null: independent signs, p=1/2.
    one_sided = sum(math.comb(12,k) for k in range(10,13))/2**12
    two_sided = min(1.,2*one_sided)
    assert one_sided > .01 and two_sided > .01
    return dict(status='PASS_ANALYTICAL_CHECKS_ONLY',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scalar_commuting_example=dict(input=1,AB=6,BA=6,skip_A=3,skip_B=2,
            cross_update=2,scope='Exact scalar residual maps; not trained LLM measurements'),
        scalar_replacement_vs_interchange=dict(input=0,canonical=canonical,
            duplicate_A=cc,duplicate_B=dd,swapped=swapped,
            replacement_KL=repl,interchange_KL=inter,
            scope='Pointwise counterexample, not a universal distribution claim'),
        publication_checks=dict(source='https://arxiv.org/html/2605.16234v1',
            table2_first_row=dict(reported_max=published_max,reported_sym=published_sym,
                required_minimum_under_stated_same_aggregation=required_minimum,
                interpretation='Requires reporting or aggregation clarification; raw experiments not audited'),
            appendix_P1_sign_test=dict(successes=10,n=12,null_p=.5,
                one_sided_p=one_sided,two_sided_p=two_sided,
                caveat='Layer pairs need not be independent; this only recomputes the ordinary stated sign test')),
        no_model_or_data_loaded=True,no_remote_access=True)


if __name__ == '__main__':
    print(json.dumps(check(),indent=2,allow_nan=False))
