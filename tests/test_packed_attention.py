"""Check experimental varlen indexing independently before GPU benchmarks."""
import unittest
import torch
import torch.nn.functional as F

from scripts.benchmark_packed_attention import Layout, dense, packed_varlen


def separate_reference(q, k, v, cq, ck, mq, mk, **kwargs):
    rows = []
    for i in range(cq.numel() - 1):
        qs = q[cq[i]:cq[i+1]].transpose(0, 1)[None]
        ks = k[ck[i]:ck[i+1]].transpose(0, 1)[None]
        vs = v[ck[i]:ck[i+1]].transpose(0, 1)[None]
        rows.append(F.scaled_dot_product_attention(qs, ks, vs, is_causal=True,
                    enable_gqa=kwargs['enable_gqa'])[0].transpose(0, 1))
    return torch.cat(rows)


class PackedAttentionTests(unittest.TestCase):
    def test_shifted_varlen_matches_dense_outputs_and_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        for rows in ([[3, 5], [1, 2, 1, 4]], [[1]*8, [1]*8], [[8], [8]]):
            layout = Layout(rows, 'cpu')
            for strict in (False, True):
                with self.subTest(rows=rows, strict=strict):
                    inputs = [torch.randn(2, h, 8, 4, requires_grad=True) for h in (4, 2, 2)]
                    actual = packed_varlen(*inputs, layout, strict, separate_reference)
                    expected = dense(*inputs, layout.allowed(strict))
                    torch.testing.assert_close(actual, expected)
                    probe = torch.randn_like(actual)
                    a = torch.autograd.grad((actual*probe).sum(), inputs)
                    e = torch.autograd.grad((expected*probe).sum(), inputs)
                    for x, y in zip(a, e): torch.testing.assert_close(x, y)


if __name__ == '__main__': unittest.main()
