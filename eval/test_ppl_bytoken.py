import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from eval.ppl_bytoken import accumulate_bytoken


class TinyCausalModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, input_ids, labels):
        # Deterministic logits whose row depends on the current input token.
        classes = torch.arange(self.vocab_size, device=input_ids.device)
        logits = -(
            classes.view(1, 1, -1) - input_ids.unsqueeze(-1)
        ).abs().float()
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(logits=logits, loss=loss)


class PPLByTokenTest(unittest.TestCase):
    def test_sliding_windows_reconstruct_full_target_nll(self):
        vocab_size = 5
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 1, 0]])
        model = TinyCausalModel(vocab_size)
        nll_sum, count, stats = accumulate_bytoken(
            model,
            input_ids,
            max_length=4,
            stride=2,
            device="cpu",
            vocab_size=vocab_size,
        )

        full = model(input_ids, labels=input_ids)
        token_nll = F.cross_entropy(
            full.logits[:, :-1].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1),
            reduction="none",
        ).double()
        expected_count = torch.bincount(
            input_ids[:, 1:].reshape(-1), minlength=vocab_size
        )
        expected_nll = torch.bincount(
            input_ids[:, 1:].reshape(-1),
            weights=token_nll,
            minlength=vocab_size,
        )

        self.assertEqual(stats["num_tokens"], input_ids.numel() - 1)
        self.assertLess(stats["max_chunk_gap"], 1e-6)
        torch.testing.assert_close(torch.from_numpy(count), expected_count)
        torch.testing.assert_close(
            torch.from_numpy(nll_sum), expected_nll, rtol=0, atol=1e-12
        )
        self.assertAlmostEqual(stats["loss"], full.loss.item(), places=6)
        self.assertAlmostEqual(stats["perplexity"], math.exp(full.loss.item()), places=6)


if __name__ == "__main__":
    unittest.main()
