"""CPU tests for the gradient direction and completion-only objective."""
import unittest
import torch
from .train import advantages, completion_mask, policy_loss


class GrpoTests(unittest.TestCase):
    def test_centered_within_prompt_not_across_prompts(self):
        result = advantages(torch.tensor([0., 1., 10., 10.]), 2)
        self.assertLess(float(result[0]), 0.)
        self.assertGreater(float(result[1]), 0.)
        self.assertEqual(result[2:].tolist(), [0., 0.])

    def test_policy_gradient_rewards_success(self):
        logp = torch.tensor([[-1., -2.], [-1., -2.]], requires_grad=True)
        old = logp.detach().clone()
        loss, kl = policy_loss(logp, old, old, torch.ones_like(logp),
                               torch.tensor([1., -1.]))
        loss.backward()
        self.assertTrue(torch.all(logp.grad[0] < 0))  # Descent raises good logp.
        self.assertTrue(torch.all(logp.grad[1] > 0))
        self.assertEqual(float(kl), 0.)

    def test_padding_is_not_trained_and_eos_is(self):
        tokens = torch.tensor([[4, 2, 2, 2], [4, 5, 6, 2]])
        mask = completion_mask(tokens, 2)
        self.assertEqual(mask.tolist(), [[True, True, False, False], [True]*4])
        logp = torch.full((2, 4), -1., requires_grad=True)
        old = logp.detach().clone()
        loss, _ = policy_loss(logp, old, old, mask, torch.ones(2))
        loss.backward()
        self.assertEqual(logp.grad[0, 2:].tolist(), [0., 0.])
        self.assertNotEqual(float(logp.grad[0, 1]), 0.)

    def test_clipping_stops_excessive_positive_update(self):
        logp = torch.tensor([[0.]], requires_grad=True)
        old = torch.tensor([[-1.]])
        loss, _ = policy_loss(logp, old, old, torch.ones_like(logp),
                              torch.ones(1), beta=0.)
        loss.backward()
        self.assertEqual(float(logp.grad), 0.)


if __name__ == "__main__":
    unittest.main()
