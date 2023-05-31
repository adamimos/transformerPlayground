import unittest
import torch
from transformer_lib import TransformerModel

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_tokens = 5
        self.vocab_size = 10
        self.residual_dim = 16
        self.num_layers = 2
        self.num_heads = 4
        self.mlp_dim = 64
        self.model = TransformerModel(self.residual_dim, self.vocab_size, self.num_tokens, self.num_layers, self.num_heads, self.mlp_dim)

    def test_forward(self):
        x = torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.num_tokens))
        y = self.model(x)
        self.assertEqual(y.shape, (self.batch_size, self.num_tokens, self.vocab_size))

if __name__ == '__main__':
    unittest.main()