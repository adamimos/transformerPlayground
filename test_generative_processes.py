import unittest
import random
from PDFA import EvenProcess

class TestEvenProcess(unittest.TestCase):

    def setUp(self):
        random.seed(1)  # for reproducibility
        self.process = EvenProcess()

    def test_initial_state(self):
        self.assertIn(self.process.current_state, ['A', 'B'])

    def test_sample(self):
        sample, state = self.process.sample()
        self.assertIn(sample, [0, 1])
        self.assertIn(state, ['A', 'B'])

    def test_generate_series(self):
        samples = self.process.generate_series(1000, return_states=False)
        self.assertEqual(len(samples), 1000)

    def test_generate_series_with_states(self):
        samples_states = self.process.generate_series(1000, return_states=True)
        self.assertEqual(len(samples_states), 1000)
        for sample, state in samples_states:
            self.assertIn(sample, [0, 1])
            self.assertIn(state, ['A', 'B'])

    def test_state_transition(self):
        for _ in range(1000):
            if self.process.current_state == 'A':
                sample, state = self.process.sample()
                if sample == 0:
                    self.assertEqual(state, 'A')
                else:
                    self.assertEqual(state, 'B')
            else:
                sample, state = self.process.sample()
                self.assertEqual(state, 'A')  # Always transition back to 'A' from 'B'

if __name__ == '__main__':
    unittest.main()
