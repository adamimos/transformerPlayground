import random

class PDFA:
    def __init__(self, states, input_symbols, transition_function, initial_state):
        """
        Initialize a PDFA.
        Args:
            states: A set of states in the PDFA.
            input_symbols: A set of input symbols that the PDFA can accept.
            transition_function: A dictionary mapping (state, symbol) tuples to lists of (next_state, probability) tuples.
            initial_state: The state in which the PDFA starts.
        """
        self.states = states
        self.input_symbols = input_symbols
        self.transition_function = transition_function
        self.initial_state = initial_state
        self.current_state = initial_state

    def reset(self):
        """
        Reset the PDFA to its initial state.
        """
        self.current_state = self.initial_state

    def generate_sequence(self, length):
        """
        Generate a sequence of a given length.
        Args:
            length: The length of the sequence.
        Returns:
            A list of symbols.
        """
        sequence = []
        for _ in range(length):
            symbol = self.generate_symbol()
            sequence.append(symbol)
        return sequence

    def generate_symbol(self):
        """
        Generate a symbol and transition to the next state according to the transition probabilities from the current state.
        Returns:
            A symbol.
        """
        symbols, next_states_and_probabilities = zip(*[(symbol, transitions) for (state, symbol), transitions in self.transition_function.items() if state == self.current_state])
        probabilities = [probability for transitions in next_states_and_probabilities for _, probability in transitions]
        symbol = random.choices(symbols, probabilities)[0]
        self.current_state = self.transition_function[(self.current_state, symbol)][0][0]
        return symbol


class EvenProcessPDFA(PDFA):
    def __init__(self):
        states = {'A', 'B'}
        input_symbols = {0, 1}
        transition_function = {
            ('A', 0): [('A', 2/3)],  # remains in 'A' with probability 2/3 after outputting 0
            ('A', 1): [('B', 1/3)],  # moves to 'B' with probability 1/3 after outputting 1
            ('B', 0): [('A', 1)],  # always moves back to 'A' after outputting 0 (ends the pair of 1s)
            ('B', 1): [('A', 1)],  # always moves back to 'A' after outputting 1 (ends the pair of 1s)
        }
        initial_state = 'A'
        super().__init__(states, input_symbols, transition_function, initial_state)




class EvenProcessPDFA(PDFA):
    def __init__(self):
        states = {'A', 'B'}
        input_symbols = {0, 1}
        transition_function = {
            ('A', 0): [('A', 2/3)],  # remains in 'A' with probability 2/3 after outputting 0
            ('A', 1): [('B', 1/3)],  # moves to 'B' with probability 1/3 after outputting 1
            ('B', 1): [('A', 1)],  # always moves back to 'A' after outputting 1 (ends the pair of 1s)
        }
        initial_state = 'A'
        super().__init__(states, input_symbols, transition_function, initial_state)

class NevenProcessPDFA(PDFA):
    states = {'A', 'B', 'C'}
    input_symbols = {0, 1}
    transition_function = {
        ('A', 0): [('C', 0.24)],
        ('A', 1): [('B', 0.76)],
        ('B', 0): [('A', 0.38)],
        ('B', 1): [('A', 0.62)],
        ('C', 0): [('A', 1.00)],
    }
    initial_state = 'A'
    super().__init__(states, input_symbols, transition_function, initial_state)






if __name__ == '__main__':
    random.seed(1)  # for reproducibility
    process = EvenProcessPDFA()
    sequence = process.generate_sequence(100)
    print(sequence)
    # check that 1s only happen in pairs, except for the final group of 1s
    num_consecutive_ones = 0
    for i in range(len(sequence)):
        if sequence[i] == 1:
            num_consecutive_ones += 1
        else:
            if num_consecutive_ones % 2 == 1 and i != len(sequence) - 1:
                print("Error: odd number of consecutive ones at index", i)
            num_consecutive_ones = 0


    