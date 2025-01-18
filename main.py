import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden 
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01 # Hidden to output
        self.bh = np.zeros((hidden_size, 1)) # Hidden biases
        self.by = np.zeros((output_size, 1)) # Output biases

    def forward(self, inputs, h_prev):
        """
            Forward pass through the RNN
            input: list of input vectors for each time step
            h_prev: initial hidden state
        """
        x_s, h_s, y_s = {}, {}, {}
        h_s[-1] = np.copy(h_prev)

        # Forward pass
        for t in range(len(inputs)):
            x_s[t] = np.array(inputs[t]).reshape(-1, 1)
                
            h_s[t] = np.tanh(
                np.dot(self.Wxh, x_s[t]) +
                np.dot(self.Whh, h_s[t-1]) +
                self.bh
            )

            y_s[t] = np.dot(self.Why, h_s[t] + self.by)

        return y_s, h_s, x_s

    def backward(self, y_s, h_s, x_s, targets):
        """
            Backward pass thorugh time (BPTT)
            y_s: output states
            h_s: hidden states
            x_s: input states
            targets: target values for each time step
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(h_s[0])

        loss = 0

        # Backward pass for each time step
        for t in reversed(range(len(targets))):
            # Compute loss (using MSE)
            dy = y_s[t] - targets[t].reshape(-1, 1)
            loss += np.sum(0.5 * dy * dy)

            # Gradients for Why and by
            dWhy += np.dot(dy, h_s[t].T)
            dby += dy

            # Initial gradient for hidden state
            dh = np.dot(self.Why.T, dy) + dh_next

            # Backdrop through tanh
            dtanh = (1 - h_s[t] * h_s[t]) * dh

            # Gradients for Wxh, Whh, and bh 
            dbh += dtanh
            dWxh += np.dot(dtanh, x_s[t].T)
            dWhh += np.dot(dtanh, h_s[t-1].T)

            # Save gradients for next iterations
            dh_next = np.dot(self.Whh.T, dtanh)

        # Clip gradients to prevent exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        gradients = {
            'dWxh': dWxh, 'dWhh': dWhh, 'dWhy': dWhy,
            'dbh': dbh, 'dby': dby
        }

        return loss, gradients

    def update_params(self, gradients):
        """ Update network parameters using gradients """
        self.Wxh -= self.learning_rate * gradients['dWxh']
        self.Whh -= self.learning_rate * gradients['dWhh']
        self.Why -= self.learning_rate * gradients['dWhy']
        self.bh -= self.learning_rate * gradients['dbh']
        self.by -= self.learning_rate * gradients['dby']

    def train_step(self, inputs, targets, h_prev):
        """ Single training step """
        # Forward pass
        y_s, h_s, x_s = self.forward(inputs, h_prev)

        # Backward pass
        loss, gradients = self.backward(y_s, h_s, x_s, targets)

        # Update parameters
        self.update_params(gradients)

        return loss, h_s[len(inputs) - 1]

# Example usage
def generate_simple_dataset(sequence_length, input_size, output_size, num_samples):
    """ Generate a simple dataset for demonstration """
    X = np.random.randn(num_samples, sequence_length, input_size)
    y = np.random.randn(num_samples, sequence_length, output_size)
    return X, y

if __name__ == "__main__":
    # Parameters
    input_size = 2
    hidden_size = 16
    output_size = 1
    sequence_length = 2
    num_epocs = 100
    num_samples = 10
    
    # Create RNN instance
    rnn = RNN(input_size, hidden_size, output_size)

    # Generate sample data 
    X, y = generate_simple_dataset(sequence_length, input_size, output_size, num_samples)

    # Training loop 
    for epoch in range(num_epocs):
        total_loss = 0
        h = np.zeros((hidden_size, 1)) # Initial hidden state

        for i in range(num_samples):
            inputs = [X[i][j] for j in range(sequence_length)]
            targets = [y[i][j] for j in range(sequence_length)]

            loss, h = rnn.train_step(inputs, targets, h)
            total_loss += loss

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/num_samples}')


