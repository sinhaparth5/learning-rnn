import torch
import torch.nn as nn

class MusicRNN(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, dropout=0.5):
        super(MusicRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden=None):
        """Forward pass through the network"""
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
            
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Pass through fully connected layer
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        
        return out, hidden  # Make sure to return both output and hidden state
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
    
    def generate(self, seed, steps=500, temperature=1.0):
        """Generate new music sequence"""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            current_sequence = seed.clone()
            hidden = None
            generated = []
            
            for _ in range(steps):
                # Forward pass
                output, hidden = self.forward(current_sequence, hidden)  # Use self.forward explicitly
                
                # Get the last prediction
                next_notes = output[:, -1:, :]
                
                # Apply temperature sampling
                if temperature != 1.0:
                    next_notes = torch.log(next_notes + 1e-7)  # Add small epsilon to avoid log(0)
                    next_notes = next_notes / temperature
                    next_notes = torch.exp(next_notes)
                
                # Sample from the output distribution
                next_notes = torch.bernoulli(next_notes)
                
                # Append to generated sequence
                generated.append(next_notes)
                
                # Update current sequence
                current_sequence = next_notes
            
            # Concatenate all generated notes
            return torch.cat(generated, dim=1)