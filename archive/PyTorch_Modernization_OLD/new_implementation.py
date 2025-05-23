import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import re

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TargetLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, sequence_length, start_token, device='cpu'):
        
        super(TargetLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.device = device
        
        # Define layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize on device
        self.to(device)
       
    def forward(self, x, hidden=None):

        emb = self.embeddings(x)                    # [batch_size, sequence_length, embedding_dim]
        lstm_out, hidden = self.lstm(emb, hidden)   # lstm_out: [batch_size, sequence_length, hidden_dim]
        logits = self.output_layer(lstm_out)        # [batch_size, sequence_length, vocab_size]
        
        return logits, hidden
    
    def generate(self, num_samples):

        with torch.no_grad():
            
            # Start token for all sequences
            x = torch.full((num_samples, 1), self.start_token, dtype=torch.long, device=self.device)
            hidden = None  # Let PyTorch initialize the hidden state

            generated_sequences = torch.zeros(num_samples, self.sequence_length, dtype=torch.long, device=self.device)

            for i in range(self.sequence_length):
                # Forward pass
                emb = self.embeddings(x[:, -1:])  # Only use the last token
                lstm_out, hidden = self.lstm(emb, hidden)
                logits = self.output_layer(lstm_out)
                
                # Sample from distribution
                probs = F.softmax(logits.squeeze(1), dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add to sequence
                generated_sequences[:, i] = next_token.squeeze()
                
                # Update input for next step (only need the current token, not the entire history)
                x = next_token
            
            return generated_sequences

    def calculate_nll(self, generated_sequences):

        with torch.no_grad():
            # Use all tokens except the last one as input
            inputs = generated_sequences[:, :-1]
            
            # Use all tokens except the first one as targets
            targets = generated_sequences[:, 1:]
            
            # Forward pass
            logits, _ = self.forward(inputs)
            
            # Calculate negative log-likelihood
            nll = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction='mean')
            
            return nll.item()

    def load_params(self, params_path):
        """
        Load parameters from a TensorFlow list format.
        """
        try:
            with open(params_path, 'rb') as f:
                try:
                    params = pickle.load(f)
                except UnicodeDecodeError:
                    f.seek(0)
                    params = pickle.load(f, encoding='latin1')
        except Exception as e:
            print(f"Error loading pickle file: {str(e)}")
            return self
        
        with torch.no_grad():
            # 1. Embeddings
            self.embeddings.weight.copy_(torch.tensor(params[0], dtype=torch.float32))
            
            # 2. LSTM Parameters
            # Extract individual LSTM weights
            Wi, Ui, bi = params[1], params[2], params[3]  # Input gate
            Wf, Uf, bf = params[4], params[5], params[6]  # Forget gate
            Wo, Uo, bo = params[7], params[8], params[9]  # Output gate
            Wc, Uc, bc = params[10], params[11], params[12]  # Cell state
            
            # Concatenate the weights in PyTorch's expected format
            weight_ih = np.vstack([Wi, Wf, Wc, Wo])
            weight_hh = np.vstack([Ui, Uf, Uc, Uo])
            
            # Bias is also concatenated
            bias_ih = np.concatenate([bi, bf, bc, bo])
            bias_hh = np.zeros_like(bias_ih)
            
            # Copy to PyTorch model
            self.lstm.weight_ih_l0.copy_(torch.tensor(weight_ih, dtype=torch.float32))
            self.lstm.weight_hh_l0.copy_(torch.tensor(weight_hh, dtype=torch.float32))
            self.lstm.bias_ih_l0.copy_(torch.tensor(bias_ih, dtype=torch.float32))
            self.lstm.bias_hh_l0.copy_(torch.tensor(bias_hh, dtype=torch.float32))
            
            # 3. Output layer
            self.output_layer.weight.copy_(torch.tensor(params[13].T, dtype=torch.float32))
            self.output_layer.bias.copy_(torch.tensor(params[14], dtype=torch.float32))
        
        return self

    def save_params(self, path):
        torch.save(self.state_dict(), path)
        
    def save_samples(self, samples, file_path):
        with open(file_path, 'w') as f:
            for sample in samples.cpu().numpy():
                f.write(' '.join([str(int(x)) for x in sample]) + '\n')

class Generator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, sequence_length, start_token, device):
        
        super(Generator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.device = device
        
        # Define layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
                
        # Initialize on device
        self.to(device)
       
    def forward(self, x, hidden=None):

        emb = self.embeddings(x)                    # [batch_size, sequence_length, embedding_dim]
        lstm_out, hidden = self.lstm(emb, hidden)   # lstm_out: [batch_size, sequence_length, hidden_dim]
        logits = self.output_layer(lstm_out)        # [batch_size, sequence_length, vocab_size]
        
        return logits, hidden
    
    def generate(self, num_samples):

        with torch.no_grad():
            
            # Start token for all sequences
            x = torch.full((num_samples, 1), self.start_token, dtype=torch.long, device=self.device)
            hidden = None  # Let PyTorch initialize the hidden state

            generated_sequences = torch.zeros(num_samples, self.sequence_length, dtype=torch.long, device=self.device)

            for i in range(self.sequence_length):
                # Forward pass
                emb = self.embeddings(x[:, -1:])  # Only use the last token
                lstm_out, hidden = self.lstm(emb, hidden)
                logits = self.output_layer(lstm_out)
                
                # Sample from distribution
                probs = F.softmax(logits.squeeze(1), dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add to sequence
                generated_sequences[:, i] = next_token.squeeze()
                
                # Update input for next step (only need the current token, not the entire history)
                x = next_token
            
            return generated_sequences

    def pretrain_step(self, x, optimizer):

        optimizer.zero_grad()
            
        inputs = x[:, :-1]                  # Forward pass - input is all tokens except last one
        targets = x[:, 1:].contiguous()     # Target is all tokens except first one (shifted by 1)
        
        logits, _ = self.forward(inputs)
    
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        return loss.item()

def pretrain_generator(target_lstm, generator, optimizer, pre_epoch_num, batch_size, generated_num, eval_freq, lr_patience, lr_decay, log_path):
    
    print('Start pre-training...')

    # Open log file
    log = open(log_path, 'w')
    log.write('pre-training...\n')

    # For learning rate scheduling
    best_loss = float('inf')
    patience_counter = 0
        
    
    # Generate Oracle Data
    target_lstm.eval()
    oracle_data = target_lstm.generate(generated_num)
    
    # Create DataLoader
    oracle_dataset = torch.utils.data.TensorDataset(oracle_data)
    oracle_loader = torch.utils.data.DataLoader(oracle_dataset, batch_size=batch_size,shuffle=True)
    
    # Training loop
    for epoch in range(pre_epoch_num):

        epoch_loss = 0
        batch_count = 0

        # Evaluate using the oracle every eval_freq epochs
        if epoch % eval_freq == 0 or epoch == pre_epoch_num - 1:

            generated_samples = generator.generate(int(generated_num/10))
            
            # Calculate NLL using the oracle
            nll = target_lstm.calculate_nll(generated_samples)
            print(f'Epoch {epoch}, NLL: {nll:.4f}')

            # Log to file
            buffer = f'epoch:\t{epoch}\tnll:\t{nll:.5f}\n'
            log.write(buffer)
            log.flush()  # Ensure it's written immediately
        
        # Train on all batches
        for batch_data in oracle_loader:
            x = batch_data[0]
            loss = generator.pretrain_step(x, optimizer)
            epoch_loss += loss
            batch_count += 1
        
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / batch_count
        #print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')

        # Learning rate scheduling
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= lr_patience:
            # Reduce learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            print(f"Learning rate reduced to {optimizer.param_groups[0]['lr']}")
            patience_counter = 0

    log.close()
    
    torch.save(generator.state_dict(), 'generator_pretrained.pth')
    
    print('Pretraining finished!')
    
def generator_adversarial_update(generator, sequences, rewards, optimizer):

    optimizer.zero_grad()

    inputs = sequences[:, :-1]  # all but the last token
    targets = sequences[:, 1:]  # all but the first token
    logits, _ = generator(inputs)
    

    log_probs = F.log_softmax(logits, dim=-1)
    
    # Create a one-hot representation of the targets
    one_hot_targets = F.one_hot(targets, num_classes=generator.vocab_size).float()
    
    # Calculate the log probability of the selected actions
    # This gives us a tensor of shape [batch_size, seq_length-1, vocab_size]
    selected_log_probs = torch.sum(log_probs * one_hot_targets, dim=-1)
    
    # Slice rewards to match (exclude reward for the start token)
    sequence_rewards = rewards[:, 1:]
    
    # Policy gradient loss: negative mean of (log_prob * reward)
    # We use negative because we're minimizing loss but want to maximize reward
    loss = -torch.mean(selected_log_probs * sequence_rewards)
    
    # Backpropagate and update
    loss.backward()
    optimizer.step()
    
    return loss.item()

class Discriminator_Simple(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, num_layers=2, device='cpu'):
        super(Discriminator_Simple, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Move to device
        self.to(device)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM processing - get final hidden state
        _, (hidden, _) = self.lstm(embedded)
        
        # Take the final hidden state from the last layer
        final_hidden = hidden[-1]
        
        # Pass through the linear layer
        logits = self.fc(final_hidden)
        
        return logits.squeeze(-1)  # Return shape: [batch_size]
    
    def get_reward(self, x):

        with torch.no_grad():
            logits = self.forward(x)
            rewards = torch.sigmoid(logits)     # Convert to probabilities
            return rewards
    
    def train_step(self, real_data, generated_data, optimizer):

        optimizer.zero_grad()
        
        # Prepare inputs and targets
        batch_size = real_data.size(0)

        inputs = torch.cat([real_data, generated_data], dim=0)
        
        targets = torch.cat([
            torch.ones(batch_size, device=self.device), 
            torch.zeros(batch_size, device=self.device)
        ], dim=0)
        
        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()

class Discriminator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, num_layers=1, device='cpu'):
        
        super(Discriminator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer - added num_layers parameter
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0  # Dropout between LSTM layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        # Move to device
        self.to(device)
    
    def forward(self, x):

        # Embedding
        embedded = self.embeddings(x)                       # [batch_size, sequence_length, embedding_dim]
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)      # hidden: [num_layers, batch_size, hidden_dim]
        
        # Use the final hidden state from the last layer
        final_hidden = hidden[-1]                           # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(final_hidden)              # [batch_size, 1]
        
        return logits.squeeze(-1)                           # [batch_size]
    
    def get_reward(self, x):

        with torch.no_grad():
            logits = self.forward(x)
            rewards = torch.sigmoid(logits)     # Get the probabilities
            return rewards
    
    def train_step(self, real_data, generated_data, optimizer):

        optimizer.zero_grad()
        
        # Prepare inputs and targets
        batch_size = real_data.size(0)

        inputs = torch.cat([real_data, generated_data], dim=0)

        targets = torch.cat([
            torch.ones(batch_size, device=self.device), 
            torch.zeros(batch_size, device=self.device)
        ], dim=0)
        
        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()

class Discriminator_CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, sequence_length, dropout_prob=0.1, device='cpu'):
        
        super(Discriminator_CNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        self.sequence_length = sequence_length
        self.dropout_prob = dropout_prob
        self.l2_reg_lambda = 0.01
        self.highway_layers=1
        self.device = device
        
        # Word embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, 
                out_channels=self.num_filters[i], 
                kernel_size=(filter_size, embedding_dim), 
                stride=1
            ) for i, filter_size in enumerate(self.filter_sizes)
        ])
        
        self.num_filters_total = sum(self.num_filters)
        
        # Highway network components
        self.highway_transforms = nn.ModuleList()
        self.highway_gates = nn.ModuleList()
        
        for i in range(self.highway_layers):
            transform = nn.Linear(self.num_filters_total, self.num_filters_total)
            gate = nn.Linear(self.num_filters_total, self.num_filters_total)
            
            # Initialize gate bias to negative value to start with identity mappings
            nn.init.constant_(gate.bias, -2.0)
            
            self.highway_transforms.append(transform)
            self.highway_gates.append(gate)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(self.num_filters_total, 1)
        
        self._init_weights()
        self.to(device)
    
    def _init_weights(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.1)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.1)
    
    def _highway_layer(self, x, transform, gate):
        """Apply a single highway layer transformation"""
        transform_result = F.relu(transform(x))
        gate_result = torch.sigmoid(gate(x))
        
        return gate_result * transform_result + (1 - gate_result) * x
    
    def forward(self, x):
        embedded = self.embeddings(x)
        embedded = embedded.unsqueeze(1)
        
        pooled_outputs = []
        for i, conv in enumerate(self.convs):
            # Conv layer
            conv_out = F.relu(conv(embedded))
            
            # Max-pooling
            pool_out = F.max_pool2d(
                conv_out, 
                kernel_size=(conv_out.size(2), 1)
            )
            
            pooled_outputs.append(pool_out)
        
        # Concatenate all pooled features
        pooled_concat = torch.cat(pooled_outputs, dim=1)
        pooled_flat = pooled_concat.view(-1, self.num_filters_total)
        
        # Highway network
        highway_out = pooled_flat
        for transform, gate in zip(self.highway_transforms, self.highway_gates):
            highway_out = self._highway_layer(highway_out, transform, gate)
        
        # Dropout
        dropped = self.dropout(highway_out)
        
        # Output layer - return logits
        logits = self.output_layer(dropped).squeeze(-1)
        
        return logits
    
    def get_reward(self, x):

        with torch.no_grad():
            logits = self.forward(x)
            rewards = torch.sigmoid(logits)
            return rewards
    
    def train_step(self, real_data, generated_data, optimizer):

        optimizer.zero_grad()
        
        # Prepare inputs
        batch_size = real_data.size(0)
        
        # Forward pass for real data
        real_logits = self.forward(real_data)
        real_labels = torch.ones(batch_size, device=self.device)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        
        # Forward pass for fake data
        fake_logits = self.forward(generated_data)
        fake_labels = torch.zeros(batch_size, device=self.device)
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        
        # Combined loss
        loss = real_loss + fake_loss
        
        # Add L2 regularization if specified
        if self.l2_reg_lambda > 0:
            l2_reg = 0.0
            for param in self.parameters():
                l2_reg += torch.norm(param, 2)
            loss += self.l2_reg_lambda * l2_reg
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        return loss.item()

def evaluate_discriminator(discriminator, target_lstm, generator, num_samples, device):
    
    discriminator.eval()
    target_lstm.eval()
    generator.eval()
    
    with torch.no_grad():
        # Generate data
        real_data = target_lstm.generate(num_samples).to(device)
        fake_data = generator.generate(num_samples).to(device)
        
        # Get predictions - using get_reward to get probabilities
        real_preds = discriminator.get_reward(real_data)
        fake_preds = discriminator.get_reward(fake_data)
        
        # Calculate metrics
        real_correct = (real_preds >= 0.5).sum().item()
        fake_correct = (fake_preds < 0.5).sum().item()
        
        accuracy = (real_correct + fake_correct) / (2 * num_samples)
        real_prob = real_preds.mean().item()
        fake_prob = fake_preds.mean().item()
    
    metrics = {
        'accuracy': accuracy,
        'real_prob': real_prob,
        'fake_prob': fake_prob
    }
    
    return metrics

def pretrain_discriminator(target_lstm, generator, discriminator, optimizer, outer_epochs, inner_epochs, batch_size, generated_num, log_file, device):
    
    print('Start pre-training discriminator...')
    
    # Open log file
    log = open(log_file, 'w')
    log.write('Discriminator pre-training...\n')
    
    # Initial evaluation
    metrics = evaluate_discriminator(discriminator, target_lstm, generator, num_samples=1000, device=device)
    print(f"Initial accuracy: {metrics['accuracy']:.4f}")
    
    total_epochs = 0
    
    # Outer loop (50 times)
    for outer_epoch in range(outer_epochs):

        # Generate positive samples from the oracle (only once)
        target_lstm.eval()
        with torch.no_grad():
            positive_samples = target_lstm.generate(generated_num)
            
        # Generate new negative samples for each outer epoch
        generator.eval()
        with torch.no_grad():
            negative_samples = generator.generate(generated_num)
        
        # Create data loaders for this outer epoch
        pos_loader = DataLoader(TensorDataset(positive_samples), batch_size=batch_size, shuffle=True)
        neg_loader = DataLoader(TensorDataset(negative_samples), batch_size=batch_size, shuffle=True)
        
        for inner_epoch in range(inner_epochs):
            
            # Set discriminator to training mode
            discriminator.train()
            
            total_loss = 0
            num_batches = 0
            
            # Iterate through batches (same data for all inner epochs)
            for (pos_batch,), (neg_batch,) in zip(pos_loader, neg_loader):
                loss = discriminator.train_step(pos_batch, neg_batch, optimizer)
                total_loss += loss
                num_batches += 1
            
            total_epochs += 1
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            eval_metrics = evaluate_discriminator(discriminator, target_lstm, generator, num_samples=1000, device=device)
            
            log_str = f'epoch:\t{total_epochs}\tloss:\t{avg_loss:.4f}\t'
            log_str += f'accuracy:\t{eval_metrics["accuracy"]:.4f}\t'
            log_str += f'real_prob\t{eval_metrics["real_prob"]:.4f}\tfake_prob\t{eval_metrics["fake_prob"]:.4f}'
            
            print(log_str)
            log.write(log_str + '\n')
            log.flush()
    
    log.close()
    
    torch.save(discriminator.state_dict(), 'discriminator_pretrained.pth')
    
    print('Discriminator pretraining finished!')

class Rollout:
    
    def __init__(self, generator, discriminator, update_rate, device='cpu'):

        self.generator = generator
        self.discriminator = discriminator
        self.update_rate = update_rate
        self.device = device
        
        # Copy the generator's parameters
        self.generator_copy = type(generator)(
            generator.vocab_size,
            generator.embedding_dim,
            generator.hidden_dim,
            generator.sequence_length,
            generator.start_token,
            device
        ).to(device)
        
        # Copy the parameters
        self._update_generator_copy(update_rate=1.0)  # Full copy on initialization
        
    def _update_generator_copy(self, update_rate=None):

        if update_rate is None:
            update_rate = self.update_rate
            
        # Update the generator copy's parameters
        for target_param, source_param in zip(self.generator_copy.parameters(), self.generator.parameters()):
            target_param.data.copy_(
                update_rate * source_param.data + (1.0 - update_rate) * target_param.data
            )
    
    def update_params(self):

        self._update_generator_copy()
        
    def get_reward(self, sequences, rollout_num=16):

        batch_size, seq_length = sequences.shape
        rewards = torch.zeros(batch_size, seq_length, device=self.device)
        
        # For the last token, use the discriminator directly
        with torch.no_grad():
            final_rewards = self.discriminator.get_reward(sequences)
            rewards[:, -1] = final_rewards
        
        # For each position, perform rollout
        for position in range(seq_length - 1):
            position_rewards = torch.zeros(batch_size, device=self.device)
            
            # Run multiple rollouts
            for _ in range(rollout_num):
                # Complete the sequences from this position
                completions = self._monte_carlo_search(sequences, fixed_length=position + 1)
                
                # Get discriminator rewards for the completions
                with torch.no_grad():
                    completion_rewards = self.discriminator.get_reward(completions)
                    position_rewards += completion_rewards
            
            # Average rewards across rollouts
            rewards[:, position] = position_rewards / rollout_num
        
        return rewards
    
    def _monte_carlo_search(self, sequences, fixed_length):

        batch_size, seq_length = sequences.shape
        
        # Create output tensor with fixed tokens from the input
        completed_sequences = sequences.clone()
        
        # Set generator_copy to evaluation mode
        self.generator_copy.eval()
        
        with torch.no_grad():
            # Start with the fixed part
            current_input = sequences[:, :fixed_length]
            
            # Initialize hidden state with the fixed part
            h = None  # The LSTM will initialize its state
            
            # Process the fixed part to get the hidden state
            if fixed_length > 0:
                # Get embeddings for the fixed tokens
                emb = self.generator_copy.embeddings(current_input)
                
                # Process through LSTM (we'll discard the outputs)
                _, h = self.generator_copy.lstm(emb, h)
            
            # Generate the remaining tokens one at a time
            current_token = sequences[:, fixed_length-1:fixed_length] if fixed_length > 0 else torch.full(
                (batch_size, 1), self.generator_copy.start_token, dtype=torch.long, device=self.device
            )
            
            # Complete the sequence token by token
            for t in range(fixed_length, seq_length):
                # Get the next token distribution
                emb = self.generator_copy.embeddings(current_token)
                lstm_out, h = self.generator_copy.lstm(emb, h)
                logits = self.generator_copy.output_layer(lstm_out)
                
                # Sample from the distribution
                probs = F.softmax(logits.squeeze(1), dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add to the completed sequence
                completed_sequences[:, t] = next_token.squeeze()
                
                # Update for next iteration
                current_token = next_token
        
        return completed_sequences

def train_seqgan(generator, discriminator, rollout, target_lstm, g_optimizer, d_optimizer, num_epochs, batch_size, generated_num, g_steps, d_steps, k_epochs, log_path, log_path_reward):
    
    # Open log file
    log = open(log_path, 'w')
    log.write('epoch\tnll\tpg_loss\td_loss\td_accuracy\treal_prob\tfake_prob\tavg_reward\n')
    
    # Optional: Create a separate reward log file if needed
    reward_log = open(log_path_reward, 'w')
    reward_log.write('epoch\tposition\treward\n')
    
    for epoch in range(num_epochs):

        print(f"Epoch {epoch}")
        
        # GENERATOR TRAINING PHASE
        pg_losses = []
        avg_token_rewards = []
        
        for _ in range(g_steps):
            # Generate sequences
            generator.train()
            sequences = generator.generate(batch_size)
            
            # Get rewards using rollout
            rewards = rollout.get_reward(sequences)
            
            # Track per-token rewards
            avg_position_rewards = rewards.mean(dim=0).cpu().numpy()  # Average across batch for each position
            avg_token_rewards.append(avg_position_rewards.mean())  # Overall average reward

            # Log per-position rewards for EVERY epoch
            for pos, reward in enumerate(avg_position_rewards):
                reward_log.write(f"{epoch}\t{pos}\t{reward:.6f}\n")
            
            # Update generator using policy gradient
            pg_loss = generator_adversarial_update(generator, sequences, rewards, g_optimizer)
            pg_losses.append(pg_loss)
        
        # Flush the reward log every epoch to ensure data is written
        reward_log.flush()
        
        # Average PG loss for this epoch
        avg_pg_loss = sum(pg_losses) / len(pg_losses) if pg_losses else 0
        avg_reward = sum(avg_token_rewards) / len(avg_token_rewards) if avg_token_rewards else 0
        
        # Update rollout module with new generator parameters
        rollout.update_params()
        
        # DISCRIMINATOR TRAINING PHASE
        d_losses = []
        
        for _ in range(d_steps):
            # Generate new data for discriminator training
            target_lstm.eval()
            generator.eval()
            positive_examples = target_lstm.generate(generated_num)
            negative_examples = generator.generate(generated_num)
            
            # Create data loaders
            pos_loader = DataLoader(TensorDataset(positive_examples), batch_size=batch_size, shuffle=True)
            neg_loader = DataLoader(TensorDataset(negative_examples), batch_size=batch_size, shuffle=True)
            
            # Train discriminator for k epochs
            for _ in range(k_epochs):
                batch_d_losses = []
                for (pos_batch,), (neg_batch,) in zip(pos_loader, neg_loader):
                    d_loss = discriminator.train_step(pos_batch, neg_batch, d_optimizer)
                    batch_d_losses.append(d_loss)
                
                if batch_d_losses:
                    d_losses.append(sum(batch_d_losses) / len(batch_d_losses))
        
        # Average discriminator loss
        avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else 0
        
        # EVALUATION PHASE
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            
            generator.eval()
            
            # Generate samples for evaluation
            eval_samples = generator.generate(generated_num)
            # Calculate NLL using oracle
            nll = target_lstm.calculate_nll(eval_samples)
            
            # Evaluate discriminator performance
            disc_metrics = evaluate_discriminator(discriminator, target_lstm, generator, num_samples=1000, device=generator.device)
            d_accuracy = disc_metrics['accuracy']
            real_prob = disc_metrics['real_prob']
            fake_prob = disc_metrics['fake_prob']
            
            # Log all metrics
            metrics_str = f"Epoch {epoch}, NLL: {nll:.4f}, PG Loss: {avg_pg_loss:.4f}, D Loss: {avg_d_loss:.4f}, "
            metrics_str += f"D Acc: {d_accuracy:.4f}, Real Prob: {real_prob:.4f}, Fake Prob: {fake_prob:.4f}, Avg Reward: {avg_reward:.4f}"
            print(metrics_str)
            
            # Write to log file
            log.write(f"{epoch}\t{nll:.6f}\t{avg_pg_loss:.6f}\t{avg_d_loss:.6f}\t{d_accuracy:.6f}\t{real_prob:.6f}\t{fake_prob:.6f}\t{avg_reward:.6f}\n")
            log.flush()
    
    log.close()
    reward_log.close()
    print("Training completed!")

# Initialize models
VOCAB_SIZE = 5000
EMB_DIM = 32 
HIDDEN_DIM = 32 
SEQ_LENGTH = 20 
START_TOKEN = 0
PRE_EPOCH_NUM = 120
BATCH_SIZE = 64
SEED = 88
set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generated_num = 10000

# Discriminator hyperparameters
DIS_EMB_DIM = 64
DIS_HIDDEN_DIM = 128
DIS_DROPOUT = 0.01
DIS_BATCH_SIZE = 64
OUTER_EPOCHS = 50
INNER_EPOCHS = 1

log_folder = "logs"

# Create models
target_lstm = TargetLSTM(VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, device)
target_lstm.load_params(params_path='save/target_params_py3.pkl')
generator = Generator(VOCAB_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, device)
generator.load_state_dict(torch.load('generator_pretrained.pth', weights_only=False))

#discriminator = Discriminator(VOCAB_SIZE, DIS_EMB_DIM, DIS_HIDDEN_DIM, dropout_rate=0.0, num_layers=1, device=1)
#discriminator = Discriminator_Simple(VOCAB_SIZE, DIS_EMB_DIM, DIS_HIDDEN_DIM, DIS_DROPOUT, device=device)
discriminator = Discriminator_CNN(VOCAB_SIZE, DIS_EMB_DIM, SEQ_LENGTH, dropout_prob=0.1, device=device)

# Initialize optimizer
g_optimizer = torch.optim.Adam(generator.parameters(), lr=6e-3)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

# PRETRAINING
pretrain_generator(target_lstm, generator, g_optimizer, PRE_EPOCH_NUM, BATCH_SIZE, generated_num, eval_freq=5, lr_patience=5, lr_decay=0.5, log_path=f"{log_folder}/generator_train.txt")
pretrain_discriminator(target_lstm, generator, discriminator, d_optimizer, OUTER_EPOCHS, INNER_EPOCHS, DIS_BATCH_SIZE, generated_num, f"{log_folder}/discriminator_pretrain.txt", device)


# Adversarial training parameters
TOTAL_BATCH = 200
G_STEPS = 1       # Generator updates per iteration
D_STEPS = 5       # Discriminator update iterations
K_EPOCHS = 3      # Discriminator epochs per update
ROLLOUT_NUM = 16  # Number of rollouts for reward estimation
ROLLOUT_UPDATE_RATE = 0.8  # Rate for updating rollout parameters

# Load pretrained models 
generator.load_state_dict(torch.load('generator_pretrained.pth', weights_only=False))
discriminator.load_state_dict(torch.load('discriminator_pretrained.pth', weights_only=False))

# Initialize rollout module
rollout = Rollout(generator, discriminator, ROLLOUT_UPDATE_RATE, device)

# Initialize optimizers for adversarial training (potentially with different learning rates)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

# Start adversarial training
train_seqgan(
    generator, discriminator, rollout, target_lstm, 
    g_optimizer, d_optimizer, TOTAL_BATCH,
    BATCH_SIZE, generated_num, G_STEPS, D_STEPS, K_EPOCHS,
    log_path=f"{log_folder}/adversarial_training_log.txt", log_path_reward=f"{log_folder}/rewards_log.txt"
)