import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



class Discriminator_LSTM_Frensi(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, device, num_layers=2):
        
        super(Discriminator_LSTM_Frensi, self).__init__()
        
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
        self.to(self.device)
    
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

        with th.no_grad():
            logits = self.forward(x)
            rewards = th.sigmoid(logits)     # Convert to probabilities
            return rewards
    
    def train_step(self, real_data, generated_data, optimizer):

        optimizer.zero_grad()
        
        # Prepare inputs and targets
        batch_size = real_data.size(0)

        real_data = real_data.to(self.device)
        generated_data = generated_data.to(self.device)

        inputs = th.cat([real_data, generated_data], dim=0)
        
        targets = th.cat([
            th.ones(batch_size, device=self.device), 
            th.zeros(batch_size, device=self.device)
        ], dim=0)
        
        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()

def pretrain_discriminator(target_lstm, generator, discriminator, optimizer, outer_epochs, inner_epochs, batch_size, generated_num, log_file, lr_patience, lr_decay):
        
    # Open log file
    log = open(log_file, 'w')
    log.write('Discriminator pre-training...\n')
    
    total_epochs = 0
    best_loss = float('inf')
    patience_counter = 0
    
    # Outer loop
    for outer_epoch in range(outer_epochs):

        # Generate positive samples from the oracle (only once)
        target_lstm.eval()
        with th.no_grad():
            positive_samples = target_lstm.generate(generated_num)
            
        # Generate new negative samples for each outer epoch
        generator.eval()
        with th.no_grad():
            negative_samples = generator.generate(generated_num)
        
        # Create data loaders for this outer epoch
        pos_loader = DataLoader(TensorDataset(positive_samples), batch_size=batch_size, shuffle=True)
        neg_loader = DataLoader(TensorDataset(negative_samples), batch_size=batch_size, shuffle=True)
        
        epoch_total_loss = 0
        epoch_batches = 0
        
        for inner_epoch in range(inner_epochs):
            
            # Set discriminator to training mode
            discriminator.train()
            
            total_loss = 0
            num_batches = 0
            
            # Iterate through batches
            for (pos_batch,), (neg_batch,) in zip(pos_loader, neg_loader):
                
                pos_batch = pos_batch.to(discriminator.device)
                neg_batch = neg_batch.to(discriminator.device)

                loss = discriminator.train_step(pos_batch, neg_batch, optimizer)
                total_loss += loss
                num_batches += 1
                
                epoch_total_loss += loss
                epoch_batches += 1
            
            total_epochs += 1
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            eval_metrics = evaluate_discriminator(discriminator, target_lstm, generator, num_samples=1000)
            
            log_str = f'epoch:\t{total_epochs}\tloss:\t{avg_loss:.4f}\t'
            log_str += f'accuracy:\t{eval_metrics["accuracy"]:.4f}\t'
            log_str += f'real_prob\t{eval_metrics["real_prob"]:.4f}\tfake_prob\t{eval_metrics["fake_prob"]:.4f}'
            
            log.write(log_str + '\n')
            log.flush()
        
        # Learning rate scheduling
        avg_epoch_loss = epoch_total_loss / epoch_batches if epoch_batches > 0 else float('inf')
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= lr_patience:
            # Reduce learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            log.write(f"Learning rate reduced to {optimizer.param_groups[0]['lr']}\n")
            log.flush()
            patience_counter = 0
    
    log.close()

def evaluate_discriminator(discriminator, target_lstm, generator, num_samples):
    
    discriminator.eval()
    target_lstm.eval()
    generator.eval()
        
    with th.no_grad():
        # Generate data
        real_data = target_lstm.generate(num_samples)
        fake_data = generator.generate(num_samples)
        
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






class Discriminator_CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, sequence_length, device, dropout_prob=0.1):
        
        super(Discriminator_CNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        self.sequence_length = sequence_length
        self.dropout_prob = dropout_prob
        self.l2_reg_lambda = 0.05
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
        self.to(self.device)
    
    def _init_weights(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.1)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.1)
    
    def _highway_layer(self, x, transform, gate):
        """Apply a single highway layer transformation"""
        transform_result = F.relu(transform(x))
        gate_result = th.sigmoid(gate(x))
        
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
        pooled_concat = th.cat(pooled_outputs, dim=1)
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

        with th.no_grad():
            logits = self.forward(x)
            rewards = th.sigmoid(logits)
            return rewards
    
    def train_step(self, real_data, generated_data, optimizer):

        optimizer.zero_grad()
        
        # Prepare inputs
        batch_size = real_data.size(0)
        
        # Forward pass for real data
        real_logits = self.forward(real_data)
        real_labels = th.ones(batch_size, device=self.device)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        
        # Forward pass for fake data
        fake_logits = self.forward(generated_data)
        fake_labels = th.zeros(batch_size, device=self.device)
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        
        # Combined loss
        loss = real_loss + fake_loss
        
        # Add L2 regularization if specified
        if self.l2_reg_lambda > 0:
            l2_reg = 0.0
            for param in self.parameters():
                l2_reg += th.norm(param, 2)
            loss += self.l2_reg_lambda * l2_reg
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        return loss.item()

class Discriminator_LSTM_Surag(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, device, num_layers=1):
        
        super(Discriminator_LSTM_Surag, self).__init__()
        
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
        self.to(self.device)
    
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

        with th.no_grad():
            logits = self.forward(x)
            rewards = th.sigmoid(logits)     # Get the probabilities
            return rewards
    
    def train_step(self, real_data, generated_data, optimizer):

        optimizer.zero_grad()
        
        # Prepare inputs and targets
        batch_size = real_data.size(0)

        real_data = real_data.to(self.device)
        generated_data = generated_data.to(self.device)

        inputs = th.cat([real_data, generated_data], dim=0)

        targets = th.cat([
            th.ones(batch_size, device=self.device), 
            th.zeros(batch_size, device=self.device)
        ], dim=0)
        
        # Forward pass
        logits = self.forward(inputs)
        
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()
