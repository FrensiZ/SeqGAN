import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Discriminator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, device, num_layers=2):
        
        super(Discriminator, self).__init__()
        
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

def pretrain_discriminator(target_lstm, generator, discriminator, optimizer, outer_epochs, inner_epochs, batch_size, generated_num, log_file):
        
    # Open log file
    log = open(log_file, 'w')
    log.write('Discriminator pre-training...\n')
    
    total_epochs = 0
    
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
            
            total_epochs += 1
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            eval_metrics = evaluate_discriminator(discriminator, target_lstm, generator, num_samples=1000)
            
            log_str = f'epoch:\t{total_epochs}\tloss:\t{avg_loss:.4f}\t'
            log_str += f'accuracy:\t{eval_metrics["accuracy"]:.4f}\t'
            log_str += f'real_prob\t{eval_metrics["real_prob"]:.4f}\tfake_prob\t{eval_metrics["fake_prob"]:.4f}'
            
            log.write(log_str + '\n')
            log.flush()
    
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

