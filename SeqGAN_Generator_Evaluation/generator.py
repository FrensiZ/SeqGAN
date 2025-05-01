import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
        self.to(self.device)
       
    def forward(self, x, hidden=None):

        emb = self.embeddings(x)                    # [batch_size, sequence_length, embedding_dim]
        lstm_out, hidden = self.lstm(emb, hidden)   # lstm_out: [batch_size, sequence_length, hidden_dim]
        logits = self.output_layer(lstm_out)        # [batch_size, sequence_length, vocab_size]
        
        return logits, hidden
    
    def generate(self, num_samples):

        with th.no_grad():
            
            # Start token for all sequences
            x = th.full((num_samples, 1), self.start_token, dtype=th.long, device=self.device)
            hidden = None  # Let Pyth initialize the hidden state

            generated_sequences = th.zeros(num_samples, self.sequence_length, dtype=th.long, device=self.device)

            for i in range(self.sequence_length):
                # Forward pass
                emb = self.embeddings(x[:, -1:])  # Only use the last token
                lstm_out, hidden = self.lstm(emb, hidden)
                logits = self.output_layer(lstm_out)
                
                # Sample from distribution
                probs = F.softmax(logits.squeeze(1), dim=-1)
                next_token = th.multinomial(probs, 1)
                
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
    oracle_dataset = th.utils.data.TensorDataset(oracle_data)
    oracle_loader = th.utils.data.DataLoader(oracle_dataset, batch_size=batch_size, shuffle=True)
    
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
            x = batch_data[0].to(generator.device)
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
    
    #th.save(generator.state_dict(), 'generator_pretrained.pth')
    
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
    selected_log_probs = th.sum(log_probs * one_hot_targets, dim=-1)
    
    # Slice rewards to match (exclude reward for the start token)
    sequence_rewards = rewards[:, 1:]
    
    # Policy gradient loss: negative mean of (log_prob * reward)
    # We use negative because we're minimizing loss but want to maximize reward
    loss = -th.mean(selected_log_probs * sequence_rewards)
    
    # Backpropagate and update
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_seqgan(generator, discriminator, rollout, target_lstm, g_optimizer, d_optimizer, 
                num_epochs, batch_size, generated_num, g_steps, d_steps, k_epochs, 
                log_path, log_path_reward, device):
    
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
            disc_metrics = evaluate_discriminator(discriminator, target_lstm, generator, num_samples=1000)
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