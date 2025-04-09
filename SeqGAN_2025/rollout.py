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

