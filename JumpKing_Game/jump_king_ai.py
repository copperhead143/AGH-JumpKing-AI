import os
import numpy as np
import random
import time
from collections import deque
import pygame
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

class JumpKingTrainer:
    """DQN-based AI trainer for Jump King with random power jumps"""
    
    def __init__(self, game):
        
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"GPU available: {physical_devices}")
            # Allow memory growth to prevent TF from taking all GPU memory
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        else:
            print("No GPU detected, using CPU.")
            
        # Game reference
        self.game = game
        self.king = game.king
        self.levels = game.levels
        
        # DQN parameters
        self.state_size = 8  # [x, y, speed, angle, is_on_ground, is_crouch, position_difference_from_start, level]
        
        # Modified action space: Each action is a jump with different direction/power range
        self.action_size = 6  # Different jump categories
        # Jump power categories: [weak_left, medium_left, strong_left, weak_right, medium_right, strong_right]
        
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        
        # Episode structure - 8 jumps per episode
        self.jumps_per_episode = 8
        self.current_jumps = 0
        self.best_height = 0
        self.previous_y = 0
        self.start_height = 0
        
        # Initialize model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Metrics
        self.total_jumps = 0
        self.current_episode = 0
        self.total_reward = 0
        self.heights_reached = []
        self.training_start_time = time.time()
    
    def _build_model(self):
        """Build more efficient Neural Network for DQN"""
        # Use smaller network architecture
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))  # Reduced from 64
        model.add(Dense(32, activation='relu'))  # Reduced from 64
        model.add(Dense(self.action_size, activation='linear'))
        
        # Use a more efficient optimizer
        optimizer = Adam(learning_rate=self.learning_rate, epsilon=1e-7)
        model.compile(loss='mse', optimizer=optimizer)
        
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def get_state(self):
        """Get current state representation"""
        x_normalized = self.king.rect_x / int(os.environ.get("screen_width"))
        y_normalized = self.king.rect_y / int(os.environ.get("screen_height"))
        speed_normalized = self.king.speed / self.king.maxSpeed if self.king.maxSpeed > 0 else 0
        angle_normalized = self.king.angle / (2 * np.pi)
        is_on_ground = 0 if self.king.isFalling else 1
        is_crouch = 1 if self.king.isCrouch else 0
        position_diff = (self.king.rect_y - self.previous_y) / int(os.environ.get("screen_height"))
        level_normalized = self.levels.current_level / self.levels.max_level if self.levels.max_level > 0 else 0
        
        return np.array([
            x_normalized, 
            y_normalized,
            speed_normalized,
            angle_normalized,
            is_on_ground,
            is_crouch,
            position_diff,
            level_normalized
        ])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def execute_action(self, action):
        """Execute selected jump action with random power based on category"""
        # Define power range for each category
        # Action 0-2: Left jumps (weak, medium, strong)
        # Action 3-5: Right jumps (weak, medium, strong)
        
        # Determine direction
        direction = 0  # 0 = left, 1 = right
        if action >= 3:
            direction = 1  # Right
            action_category = action - 3  # Convert to 0-2 range for power
        else:
            action_category = action  # Already in 0-2 range for left jumps
        
        # Determine power range based on category
        if action_category == 0:  # Weak
            power = random.uniform(0.2, 0.4)
        elif action_category == 1:  # Medium
            power = random.uniform(0.4, 0.7)
        else:  # Strong
            power = random.uniform(0.7, 1.0)
        
        # Execute the jump in the game
        # First, need to move in correct direction
        if direction == 0:  # Left
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_LEFT}))
        else:  # Right
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT}))
        
        # Apply crouch (simulate holding the jump button)
        self.king.isCrouch = True
        
        # Set jump power based on our category
        self.king.jumpCount = int(80 * power)  # Assuming 80 is max jumpCount
        
        # Return jump characteristics for debugging
        return {"direction": "left" if direction == 0 else "right", "power": power}
    
    def wait_for_jump_completion(self):
        """Wait until the jump is complete before continuing"""
        max_wait_frames = 120  # Safety to prevent infinite loops
        frames = 0
        
        # First ensure jump has started
        while not self.king.isJump and frames < 30:
            self.game._check_events()
            self.king.update()
            self.game._update_gamescreen()
            pygame.display.update()
            frames += 1
        
        # Then wait until character is on ground again
        frames = 0
        while (self.king.isJump or self.king.isFalling or self.king.isSplat) and frames < max_wait_frames:
            self.game._check_events()
            self.king.update()
            self.game._update_gamescreen()
            pygame.display.update()
            frames += 1
            
            # Add a small delay to better observe the jump
            pygame.time.delay(10)
    
    def replay(self):
        """Train the model with experiences from memory"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for current states
        targets = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values for actions taken
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, state, next_state, action, jump_info):
        """Calculate reward based on progress after a jump"""
        current_y = self.king.rect_y
        current_level = self.levels.current_level
        
        # Track vertical progress
        height_difference = self.previous_y - current_y
        
        # Calculate overall height including level progression
        overall_height = (current_level * int(os.environ.get("screen_height"))) + (int(os.environ.get("screen_height")) - current_y)
        height_gain = overall_height - self.start_height
        
        # Strong reward for vertical progress
        position_reward = height_gain * 3
        
        # Extra reward for level transition (going up)
        if current_level > int(state[7] * self.levels.max_level):
            position_reward += 300
        
        # Penalty for falling down a level
        if current_level < int(state[7] * self.levels.max_level):
            position_reward -= 150
        
        # Update best height if improved
        if overall_height > self.best_height:
            position_reward += (overall_height - self.best_height) * 2
            self.best_height = overall_height
        
        # Penalties
        penalty = 0
        
        # Penalty for hitting obstacles
        if self.king.collideTop or self.king.collideRight or self.king.collideLeft:
            penalty -= 15
        
        # Penalty for splat
        if self.king.isSplat:
            penalty -= 30
        
        # Encourage exploration with jump variety
        jump_variety_bonus = 5  # Small bonus for variety
        
        return position_reward + penalty + jump_variety_bonus
    
    def train(self):
        """Train with modified episode structure (8 jumps per episode)"""
        print("Starting DQN training for Jump King with 8-jump episodes...")
        
        # Set environment variables
        os.environ["gaming"] = "True"
        os.environ["active"] = "True"
        os.environ["pause"] = "False"
        os.environ["music"] = "False"
        os.environ["ambience"] = "False"
        os.environ["sfx"] = "False"
        
        # Training parameters
        episodes = 1000
        save_interval = 10  # Save model every 10 episodes
        
        for episode in range(episodes):
            # Reset environment
            self.current_episode = episode + 1
            self.king.reset()
            self.current_jumps = 0
            self.previous_y = self.king.rect_y
            self.start_height = (self.levels.current_level * int(os.environ.get("screen_height"))) + (int(os.environ.get("screen_height")) - self.king.rect_y)
            self.best_height = self.start_height
            
            episode_rewards = []
            
            print(f"Starting Episode {episode+1}/{episodes}")
            
            # Each episode consists of exactly 8 jumps
            while self.current_jumps < self.jumps_per_episode:
                # Wait until the character is on the ground
                while self.king.isFalling or self.king.isJump or self.king.isSplat:
                    self.game._check_events()
                    self.king.update()
                    self.game._update_gamescreen()
                    pygame.display.update()
                    pygame.time.delay(10)
                
                # Get current state
                state = self.get_state()
                self.previous_y = self.king.rect_y
                
                # Choose action
                action = self.act(state)
                
                # Execute action (jump with random power)
                jump_info = self.execute_action(action)
                
                # Wait for jump to complete
                self.wait_for_jump_completion()
                
                # Get next state
                next_state = self.get_state()
                
                # Calculate reward
                reward = self.calculate_reward(state, next_state, action, jump_info)
                episode_rewards.append(reward)
                
                # Check if this is the last jump in the episode
                done = (self.current_jumps == self.jumps_per_episode - 1)
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Update counters
                self.current_jumps += 1
                self.total_jumps += 1
                
                # Print jump info
                current_height = (self.levels.current_level * int(os.environ.get("screen_height"))) + (int(os.environ.get("screen_height")) - self.king.rect_y)
                print(f"Jump {self.current_jumps}/8: {jump_info['direction']} power={jump_info['power']:.2f}, Height: {current_height}, Reward: {reward:.2f}")
                
                # Train the model
                self.replay()
                
                # Update target network periodically
                if self.total_jumps % 20 == 0:
                    self.update_target_model()
            
            # Episode complete (8 jumps done)
            episode_total_reward = sum(episode_rewards)
            final_height = (self.levels.current_level * int(os.environ.get("screen_height"))) + (int(os.environ.get("screen_height")) - self.king.rect_y)
            self.heights_reached.append(final_height)
            
            # Print episode stats
            print(f"Episode {episode+1} completed")
            print(f"  Total Reward: {episode_total_reward:.2f}")
            print(f"  Starting Height: {self.start_height}")
            print(f"  Final Height: {final_height}")
            print(f"  Height Gain: {final_height - self.start_height}")
            print(f"  Best height this episode: {self.best_height}")
            print(f"  Level: {self.levels.current_level}")
            print(f"  Epsilon: {self.epsilon:.4f}")
            print("-" * 40)
            
            # Save model periodically
            if episode % save_interval == 0:
                model_path = f"AI_Models/jumpking_ai_checkpoint_{episode}.h5"
                self.save_model(model_path)
                print(f"Checkpoint saved at episode {episode}")
            
            # Save best model if performance improved
            if len(self.heights_reached) > 1 and final_height > max(self.heights_reached[:-1]):
                self.save_model("AI_Models/jumpking_ai_best.h5")
                print("New best model saved!")
            
            # Log progress
            with open("AI_Models/training_progress.txt", "a") as f:
                f.write(f"Episode {episode+1}: Reward={episode_total_reward:.2f}, Final Height={final_height}, Level={self.levels.current_level}, Epsilon={self.epsilon:.4f}\n")
        
        # Save final model
        self.save_model("AI_Models/jumpking_ai_final.h5")
        print("Training completed! Final model saved.")
        
    def save_model(self, filepath):
        """Save the model to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the model from disk with compatibility fixes"""
        try:
            # Try standard loading first
            self.model = load_model(filepath)
        except TypeError as e:
            if "Could not locate function 'mse'" in str(e):
                print("Handling MSE compatibility issue...")
                # Create a new model with the same architecture
                new_model = self._build_model()
                
                # Load just the weights instead of the full model
                import h5py
                with h5py.File(filepath, 'r') as f:
                    weight_names = [n.decode('utf8') for n in f.attrs['weight_names']]
                    for name in weight_names:
                        g = f[name]
                        weights = [np.array(g[wn]) for wn in g.attrs['weight_names']]
                        new_model.set_weights(weights)
                
                self.model = new_model
            else:
                # Re-raise if it's a different error
                raise
                
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        print(f"Model loaded from {filepath}")
    
    def load_and_run(self, model_path):
        """Load a trained model and run it in the game with 8-jump episodes"""
        os.environ["gaming"] = "True"  
        os.environ["active"] = "True"
        os.environ["pause"] = "False"
        
        # Load model
        self.load_model(model_path)
        self.epsilon = 0.05  # Small exploration during running for variety
        
        print(f"Running model from {model_path} in 8-jump episodes...")
        
        episode = 1
        
        while True:
            # Reset for new episode
            self.current_jumps = 0
            self.previous_y = self.king.rect_y
            self.start_height = (self.levels.current_level * int(os.environ.get("screen_height"))) + (int(os.environ.get("screen_height")) - self.king.rect_y)
            
            print(f"\nStarting Demo Episode {episode}")
            
            # Run 8 jumps
            while self.current_jumps < self.jumps_per_episode:
                # Wait until on ground
                while self.king.isFalling or self.king.isJump or self.king.isSplat:
                    self.game.clock.tick(60)  # Full speed for running
                    self.game._check_events()
                    self.king.update()
                    self.game._update_gamescreen()
                    self.game._update_guistuff()
                    pygame.display.update()
                    self.game._update_audio()
                
                # Get state
                state = self.get_state()
                
                # Choose action
                action = self.act(state)
                
                # Execute jump
                jump_info = self.execute_action(action)
                
                # Print jump info
                current_height = (self.levels.current_level * int(os.environ.get("screen_height"))) + (int(os.environ.get("screen_height")) - self.king.rect_y)
                print(f"Jump {self.current_jumps + 1}/8: {jump_info['direction']} power={jump_info['power']:.2f}, Height: {current_height}")
                
                # Wait for jump completion
                self.wait_for_jump_completion()
                
                # Update counters
                self.current_jumps += 1
                
                # Handle game updates
                self.game._update_guistuff()
                self.game._update_audio()
            
            # Episode complete
            final_height = (self.levels.current_level * int(os.environ.get("screen_height"))) + (int(os.environ.get("screen_height")) - self.king.rect_y)
            print(f"Episode {episode} complete - Height: {final_height}, Level: {self.levels.current_level}")
            
            # Pause briefly between episodes
            pygame.time.delay(2000)
            
            episode += 1
