import numpy as np
import tensorflow as tf
import random
import os
import math
from collections import deque

class JumpKingAI:
    """Neural network to play Jump King and make upward progress through levels"""
    
    def __init__(self, state_size, action_size):
        # Define input state size and available actions
        self.state_size = state_size  # position, velocity, etc.
        self.action_size = action_size  # possible commands
        
        # Hyperparameters
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Memory for experience replay
        self.memory = deque(maxlen=2000)
        
        # Neural Network model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Tracking stats
        self.highest_y = 1000  # starting with a high value (lower is better in this game)
        self.highest_level = 0
        self.steps_since_progress = 0
    
    def _build_model(self):
        """Build a neural network model for deep Q learning"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Return action based on current state using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train network with batch of experiences from memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
    
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)


class JumpKingEnvironment:
    """Interface between the game and the AI"""
    
    def __init__(self, game):
        self.game = game
        self.king = game.king
        self.levels = game.levels
        
        # Define action space
        # 0: Do nothing (stand)
        # 1: Walk left
        # 2: Walk right
        # 3: Jump straight up (charge + release)
        # 4: Jump left (charge + release while holding left)
        # 5: Jump right (charge + release while holding right)
        self.actions = [
            None,  # Do nothing
            "WalkLeft",  # Walk left
            "WalkRight",  # Walk right
            "Jump",  # Jump up
            "JumpLeft",  # Jump left
            "JumpRight"  # Jump right
        ]
        
        # State parameters
        # [x_pos, y_pos, x_velocity, y_velocity, angle, speed, is_on_ground, current_level]
        self.state_size = 8
        self.action_size = len(self.actions)
        
        # Progress tracking
        self.best_y = float('inf')  # Lower y values are better (game coordinates)
        self.best_level = 0
        self.progress_counter = 0
        self.last_y = self.king.rect_y
        self.last_level = self.levels.current_level
        
        # For calculating rewards
        self.frames_since_input = 0
        self.consecutive_no_progress = 0
        self.last_action = None
    
    def get_state(self):
        """Get current state of the game for AI input"""
        king = self.king
        
        # Calculate if king is on ground
        on_ground = 0
        if not king.isFalling:
            on_ground = 1
        
        # Create state vector
        state = np.array([
            king.rect_x / 480.0,  # Normalize x position
            king.rect_y / 360.0,  # Normalize y position
            math.sin(king.angle) * king.speed / 11.0,  # Normalized x velocity
            -math.cos(king.angle) * king.speed / 11.0,  # Normalized y velocity
            king.angle / (2 * math.pi),  # Normalized angle
            king.speed / 11.0,  # Normalized speed
            on_ground,  # Binary: on ground or not
            self.levels.current_level / 42.0  # Normalized current level
        ])
        
        return np.reshape(state, [1, self.state_size])
    
    def perform_action(self, action_index):
        """Execute the selected action in the game"""
        action = self.actions[action_index]
        
        # Don't do anything if king is already falling
        if self.king.isFalling and action_index != 0:
            return
        
        # If king is on ground, execute command
        if action:
            self.last_action = action
            # Execute the robot command in the king class
            self.king._robot_check_events(action)
            self.frames_since_input = 0
        else:
            self.frames_since_input += 1
    
    def compute_reward(self):
        """Calculate reward based on progress"""
        reward = 0
        king = self.king
        
        # Get current position and level
        current_y = king.rect_y
        current_level = self.levels.current_level
        
        # Check if we've gone up in levels (major progress)
        if current_level > self.last_level:
            reward += 100  # Major reward for level progress
            self.best_level = current_level
            self.consecutive_no_progress = 0
            print(f"LEVEL UP! Now on level {current_level}")
        
        # Check if we've made height progress within the same level
        if current_level == self.last_level:
            # Lower y is better (moving up the screen)
            if current_y < self.last_y:
                progress = (self.last_y - current_y) / 10.0  # Scale the progress
                reward += progress
                
                # Extra reward for best height
                if current_y < self.best_y:
                    self.best_y = current_y
                    reward += 5  # Bonus for new best height
                    self.consecutive_no_progress = 0
                    print(f"New best height: {self.best_y:.1f} on level {current_level}")
                
            # Small negative reward for falling (moving down)
            elif current_y > self.last_y + 5:  # Only penalize significant falls
                reward -= 0.1
        
        # Penalize being stuck
        if self.frames_since_input > 60:  # If we haven't input a command for a while
            reward -= 0.5
        
        # Penalize being "splat" (failed landing)
        if king.isSplat:
            reward -= 2
        
        # Penalize for level drop (going down levels)
        if current_level < self.last_level:
            reward -= 10
        
        # Check for no progress
        if abs(current_y - self.last_y) < 1 and current_level == self.last_level:
            self.consecutive_no_progress += 1
        else:
            self.consecutive_no_progress = 0
        
        # Larger penalty if stuck in same position for too long
        if self.consecutive_no_progress > 200:
            reward -= 1
        
        # Update last position
        self.last_y = current_y
        self.last_level = current_level
        
        return reward
    
    def is_done(self):
        """Check if episode should end"""
        # End episode if we've been stuck for too long
        if self.consecutive_no_progress > 1000:
            return True
            
        # End episode if we reach the top level
        if self.levels.current_level >= 42:
            return True
            
        return False


class JumpKingTrainer:
    """Handle training the AI to play Jump King"""
    
    def __init__(self, game):
        self.game = game
        self.env = JumpKingEnvironment(game)
        self.agent = JumpKingAI(self.env.state_size, self.env.action_size)
        
        # Training parameters
        self.batch_size = 32
        self.episodes = 1000
        self.max_steps = 5000
        
        # Saving parameters
        self.save_freq = 50  # Save model every 50 episodes
        self.model_dir = "AI_Models"
        
        # Create directory for saving models if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def train(self):
        """Train the agent"""
        # Ensure the game is active but paused
        os.environ["gaming"] = "1"
        os.environ["active"] = "1"
        
        for episode in range(self.episodes):
            print(f"Episode: {episode + 1}/{self.episodes}")
            
            # Reset the game and environment
            self.reset_game()
            
            # Get initial state
            state = self.env.get_state()
            
            # Set pause to 0 to allow game to run
            os.environ["pause"] = "0"
            
            # Run episode
            for step in range(self.max_steps):
                # Get action from agent
                action = self.agent.act(state)
                
                # Perform action
                self.env.perform_action(action)
                
                # Update game logic for one frame
                self.game._update_gamestuff()
                self.game._update_gamescreen()
                
                # Get new state and reward
                next_state = self.env.get_state()
                reward = self.env.compute_reward()
                done = self.env.is_done()
                
                # Store experience in memory
                self.agent.remember(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                
                # Train agent
                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)
                
                # If episode is done, exit loop
                if done:
                    break
            
            # Update target model
            self.agent.update_target_model()
            
            # Save model periodically
            if episode % self.save_freq == 0:
                self.agent.save(f"{self.model_dir}/jumpking_ai_ep{episode}.h5")
                print(f"Model saved at episode {episode}")
            
            # Print episode stats
            print(f"Episode {episode} - Steps: {step}, Best Level: {self.env.best_level}, Best Y: {self.env.best_y}")
            
            # Pause game between episodes
            os.environ["pause"] = "1"
    
    def reset_game(self):
        """Reset the game state for a new episode"""
        # Reset king and level
        self.game.king.reset()
        self.game.levels.reset()
        
        # Reset environment tracking
        self.env.best_y = float('inf')
        self.env.best_level = 0
        self.env.consecutive_no_progress = 0
        self.env.last_y = self.game.king.rect_y
        self.env.last_level = self.game.levels.current_level
        self.env.frames_since_input = 0
    
    def load_and_run(self, model_path):
        """Load a trained model and run the game"""
        self.agent.load(model_path)
        self.agent.epsilon = 0.01  # Low epsilon for mostly exploitation
        
        # Start the game
        os.environ["gaming"] = "1"
        os.environ["active"] = "1"
        os.environ["pause"] = "0"
        
        state = self.env.get_state()
        done = False
        
        while not done:
            # Get action
            action = self.agent.act(state)
            
            # Perform action
            self.env.perform_action(action)
            
            # Update game logic for one frame
            self.game._update_gamestuff()
            self.game._update_gamescreen()
            
            # Get new state
            next_state = self.env.get_state()
            done = self.env.is_done()
            
            # Update state
            state = next_state
