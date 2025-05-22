#!/usr/bin/env python
"""
Enhanced Jump King AI with Improved Obstacle Avoidance and Trajectory Planning
"""

import math
import pygame
from typing import List, Tuple, Optional, Dict, Set
import random
import time

class JumpPhysicsCalculator:
    """Accurate physics calculator matching your King.py implementation"""
    
    def __init__(self):
        # Physics constants from your code
        self.gravity_force = 0.27
        self.gravity_angle = math.pi  # Downward
        self.max_speed = 11
        self.elasticity = 0.925
        
        # Jump constants (from King.py analysis)
        self.base_jump_speed = 1.5
        self.charge_power = 1.13
        self.charge_divisor = 5
        self.directional_bonus = 0.9
        self.angle_modifier = 45.5
        
        # King dimensions
        self.king_width = 20
        self.king_height = 24
        
    def add_vectors(self, angle1, length1, angle2, length2):
        """Vector addition matching your physics.py"""
        x = math.sin(angle1) * length1 + math.sin(angle2) * length2
        y = math.cos(angle1) * length1 + math.cos(angle2) * length2
        
        angle = math.pi/2 - math.atan2(y, x)
        length = math.hypot(x, y)
        
        return angle, length
    
    def calculate_jump_trajectory(self, start_pos, direction, charge_time, max_steps=150):
        """Calculate complete jump trajectory with accurate physics"""
        x, y = float(start_pos[0]), float(start_pos[1])
        
        # Calculate initial velocity (matching King.py _jump method)
        speed = self.base_jump_speed + ((charge_time / self.charge_divisor) ** self.charge_power)
        
        if direction == "up":
            angle = 0
        else:
            # Directional jumps
            base_angle = -math.pi/3 if direction == "left" else math.pi/3
            angle = base_angle * (1 - charge_time / self.angle_modifier)
            speed += self.directional_bonus
        
        trajectory = [(x, y)]
        
        for step in range(max_steps):
            # Apply gravity
            angle, speed = self.add_vectors(angle, speed, self.gravity_angle, self.gravity_force)
            
            # Clamp speed
            if speed > self.max_speed:
                speed = self.max_speed
            
            # Update position
            x += math.sin(angle) * speed
            y -= math.cos(angle) * speed
            
            trajectory.append((x, y))
            
            # Stop if we're clearly falling off screen
            if y > 700 or x < -100 or x > 600:
                break
                
        return trajectory
    
    def check_platform_collision(self, trajectory, platforms, king_width=20, king_height=24):
        """Check trajectory against platforms with proper collision detection"""
        collisions = []
        
        for i in range(1, len(trajectory)):
            prev_x, prev_y = trajectory[i-1]
            curr_x, curr_y = trajectory[i]
            
            # King's bounding box
            king_rect = pygame.Rect(curr_x, curr_y, king_width, king_height)
            prev_king_rect = pygame.Rect(prev_x, prev_y, king_width, king_height)
            
            for platform in platforms:
                platform_rect = platform.rect
                
                # Check for landing on top (successful collision)
                if (prev_king_rect.bottom <= platform_rect.top and 
                    king_rect.bottom >= platform_rect.top and
                    king_rect.left < platform_rect.right and 
                    king_rect.right > platform_rect.left):
                    
                    # Successfully landed on platform
                    landing_y = platform_rect.top - king_height
                    collisions.append({
                        'type': 'landing',
                        'position': (curr_x, landing_y),
                        'platform': platform,
                        'step': i,
                        'success': True
                    })
                    return collisions  # Return first successful landing
                
                # Check for obstacle collision (sides/bottom)
                elif (king_rect.colliderect(platform_rect) and 
                      not (prev_king_rect.bottom <= platform_rect.top)):
                    
                    # Hit an obstacle
                    collisions.append({
                        'type': 'obstacle',
                        'position': (curr_x, curr_y),
                        'platform': platform,
                        'step': i,
                        'success': False
                    })
                    return collisions  # Return first obstacle hit
        
        return collisions

class SmartJumpPlanner:
    """Intelligent jump planning with obstacle avoidance"""
    
    def __init__(self, physics_calc):
        self.physics = physics_calc
        self.charge_ranges = {
            'short': range(3, 12),      # Short hops
            'medium': range(12, 25),    # Medium jumps  
            'long': range(25, 40),      # Long jumps
            'max': range(40, 50)        # Maximum power
        }
        
    def check_overhead_clearance(self, start_pos, platforms, clearance_height=100):
        """Check if there's enough vertical clearance above the king"""
        start_x, start_y = start_pos
        
        # Check for platforms directly above
        overhead_obstacles = []
        for platform in platforms:
            rect = platform.rect
            # Check if platform is above and overlapping horizontally
            if (rect.bottom < start_y and  # Platform is above
                rect.bottom > start_y - clearance_height and  # Within clearance zone
                rect.left < start_x + self.physics.king_width and  # Overlaps horizontally
                rect.right > start_x):
                overhead_obstacles.append(platform)
        
        return len(overhead_obstacles) == 0, overhead_obstacles
        
    def find_best_jumps(self, start_pos, platforms, target_direction='up'):
        """Find the best possible jumps from current position with overhead obstacle awareness"""
        all_jumps = []
        
        # Check overhead clearance first
        has_clearance, overhead_obstacles = self.check_overhead_clearance(start_pos, platforms)
        
        # If there are overhead obstacles, prioritize horizontal movement
        if not has_clearance:
            print(f"AI: Overhead obstacles detected, prioritizing horizontal movement")
            direction_priorities = ['left', 'right', 'up']
            # Use lighter charges for horizontal movement to avoid hitting ceiling
            preferred_ranges = ['short', 'medium']
        else:
            # Normal prioritization
            direction_priorities = ['up', 'left', 'right']
            preferred_ranges = ['short', 'medium', 'long']
        
        for direction in direction_priorities:
            # Choose appropriate charge ranges based on situation
            if direction == 'up':
                if not has_clearance:
                    # Very light upward jumps only if overhead blocked
                    charge_ranges = ['short']
                else:
                    charge_ranges = preferred_ranges
            else:
                # Horizontal jumps - use medium to long charges
                if not has_clearance:
                    # Prefer medium charges to clear obstacles without hitting ceiling
                    charge_ranges = ['medium', 'long']
                else:
                    charge_ranges = ['medium', 'long']
            
            for range_name in charge_ranges:
                step_size = 2 if not has_clearance and direction != 'up' else 3
                for charge in self.charge_ranges[range_name][::step_size]:
                    jump_info = self.evaluate_jump(start_pos, direction, charge, platforms, overhead_obstacles)
                    if jump_info:
                        all_jumps.append(jump_info)
        
        # Sort by quality score
        all_jumps.sort(key=lambda x: x['score'], reverse=True)
        return all_jumps
    
    def evaluate_jump(self, start_pos, direction, charge, platforms, overhead_obstacles=None):
        """Evaluate a single jump option with overhead obstacle awareness"""
        trajectory = self.physics.calculate_jump_trajectory(start_pos, direction, charge)
        
        # Check for overhead collisions first
        if overhead_obstacles and direction == 'up':
            # Check if trajectory hits overhead obstacles
            for point in trajectory[:len(trajectory)//3]:  # Check early part of trajectory
                king_rect = pygame.Rect(point[0], point[1], self.physics.king_width, self.physics.king_height)
                for obstacle in overhead_obstacles:
                    if king_rect.colliderect(obstacle.rect):
                        # Will hit overhead obstacle
                        return None
        
        # Check platform collisions
        collisions = self.physics.check_platform_collision(trajectory, platforms)
        
        if not collisions:
            # No collision = falling off screen
            return None
        
        collision = collisions[0]
        if not collision['success']:
            # Hit obstacle
            return None
        
        # Calculate quality score with overhead awareness
        landing_pos = collision['position']
        score = self.calculate_jump_score(start_pos, landing_pos, direction, charge, trajectory, overhead_obstacles)
        
        return {
            'direction': direction,
            'charge': charge,
            'landing_pos': landing_pos,
            'trajectory': trajectory,
            'collision': collision,
            'score': score,
            'is_safe': True,
            'avoids_overhead': overhead_obstacles is not None and direction != 'up'
        }
    
    def calculate_jump_score(self, start_pos, landing_pos, direction, charge, trajectory, overhead_obstacles=None):
        """Calculate quality score for a jump with overhead obstacle considerations"""
        start_x, start_y = start_pos
        land_x, land_y = landing_pos
        
        # Base score: vertical progress (higher is better)
        vertical_progress = start_y - land_y
        score = vertical_progress * 10
        
        # Special handling for overhead obstacles
        if overhead_obstacles:
            if direction in ['left', 'right']:
                # Horizontal movement is valuable when overhead is blocked
                horizontal_distance = abs(land_x - start_x)
                
                # Bonus for meaningful horizontal movement
                if horizontal_distance > 50:
                    score += 200  # Big bonus for getting out from under obstacles
                
                # Additional bonus if we're moving away from screen center (exploring)
                screen_center_x = 240
                if abs(land_x - screen_center_x) > abs(start_x - screen_center_x):
                    score += 100  # Bonus for exploring away from center
                
                # Bonus for moderate vertical progress even with horizontal movement
                if vertical_progress > -20:  # Not falling too much
                    score += 50
                    
            elif direction == 'up':
                # Penalize upward jumps when overhead is blocked (unless very short)
                if charge > 10:
                    score -= 500  # Heavy penalty for risky upward jumps
        else:
            # Normal scoring when no overhead obstacles
            
            # Bonus for upward movement
            if vertical_progress > 0:
                score += 100
            
            # Penalty for excessive horizontal movement (when not needed)
            horizontal_distance = abs(land_x - start_x)
            if horizontal_distance > 150:
                score -= horizontal_distance * 0.3
        
        # Bonus for staying on screen
        if 50 < land_x < 430:  # Well within screen bounds
            score += 30
        elif land_x < 20 or land_x > 460:  # Near edges
            score -= 100
        
        # Charge efficiency considerations
        if direction in ['left', 'right']:
            # For horizontal jumps, prefer medium charges
            if 15 <= charge <= 30:
                score += 30
            elif charge < 10:
                score -= 50  # Too weak for meaningful horizontal movement
        else:
            # For upward jumps, prefer various charges based on situation
            if 10 <= charge <= 35:
                score += 20
            elif charge > 45:
                score -= 20  # Very high jumps are risky
        
        # Trajectory safety - check for reasonable arc
        if len(trajectory) > 5:
            max_height = min(y for x, y in trajectory[:len(trajectory)//2])
            height_gained = start_y - max_height
            
            # Reasonable jump arc bonus
            if 20 < height_gained < 150:
                score += 25
            elif height_gained > 200:  # Very high jump
                if land_y < 50:  # Might be level transition
                    score += 800
                else:
                    score -= 100  # Too high for normal platform
        
        # Platform positioning bonus
        if 100 < land_y < 500:  # Good middle range of screen
            score += 20
        
        return score

class AdaptiveJumpKingAI:
    """Main AI controller with learning and adaptation"""
    
    def __init__(self, king, levels):
        self.king = king
        self.levels = levels
        self.physics = JumpPhysicsCalculator()
        self.planner = SmartJumpPlanner(self.physics)
        
        # State management
        self.current_plan = []
        self.plan_step = 0
        self.charging = False
        self.target_charge = 0
        
        # Learning system
        self.position_memory = {}  # Track what works from each position
        self.failed_positions = set()
        self.last_position = None
        self.stuck_counter = 0
        self.position_attempts = {}
        
        # Performance tracking
        self.last_level = 0
        self.jump_history = []
        self.level_start_time = time.time()
        
        # Adaptive parameters
        self.exploration_rate = 0.1  # Chance to try suboptimal jumps
        self.patience_threshold = 180  # Frames to wait before replanning
        
    def get_action(self):
        """Main AI decision function"""
        current_pos = (int(self.king.rect_x), int(self.king.rect_y))
        
        # Check for level change
        if self.levels.current_level != self.last_level:
            self.on_level_change()
        
        # Update stuck detection
        self.update_stuck_detection(current_pos)
        
        # If we're falling or splatted, wait
        if self.king.isFalling or (self.king.isSplat and self.king.splatCount <= self.king.splatDuration):
            return "wait"
        
        # Check if we need to replan
        if self.should_replan():
            self.create_new_plan(current_pos)
        
        # Execute current plan
        return self.execute_current_plan()
    
    def create_new_plan(self, current_pos):
        """Create a new jump plan from current position with overhead obstacle awareness"""
        # Get current level platforms
        platforms = []
        if self.levels.current_level in self.levels.levels:
            platforms = self.levels.levels[self.levels.current_level].platforms
        
        if not platforms:
            self.current_plan = [('emergency', 'up', 20)]
            self.plan_step = 0
            return
        
        # Check for overhead obstacles
        has_clearance, overhead_obstacles = self.planner.check_overhead_clearance(current_pos, platforms)
        
        # Find best jumps with overhead awareness
        possible_jumps = self.planner.find_best_jumps(current_pos, platforms)
        
        if not possible_jumps:
            print("AI: No valid jumps found, using emergency action")
            direction = 'left' if not has_clearance else 'up'
            self.current_plan = [('emergency', direction, 25)]
            self.plan_step = 0
            return
        
        # Filter and select jumps with overhead considerations
        pos_key = self.position_key(current_pos)
        filtered_jumps = []
        
        for jump in possible_jumps[:15]:  # Consider more jumps when dealing with obstacles
            landing_key = self.position_key(jump['landing_pos'])
            
            # Skip if we've failed from this landing position multiple times
            if landing_key in self.failed_positions:
                continue
                
            # Skip if we've tried this exact jump too many times
            jump_signature = (pos_key, jump['direction'], jump['charge'])
            attempts = self.position_attempts.get(jump_signature, 0)
            max_attempts = 2 if not has_clearance else 3  # Be more flexible with overhead obstacles
            if attempts > max_attempts:
                continue
                
            filtered_jumps.append(jump)
        
        if not filtered_jumps:
            # All jumps have been tried, try the best ones anyway
            filtered_jumps = possible_jumps[:5]
        
        # Enhanced selection logic for overhead obstacles
        if not has_clearance:
            # Prioritize horizontal jumps when overhead is blocked
            horizontal_jumps = [j for j in filtered_jumps if j['direction'] in ['left', 'right']]
            if horizontal_jumps:
                print("AI: Overhead blocked, selecting horizontal jump")
                selected_jump = horizontal_jumps[0]  # Best horizontal jump
            else:
                # No horizontal jumps available, try very light upward jump
                light_jumps = [j for j in filtered_jumps if j['direction'] == 'up' and j['charge'] < 15]
                if light_jumps:
                    selected_jump = light_jumps[0]
                else:
                    selected_jump = filtered_jumps[0] if filtered_jumps else possible_jumps[0]
        else:
            # Normal selection when no overhead obstacles
            if random.random() < self.exploration_rate and len(filtered_jumps) > 1:
                # Occasionally try a suboptimal jump for exploration
                selected_jump = random.choice(filtered_jumps[1:min(4, len(filtered_jumps))])
            else:
                # Usually pick the best jump
                selected_jump = filtered_jumps[0]
        
        # Record attempt
        jump_signature = (pos_key, selected_jump['direction'], selected_jump['charge'])
        self.position_attempts[jump_signature] = self.position_attempts.get(jump_signature, 0) + 1
        
        # Create plan
        self.current_plan = [('jump', selected_jump['direction'], selected_jump['charge'])]
        self.plan_step = 0
        self.charging = False
        self.target_charge = 0
        
        obstacle_info = " (avoiding overhead)" if not has_clearance else ""
        print(f"AI: Planning {selected_jump['direction']} jump with charge {selected_jump['charge']} (score: {selected_jump['score']:.1f}){obstacle_info}")
    
    def execute_current_plan(self):
        """Execute the current plan step by step"""
        if not self.current_plan or self.plan_step >= len(self.current_plan):
            return "wait"
        
        action_type, direction, charge = self.current_plan[self.plan_step]
        
        if action_type == 'emergency':
            return self.emergency_action(direction)
        
        # Normal jump execution
        if not self.charging:
            # Start charging
            self.charging = True
            self.target_charge = charge
            self.king.jumpCount = 0  # Reset charge counter
            return "crouch"
        
        # Continue charging
        if self.king.jumpCount < self.target_charge:
            return "crouch"
        
        # Execute jump
        self.charging = False
        self.plan_step += 1
        
        if direction == 'up':
            return "jump"
        elif direction == 'left':
            return "jump_left"
        elif direction == 'right':
            return "jump_right"
        
        return "wait"
    
    def emergency_action(self, preferred_direction='up'):
        """Emergency action when planning fails"""
        if not self.king.isCrouch:
            return "crouch"
        elif self.king.jumpCount < 20:
            return "crouch"
        else:
            if preferred_direction == 'up':
                return "jump"
            elif preferred_direction == 'left':
                return "jump_left"
            elif preferred_direction == 'right':
                return "jump_right"
            else:
                return "jump"
    
    def should_replan(self):
        """Check if we need to create a new plan"""
        return (not self.current_plan or 
                self.plan_step >= len(self.current_plan) or
                self.stuck_counter > self.patience_threshold)
    
    def update_stuck_detection(self, current_pos):
        """Update stuck detection system"""
        if self.last_position:
            distance = math.sqrt((current_pos[0] - self.last_position[0])**2 + 
                               (current_pos[1] - self.last_position[1])**2)
            
            if distance < 10:  # Barely moved
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 2)
        
        self.last_position = current_pos
        
        # If stuck for too long, mark position as problematic
        if self.stuck_counter > self.patience_threshold * 2:
            pos_key = self.position_key(current_pos)
            self.failed_positions.add(pos_key)
            self.stuck_counter = 0
            print(f"AI: Marking position {pos_key} as problematic")
    
    def on_level_change(self):
        """Handle level change events"""
        print(f"AI: Advanced to level {self.levels.current_level}!")
        self.last_level = self.levels.current_level
        self.current_plan = []
        self.stuck_counter = 0
        self.failed_positions.clear()  # Reset for new level
        self.position_attempts.clear()
        
        # Record level completion time
        current_time = time.time()
        level_time = current_time - self.level_start_time
        print(f"AI: Level completed in {level_time:.1f} seconds")
        self.level_start_time = current_time
    
    def position_key(self, pos):
        """Convert position to a key for memory storage"""
        return (round(pos[0] / 20) * 20, round(pos[1] / 20) * 20)
    
    def get_debug_info(self):
        """Get debugging information"""
        return {
            'current_level': self.levels.current_level,
            'plan_length': len(self.current_plan),
            'plan_step': self.plan_step,
            'stuck_counter': self.stuck_counter,
            'charging': self.charging,
            'target_charge': self.target_charge,
            'failed_positions': len(self.failed_positions),
            'position_attempts': len(self.position_attempts)
        }

# Integration function
def create_ai_controlled_king(king, levels):
    """Create and integrate the enhanced AI with the king"""
    ai = AdaptiveJumpKingAI(king, levels)
    
    def ai_check_events():
        """AI-controlled event checking"""
        action = ai.get_action()
        
        if action == "crouch":
            king.jumpCount += 1
            king.isCrouch = True
        elif action == "jump":
            king._jump("up")
        elif action == "jump_left":
            king._jump("left")
        elif action == "jump_right":
            king._jump("right")
        elif action == "wait":
            pass  # Do nothing
        
        # Debug output occasionally
        if king.jumpCount % 300 == 0:  # Every 5 seconds at 60fps
            debug_info = ai.get_debug_info()
            print(f"AI Debug: Level {debug_info['current_level']}, Stuck: {debug_info['stuck_counter']}, Failed pos: {debug_info['failed_positions']}")
    
    # Replace the AI check events method
    king._ai_check_events = ai_check_events
    king._ai = ai  # Store reference for debugging
    
    return ai