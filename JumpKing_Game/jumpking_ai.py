#!/usr/bin/env python
"""
Enhanced Jump King AI Implementation with Cross-Level Platform Detection and Obstacle Avoidance
Save this as jumpking_ai_enhanced.py in your game folder
"""

import math
import heapq
import pygame
from typing import List, Tuple, Optional, Dict, Set
import random

class JumpTrajectory:
    """Handles jump trajectory calculations with accurate physics and obstacle detection"""
    
    def __init__(self):
        # Physics from your physics.py
        self.gravity_angle = math.pi  # Downward
        self.gravity_force = 0.27
        
        # Jump angles from King.py
        self.jump_angles = {'up': 0, 'left': -math.pi/3, 'right': math.pi/3}
        
        # Physics constants from King.py
        self.max_speed = 11
        self.elasticity = 0.925
        self.angle_elasticity = 0.5
        
    def add_vectors(self, angle1, length1, angle2, length2):
        """Physics vector addition from your physics.py"""
        x = math.sin(angle1) * length1 + math.sin(angle2) * length2
        y = math.cos(angle1) * length1 + math.cos(angle2) * length2
        
        angle = math.pi/2 - math.atan2(y, x)
        length = math.hypot(x, y)
        
        return angle, length
    
    def get_all_relevant_platforms(self, levels, current_level):
        """Get platforms from current level and next level for trajectory calculation"""
        all_platforms = []
        
        # Add current level platforms
        if current_level in levels.levels:
            current_platforms = levels.levels[current_level].platforms
            for platform in current_platforms:
                all_platforms.append({
                    'platform': platform,
                    'level': current_level,
                    'is_current_level': True
                })
        
        # Add next level platforms (for level transitions)
        next_level = current_level + 1
        if next_level in levels.levels:
            next_platforms = levels.levels[next_level].platforms
            for platform in next_platforms:
                # Offset platform positions for next level (assuming next level starts at y=0)
                all_platforms.append({
                    'platform': platform,
                    'level': next_level,
                    'is_current_level': False
                })
        
        return all_platforms
    
    def simulate_jump(self, start_pos, direction, charge_time, all_platforms, max_time=120):
        """
        Simulate a complete jump trajectory including collisions with multi-level platform detection
        Returns: (landing_pos, success, trajectory_points, landing_platform_info, obstacles_hit)
        """
        x, y = start_pos
        
        # Calculate initial jump velocity (from King.py _jump method)
        speed = 1.5 + ((charge_time/5)**1.13)
        
        if direction == "up":
            angle = 0
        else:
            angle = self.jump_angles[direction] * (1 - charge_time / 45.5)
            speed += 0.9
            
        trajectory = [(x, y)]
        obstacles_hit = []
        
        for t in range(max_time):
            # Apply gravity
            angle, speed = self.add_vectors(angle, speed, self.gravity_angle, self.gravity_force)
            
            # Update position
            old_x, old_y = x, y
            x += math.sin(angle) * speed
            y -= math.cos(angle) * speed
            
            # Clamp speed
            if speed > self.max_speed:
                speed = self.max_speed
                
            trajectory.append((x, y))
            
            # CHECK FOR LEVEL TRANSITION FIRST (before platform collision)
            if y < -50:  # Reached well above screen = level transition!
                return (x, y), True, trajectory, {
                    'type': 'LEVEL_TRANSITION',
                    'level': 'next',
                    'platform': None,
                    'is_current_level': False
                }, obstacles_hit
            
            # Check collision with all platforms (current and next level)
            collision_result = self._check_trajectory_collision(
                (old_x, old_y), (x, y), all_platforms
            )
            
            if collision_result:
                platform_info, collision_point, is_obstacle = collision_result
                
                if is_obstacle:
                    # Hit an obstacle (side/bottom collision)
                    obstacles_hit.append(platform_info)
                    # Continue trajectory with bounced/modified path
                    # For now, just return as failed
                    return collision_point, False, trajectory, platform_info, obstacles_hit
                else:
                    # Successful landing on top of platform
                    return collision_point, True, trajectory, platform_info, obstacles_hit
                
            # Check bounds (but allow negative Y for level transition)
            if x < -50 or x > 530 or y > 600:  # Extended bounds
                return (x, y), False, trajectory, None, obstacles_hit
                
        return (x, y), False, trajectory, None, obstacles_hit
    
    def _check_trajectory_collision(self, old_pos, new_pos, all_platforms):
        """
        Check if trajectory segment collides with any platform
        Returns: (platform_info, collision_point, is_obstacle) or None
        """
        old_x, old_y = old_pos
        new_x, new_y = new_pos
        
        # King's collision box (from King.py)
        king_width, king_height = 20, 24  # rect_width, rect_height
        
        for platform_info in all_platforms:
            platform = platform_info['platform']
            rect = platform.rect
            
            # Adjust rect position if it's from next level
            if not platform_info['is_current_level']:
                # Assuming next level platforms are positioned relative to their level
                # You might need to adjust this based on your level system
                adjusted_rect = pygame.Rect(rect.x, rect.y - 600, rect.width, rect.height)  # Assuming 600px per level
            else:
                adjusted_rect = rect
            
            # King's bounding box at new position
            king_left = new_x
            king_right = new_x + king_width
            king_top = new_y
            king_bottom = new_y + king_height
            
            # King's bounding box at old position
            king_bottom_old = old_y + king_height
            king_bottom_new = new_y + king_height
            
            # Check for TOP collision (successful landing)
            if (king_bottom_old <= adjusted_rect.top and 
                king_bottom_new >= adjusted_rect.top and
                king_left < adjusted_rect.right and 
                king_right > adjusted_rect.left):
                
                # Land on top of platform
                landing_y = adjusted_rect.top - king_height
                return platform_info, (new_x, landing_y), False
            
            # Check for SIDE/BOTTOM collision (obstacle)
            if (king_left < adjusted_rect.right and 
                king_right > adjusted_rect.left and
                king_top < adjusted_rect.bottom and 
                king_bottom > adjusted_rect.top):
                
                # Hit side or bottom of platform (obstacle)
                return platform_info, (new_x, new_y), True
                
        return None
    
    def get_reachable_platforms(self, start_pos, all_platforms, max_charge=30):
        """Get all platforms reachable from start position with obstacle avoidance"""
        reachable = []
        
        for direction in ['up', 'left', 'right']:
            for charge in range(1, max_charge + 1, 2):  # Skip some charges for performance
                landing_pos, success, trajectory, platform_info, obstacles = self.simulate_jump(
                    start_pos, direction, charge, all_platforms
                )
                
                if success and platform_info:
                    # Check if this is a level transition
                    if platform_info.get('type') == 'LEVEL_TRANSITION':
                        reachable.append({
                            'direction': direction,
                            'charge': charge,
                            'landing': landing_pos,
                            'platform_info': platform_info,
                            'trajectory': trajectory,
                            'is_level_transition': True,
                            'obstacles_hit': obstacles,
                            'quality_score': self._calculate_jump_quality(start_pos, landing_pos, obstacles, True)
                        })
                    else:
                        # Regular platform landing
                        # Only consider jumps that don't hit too many obstacles
                        if len(obstacles) <= 2:  # Allow some obstacle tolerance
                            quality_score = self._calculate_jump_quality(start_pos, landing_pos, obstacles, False)
                            
                            reachable.append({
                                'direction': direction,
                                'charge': charge,
                                'landing': landing_pos,
                                'platform_info': platform_info,
                                'trajectory': trajectory,
                                'is_level_transition': False,
                                'obstacles_hit': obstacles,
                                'quality_score': quality_score
                            })
        
        return reachable
    
    def _calculate_jump_quality(self, start_pos, landing_pos, obstacles, is_level_transition):
        """Calculate quality score for a jump considering obstacles and position"""
        if is_level_transition:
            return 10000  # Always prioritize level transitions
        
        # Base score from vertical progress
        vertical_progress = start_pos[1] - landing_pos[1]  # Positive is upward
        score = vertical_progress * 10
        
        # Penalty for obstacles
        obstacle_penalty = len(obstacles) * 50
        score -= obstacle_penalty
        
        # Penalty for excessive horizontal movement
        horizontal_distance = abs(landing_pos[0] - start_pos[0])
        if horizontal_distance > 200:
            score -= horizontal_distance * 0.5
        
        # Bonus for staying near screen center (avoid edges)
        center_x = 240  # Screen center
        distance_from_center = abs(landing_pos[0] - center_x)
        if distance_from_center < 100:
            score += 20
        elif distance_from_center > 200:
            score -= 30
        
        return score

class JumpKingAI:
    """Enhanced AI controller for Jump King with cross-level awareness"""
    
    def __init__(self, king, levels):
        self.king = king
        self.levels = levels
        self.trajectory_calc = JumpTrajectory()
        
        # Current plan
        self.current_plan = []
        self.plan_index = 0
        self.current_jump = None
        self.charge_target = 0
        
        # State tracking
        self.last_position = None
        self.stuck_counter = 0
        self.last_level = 0
        self.level_transition_attempts = 0
        self.failed_jump_positions = set()  # Track positions where jumps failed
        
        # Performance tracking
        self.successful_jumps = 0
        self.failed_jumps = 0
        self.level_completion_time = {}
        
    def get_next_action(self):
        """Main AI decision making function"""
        current_pos = (self.king.rect_x, self.king.rect_y)
        
        # Check if we've changed levels
        if self.levels.current_level != self.last_level:
            print(f"AI: Successfully reached level {self.levels.current_level}!")
            self.current_plan = []
            self.last_level = self.levels.current_level
            self.level_transition_attempts = 0
            self.stuck_counter = 0
            self.failed_jump_positions.clear()  # Reset failed positions
        
        # Check if we're stuck or in a failed position
        if self.last_position and self._distance(current_pos, self.last_position) < 5:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        # Check if we're in a previously failed position
        current_pos_rounded = (round(current_pos[0] / 10) * 10, round(current_pos[1] / 10) * 10)
        if current_pos_rounded in self.failed_jump_positions:
            self.stuck_counter += 30  # Accelerate replanning
            
        self.last_position = current_pos
        
        # If stuck for too long, replan
        if self.stuck_counter > 120:  # 2 seconds at 60fps
            print("AI: Stuck! Replanning...")
            # Mark this position as problematic
            self.failed_jump_positions.add(current_pos_rounded)
            self.current_plan = []
            self.stuck_counter = 0
            self.level_transition_attempts += 1
        
        # If we don't have a plan or finished current plan, make new one
        if not self.current_plan or self.plan_index >= len(self.current_plan):
            if not self._create_plan():
                return self._emergency_action()
        
        # Execute current plan
        return self._execute_plan()
    
    def _create_plan(self):
        """Create a new movement plan with cross-level platform awareness"""
        current_level = self.levels.current_level
        current_pos = (self.king.rect_x, self.king.rect_y)
        
        # Get all relevant platforms (current and next level)
        all_platforms = self.trajectory_calc.get_all_relevant_platforms(self.levels, current_level)
        
        if not all_platforms:
            print("AI: No platforms found!")
            return False
        
        # Find reachable platforms with obstacle awareness
        reachable = self.trajectory_calc.get_reachable_platforms(current_pos, all_platforms)
        
        if not reachable:
            print("AI: No reachable platforms found!")
            return False
        
        # Filter out jumps to previously failed positions
        filtered_reachable = []
        for jump in reachable:
            landing_pos_rounded = (round(jump['landing'][0] / 10) * 10, round(jump['landing'][1] / 10) * 10)
            if landing_pos_rounded not in self.failed_jump_positions:
                filtered_reachable.append(jump)
        
        if not filtered_reachable:
            print("AI: All reachable positions previously failed, trying anyway...")
            filtered_reachable = reachable
        
        # Sort by quality score (already calculated in get_reachable_platforms)
        filtered_reachable.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Enhanced selection logic
        best_jumps = []
        for jump in filtered_reachable[:5]:  # Consider top 5 jumps
            # Additional scoring based on current situation
            situational_score = jump['quality_score']
            
            # If we're near the top of the screen, strongly prefer level transitions
            if current_pos[1] < 100:
                if jump.get('is_level_transition', False):
                    situational_score += 5000
                else:
                    situational_score -= 1000  # Penalty for not transitioning when near top
            
            # If we've been trying to transition and failing, prefer safer intermediate jumps
            if self.level_transition_attempts > 5:
                if not jump.get('is_level_transition', False):
                    if jump['landing'][1] < current_pos[1]:  # Upward movement
                        situational_score += 500
            
            # Prefer jumps with fewer obstacles
            obstacle_bonus = max(0, 3 - len(jump['obstacles_hit'])) * 100
            situational_score += obstacle_bonus
            
            best_jumps.append((situational_score, jump))
        
        if best_jumps:
            # Sort by situational score
            best_jumps.sort(key=lambda x: x[0], reverse=True)
            best_jump = best_jumps[0][1]
            
            self.current_plan = [best_jump]
            self.plan_index = 0
            
            jump_type = "LEVEL TRANSITION" if best_jump.get('is_level_transition', False) else "platform"
            obstacles_info = f" (obstacles: {len(best_jump['obstacles_hit'])})" if best_jump['obstacles_hit'] else ""
            print(f"AI: Planning {jump_type} jump {best_jump['direction']} with charge {best_jump['charge']} to {best_jump['landing']}{obstacles_info}")
            return True
            
        return False
    
    def _execute_plan(self):
        """Execute the current plan step"""
        if not self.current_plan or self.plan_index >= len(self.current_plan):
            return "Freeze"
            
        current_jump = self.current_plan[self.plan_index]
        
        # If king is falling, wait
        if self.king.isFalling:
            return "Freeze"
            
        # If king is splatted, wait
        if self.king.isSplat and self.king.splatCount <= self.king.splatDuration:
            return "Freeze"
        
        # Start crouching to charge
        if not self.king.isCrouch:
            self.charge_target = current_jump['charge']
            return "Crouch"
        
        # Continue charging until we reach target
        if self.king.jumpCount < self.charge_target:
            return "Crouch"
        
        # Execute jump
        direction = current_jump['direction']
        self.plan_index += 1  # Move to next action
        
        print(f"AI: Executing {direction} jump with charge {self.charge_target}")
        
        if direction == 'up':
            return "Jump"
        elif direction == 'left':
            return "JumpLeft"
        elif direction == 'right':
            return "JumpRight"
            
        return "Freeze"
    
    def _emergency_action(self):
        """Enhanced emergency action when AI is confused"""
        current_pos = (self.king.rect_x, self.king.rect_y)
        
        # If we're near the top of the screen, try upward jumps
        if current_pos[1] < 100:
            print("AI: Near top of screen, trying upward jump!")
            if not self.king.isCrouch:
                return "Crouch"
            elif self.king.jumpCount < 25:  # Strong charge for level transition
                return "Crouch"
            else:
                return "Jump"
        
        # If we're stuck in the middle, try different approaches
        if current_pos[1] > 300:
            # Lower on screen, try to get higher first
            actions = ["Jump", "JumpLeft", "JumpRight"]
            weights = [0.5, 0.25, 0.25]  # Prefer upward jumps
        else:
            # Higher on screen, try lateral movement
            actions = ["JumpLeft", "JumpRight", "Jump"]
            weights = [0.4, 0.4, 0.2]
        
        if not self.king.isFalling and not self.king.isSplat:
            # Weighted random selection
            return random.choices(actions, weights=weights)[0]
        
        return "Freeze"
    
    def _distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def reset(self):
        """Reset AI state"""
        self.current_plan = []
        self.plan_index = 0
        self.stuck_counter = 0
        self.last_position = None
        self.level_transition_attempts = 0
        self.failed_jump_positions.clear()
        
    def get_stats(self):
        """Get AI performance statistics"""
        return {
            'successful_jumps': self.successful_jumps,
            'failed_jumps': self.failed_jumps,
            'current_level': self.levels.current_level,
            'plan_length': len(self.current_plan),
            'stuck_counter': self.stuck_counter,
            'level_transition_attempts': self.level_transition_attempts,
            'failed_positions': len(self.failed_jump_positions)
        }

# Integration functions (same as before but with enhanced AI)
def integrate_ai_with_game(king, levels):
    """
    Integration function to add enhanced AI to the existing game
    Call this after creating your King and Levels objects
    """
    ai = JumpKingAI(king, levels)
    
    # Store original method
    original_robot_check_events = getattr(king, '_robot_check_events', None)
    
    def ai_robot_check_events(command=None):
        """Enhanced robot_check_events that uses AI when no command given"""
        
        if command is None:
            # Get AI decision
            command = ai.get_next_action()
            
        # Execute command using existing logic
        if command == "Crouch":
            king.jumpCount += 1
            king.isCrouch = True
        elif command == "Jump":
            king._jump("up")
        elif command == "JumpLeft":
            king._jump("left")
        elif command == "JumpRight":
            king._jump("right")
        elif command == "Freeze":
            king.angle, king.speed = 0, 0
            king.angle, king.speed = king.physics.add_vectors(
                king.angle, king.speed, 
                -king.physics.gravity[0], -king.physics.gravity[1]
            )
        elif command == "WalkLeft":
            king._walk("left")
        elif command == "WalkRight":
            king._walk("right")
        else:
            king.isWalk = False
    
    # Replace the method
    king._robot_check_events = ai_robot_check_events
    
    return ai

def create_ai_controlled_king(king, levels):
    """
    Simple function to make the king AI-controlled with enhanced capabilities
    """
    ai = integrate_ai_with_game(king, levels)
    
    # Override the normal _check_events with AI control
    def ai_check_events():
        """Replace normal input with AI decisions"""
        if not king.isFalling and not king.levels.ending:
            command = ai.get_next_action()
            king._robot_check_events(command)
    
    # You can manually call this or integrate it into your game loop
    king._ai_check_events = ai_check_events
    king._ai = ai  # Store reference for debugging
    
    return ai