#!/usr/bin/env python
"""
Adaptive Jump King AI with Wall Bounce and Horizontal-Only Jumps
+ Monitoring metryk skoków
"""

import math
import pygame
from typing import List, Tuple, Optional, Dict, Set
import random
import time

# Dodaj import monitora
from monitor import AIMonitor

class JumpPhysicsCalculator:
    """Accurate physics calculator with wall-bounce support"""
    def __init__(self, king=None, levels=None):
        self.king = king
        self.levels = levels
        self.gravity_force = 0.27
        self.gravity_angle = math.pi
        self.max_speed = 11.0
        self.elasticity = 0.925
        self.base_jump_speed = 1.5
        self.charge_power = 1.11
        self.charge_divisor = 6.0
        self.directional_bonus = 0.90
        self.angle_modifier = 45.5
        self.king_width = 24
        self.king_height = 28

    def add_vectors(self, angle1, length1, angle2, length2):
        x = math.sin(angle1)*length1 + math.sin(angle2)*length2
        y = math.cos(angle1)*length1 + math.cos(angle2)*length2
        angle = math.pi/2 - math.atan2(y, x)
        length = math.hypot(x, y)
        return angle, length

    def calculate_jump_trajectory(self, start_pos, direction, charge_time, max_steps=150):
        x, y = float(start_pos[0]), float(start_pos[1])
        speed = self.base_jump_speed + ((charge_time/self.charge_divisor)**self.charge_power)
        base_angle = -math.pi/3 if direction == 'left' else math.pi/3
        angle = base_angle * (1 - min(charge_time/self.angle_modifier, 1))
        speed += self.directional_bonus
        trajectory = [(x, y)]
        for step in range(max_steps):
            angle, speed = self.add_vectors(angle, speed, self.gravity_angle, self.gravity_force)
            speed = min(speed, self.max_speed)
            x += math.sin(angle)*speed
            y -= math.cos(angle)*speed
            trajectory.append((x, y))
            if y > 700 or x < -100 or x > 600:
                break
        return trajectory

    def check_platform_collision(self, trajectory, platforms,
                                 king_width=20, king_height=24):
        collisions = []
        for i in range(1, len(trajectory)):
            prev_x, prev_y = trajectory[i-1]
            curr_x, curr_y = trajectory[i]
            king_rect = pygame.Rect(curr_x, curr_y, king_width, king_height)
            prev_rect = pygame.Rect(prev_x, prev_y, king_width, king_height)
            for platform in platforms:
                plat = platform.rect
                # Landing on top
                if (prev_rect.bottom <= plat.top and king_rect.bottom >= plat.top
                        and king_rect.left < plat.right and king_rect.right > plat.left):
                    landing_y = plat.top - king_height
                    collisions.append({
                        'type': 'landing',
                        'position': (curr_x, landing_y),
                        'platform': platform,
                        'step': i,
                        'success': True
                    })
                    return collisions
                # Side or bottom collision
                if king_rect.colliderect(plat):
                    if prev_rect.right <= plat.left:
                        collisions.append({
                            'type': 'bounce',
                            'position': (plat.left - king_width, curr_y),
                            'side': 'left',
                            'platform': platform,
                            'step': i,
                            'success': True
                        })
                    elif prev_rect.left >= plat.right:
                        collisions.append({
                            'type': 'bounce',
                            'position': (plat.right, curr_y),
                            'side': 'right',
                            'platform': platform,
                            'step': i,
                            'success': True
                        })
                    else:
                        collisions.append({
                            'type': 'obstacle',
                            'position': (curr_x, curr_y),
                            'platform': platform,
                            'step': i,
                            'success': False
                        })
                    return collisions
        return collisions

class SmartJumpPlanner:
    """Intelligent jump planning (horizontal-only, with bounce)"""
    def __init__(self, physics_calc):
        self.physics = physics_calc
        self.charge_ranges = {
            'short': range(3, 12),
            'medium': range(12, 25),
            'long': range(25, 40),
            'max': range(40, 50)
        }

    def find_best_jumps(self, start_pos, platforms):
        all_jumps = []
        for direction in ['left', 'right']:
            for range_name in ['medium', 'long', 'max']:
                for charge in self.charge_ranges[range_name][::3]:
                    traj = self.physics.calculate_jump_trajectory(start_pos, direction, charge)
                    collisions = self.physics.check_platform_collision(traj, platforms)
                    if not collisions:
                        continue
                    col = collisions[0]
                    if not col['success']:
                        continue
                    landing = col['position']
                    dist = abs(landing[0] - start_pos[0])
                    score = dist - 0.5*(landing[1]-start_pos[1])
                    all_jumps.append({
                        'direction': direction,
                        'charge': charge,
                        'landing_pos': landing,
                        'trajectory': traj,
                        'collision': col,
                        'score': score,
                        'bounce': col['type']=='bounce'
                    })
        all_jumps.sort(key=lambda j: j['score'], reverse=True)
        return all_jumps

class AdaptiveJumpKingAI:
    """Main AI controller with wall-bounce and horizontal-only jumps"""
    def __init__(self, king, levels):
        self.king = king
        self.levels = levels
        self.physics = JumpPhysicsCalculator(king, levels)
        self.planner = SmartJumpPlanner(self.physics)
        # Inicjalizacja monitora
        self.monitor = AIMonitor()
        self.current_plan = []
        self.plan_step = 0
        self.charging = False
        self.target_charge = 0
        self.stuck_counter = 0
        self.position_memory = set()

    def get_action(self):
        pos = (int(self.king.rect_x), int(self.king.rect_y))
        on_ground = self.king.lastCollision is not None
        platforms = []
        lvl = self.levels.current_level
        if lvl in self.levels.levels:
            platforms = self.levels.levels[lvl].platforms
        if not on_ground:
            return 'wait'
        if not self.current_plan or self.plan_step >= len(self.current_plan):
            self.create_plan(pos, platforms)
        return self.current_plan and self.current_plan_action()

    def current_plan_action(self):
        action, direction, charge = self.current_plan[self.plan_step]
        if not self.charging:
            return 'crouch'
        if self.king.jumpCount < self.target_charge:
            return 'crouch'
        return 'jump_' + direction

    def create_plan(self, pos, platforms):
        jumps = self.planner.find_best_jumps(pos, platforms)
        if not jumps:
            direction, charge = random.choice(['left','right']), 20
        else:
            best = jumps[0]
            direction, charge = best['direction'], best['charge']
        self.current_plan = [('jump', direction, charge)]
        self.plan_step = 0
        self.charging = False
        self.target_charge = charge

    def execute_plan(self):
        act = self.get_action()
        if act == 'wait':
            return 'wait'
        if act == 'crouch':
            if not self.charging:
                self.charging = True
                self.king.jumpCount = 0
            return 'crouch'
        if self.king.lastCollision is None:
            return 'wait'
        # Wykonanie skoku
        self.charging = False
        self.plan_step += 1
        _, direction, _ = self.current_plan[self.plan_step-1]
        return 'jump_' + direction

    def update_ai(self):
        action = self.execute_plan()
        if action == 'crouch':
            self.king.isCrouch = True
            self.king.jumpCount += 1
        elif action in ('jump_left', 'jump_right'):
            # Upewnij się, że AI jest na ziemi
            if self.king.lastCollision:
                dir = 'left' if action.endswith('left') else 'right'
                # Logowanie metryki zanim wykonamy skok
                # typ skoku to col['type'], score itd. – można przechować w last_collision_info
                last = self.planner.find_best_jumps((self.king.rect_x, self.king.rect_y), 
                                                     self.levels.levels[self.levels.current_level].platforms)[0]
                self.monitor.log_jump(
                    jump_type = last['collision']['type'],
                    score = last['score'],
                    start_pos = (self.king.rect_x, self.king.rect_y),
                    landing_pos = last['landing_pos'],
                    stuck_counter = self.stuck_counter
                )
                self.king._jump(dir)
                # Reset stuck counter po udanym ruchu
                if last['collision']['type'] == 'landing':
                    self.stuck_counter = 0
                else:
                    self.stuck_counter += 1
        # Inne akcje: 'wait'
        # Tu możesz dodać np. self.monitor.get_stats() do UI

# Integration

def create_ai_controlled_king(king, levels):
    ai = AdaptiveJumpKingAI(king, levels)
    king._ai = ai
    king._ai_check_events = ai.update_ai
    return ai
