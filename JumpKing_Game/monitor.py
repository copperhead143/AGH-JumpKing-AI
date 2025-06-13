import logging
from logging.handlers import RotatingFileHandler
import json
import time
from datetime import datetime

logger = logging.getLogger("AIMonitor")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("ai.log", maxBytes=1000000, backupCount=3)
logger.addHandler(handler)

class AIMonitor:
    def __init__(self):
        self.jump_data = []
        self.total_jumps = 0
        self.successful_landings = 0
        self.bounces = 0
        self.last_action_time = time.time()

    def log_jump(self, jump_type, score, start_pos, landing_pos, stuck_counter):
        self.total_jumps += 1
        if jump_type == "landing":
            self.successful_landings += 1
        elif jump_type == "bounce":
            self.bounces += 1

        self.jump_data.append(score)
        self.last_action_time = time.time()

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": jump_type,
            "score": score,
            "start_pos": start_pos,
            "landing_pos": landing_pos,
            "stuck_counter": stuck_counter
        }

        logger.info(json.dumps(log_entry))

        if self.total_jumps % 10 == 0:
            self.print_summary()

        if stuck_counter > 5:
            self.send_alert(stuck_counter)

    def print_summary(self):
        avg_score = sum(self.jump_data) / len(self.jump_data) if self.jump_data else 0
        landing_rate = (self.successful_landings / self.total_jumps) * 100 if self.total_jumps else 0
        print(f"\n--- AI MONITOR SUMMARY ---")
        print(f"Total jumps: {self.total_jumps}")
        print(f"Average score: {avg_score:.2f}")
        print(f"Landing success rate: {landing_rate:.1f}%")
        print(f"Total bounces: {self.bounces}")
        print(f"--------------------------\n")

    def get_stats(self):
        avg_score = sum(self.jump_data) / len(self.jump_data) if self.jump_data else 0
        landing_rate = (self.successful_landings / self.total_jumps) * 100 if self.total_jumps else 0
        return {
            "total_jumps": self.total_jumps,
            "average_score": avg_score,
            "landing_rate": landing_rate,
            "bounces": self.bounces
        }

    def send_alert(self, stuck_counter):
        try:
            from termcolor import cprint
            cprint(f"[ALERT] AI stuck for {stuck_counter} steps!", "red", attrs=["bold"])
        except ImportError:
            print(f"\033[91m[ALERT] AI stuck for {stuck_counter} steps!\033[0m")
