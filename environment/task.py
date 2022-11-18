import random

from environment.env_config import MAX_TASK_SIZE
from environment.env_config import MIN_TASK_SIZE
from environment.env_config import MAX_MAX_TASK_TOLERANCE_DELAY
from environment.env_config import MIN_MAX_TASK_TOLERANCE_DELAY
from environment.env_config import TASK_COMPUTING_DENSITY

class Task(object):
    def __init__(self, time, min_task_size=MIN_TASK_SIZE, max_task_size=MAX_TASK_SIZE, min_tolerance_delay=MIN_MAX_TASK_TOLERANCE_DELAY,
                 max_tolerance_delay=MAX_MAX_TASK_TOLERANCE_DELAY, task_computing_density=TASK_COMPUTING_DENSITY):
        self.min_task_size = min_task_size
        self.max_task_size = max_task_size
        self.min_tolerance_delay = min_tolerance_delay
        self.max_tolerance_delay = max_tolerance_delay

        self.task_computing_density = task_computing_density

        self.time = time
        self.task_size = random.randint(self.min_task_size, self.max_task_size)
        self.tolerance_delay = random.randint(self.min_tolerance_delay, self.max_tolerance_delay)
        self.require_computing = self.task_size * self.task_computing_density

    def reset(self, time):
        self.time = time
        self.task_size = random.randint(self.min_task_size, self.max_task_size)
        self.tolerance_delay = random.randint(self.min_tolerance_delay, self.max_tolerance_delay)
        self.require_computing = self.task_size * self.task_computing_density