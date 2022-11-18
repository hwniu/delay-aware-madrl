class Task_Information_Buffer(object):
    def __init__(self):
        self.task_buffer = []


class Task_Information(object):
    def __init__(self, generation_time, mec_index, ue_index, task_size, computational_density, tolerance_delay, offloading_status):
        self.generation_time = generation_time
        self.mec_index= mec_index
        self.ue_index = ue_index
        self.task_size = task_size
        self.computational_density = computational_density
        self.tolerance_delay = tolerance_delay
        self.offloading_status = offloading_status

    def set_offloading_status(self, status):
        self.offloading_status = status

