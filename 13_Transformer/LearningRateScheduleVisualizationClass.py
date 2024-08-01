import matplotlib.pyplot as plt
import numpy as np

class LearningRateScheduleVisualization:
    def __init__(self, d_model, warmup_steps):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lrates = self._calculate_lrates()
    
    def _calculate_lrates(self):
        lrates = []
        for step_num in range(1, 50000):
            lrate = (np.power(self.d_model, -0.5)) * np.min(
                [np.power(step_num, -0.5), step_num * np.power(self.warmup_steps, -1.5)])
            lrates.append(lrate)
        return lrates
    
    def plot_lrates(self):
        plt.figure(figsize=(6, 3))
        plt.plot(self.lrates)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step Number')
        plt.ylabel('Learning Rate')
        plt.show()

# Usage example
d_model = 512
warmup_steps = 4000

lr_schedule_vis = LearningRateScheduleVisualization(d_model, warmup_steps)
lr_schedule_vis.plot_lrates()
