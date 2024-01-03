import torch
import numpy as np
from tasks import tasks

if __name__ == "__main__":

    torch.manual_seed(314)
    np.random.seed(314)

    tasks.task_1(EPOCHS=500)
    tasks.task_2(EPOCHS=500)