import torch
import numpy as np
from tasks.task_1 import task_1

if __name__ == "__main__":

    torch.manual_seed(314)
    np.random.seed(314)

    task_1()