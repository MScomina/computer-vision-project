import torch
import numpy as np
from tasks import tasks
import random

if __name__ == "__main__":

    torch.manual_seed(314)
    np.random.seed(314)
    random.seed(314)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tasks.task_1(MAX_EPOCHS=1500)
    tasks.task_2(NUMBER_OF_MODELS=10,CONV_KERNEL_SIZES=(7,5,3),PADDING=(3,2,1),DROPOUT=0.2,MAX_EPOCHS=1500)