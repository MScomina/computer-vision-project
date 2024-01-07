import torch
import numpy as np
from tasks import tasks
import random

if __name__ == "__main__":

    seed : int = 314

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Task 1:")
    tasks.task_1(MAX_EPOCHS=1500)
    print("Task 2:")
    tasks.task_2(NUMBER_OF_MODELS=10,CONV_KERNEL_SIZES=(7,7,5),PADDING=(3,3,2),DROPOUT=0.25,MAX_EPOCHS=2000)
    print("Task 3:")
    tasks.task_3(MAX_EPOCHS=300,PATIENCE=100,EPOCHS_SVM=3)