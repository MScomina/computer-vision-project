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
    tasks.task_1(MAX_EPOCHS=300,PATIENCE=30)
    print("Task 2:")
    tasks.task_2(NUMBER_OF_MODELS=10,CONV_KERNEL_SIZES=(7,7,7),PADDING=(3,3,3),DROPOUT=0.25,MAX_EPOCHS=300,PATIENCE=50,LEARNING_RATE=0.001,BETA=1.5,CAP=10.0)
    print("Task 3:")
    tasks.task_3(MAX_EPOCHS=300,PATIENCE=50,EPOCHS_SVM=3)