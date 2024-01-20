import torch
import numpy as np
from scipy import stats
import random

from tasks import tasks

def estimate_mean_confidence_interval(accuracies : list[float]) -> tuple[float, float]:
    if len(accuracies) == 1:
        return (accuracies[0], float("nan"))
    mean_accuracy = stats.tmean(accuracies)
    sem = stats.sem(accuracies)
    confidence_interval = mean_accuracy - stats.t.interval(0.95, len(accuracies)-1, loc=mean_accuracy, scale=sem)[0]
    return (mean_accuracy, confidence_interval)

if __name__ == "__main__":

    seed : int = 314

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Tasks to run:
    task_1 : bool = True
    task_2 : bool = True
    task_3 : bool = True

    # Whether to save the models' weights as .pt:
    save_models : bool = False
    
    # Number of iterations per task:
    iterations : int = 10

    accuracies_task_1 = []
    accuracies_task_2_data_aug = []
    accuracies_task_2_regular = []
    accuracies_task_3_classifier = []
    accuracies_task_3_svm = []

    # Task 1
    if task_1:
        with open("results.txt", "w") as f:
            for i in range(1,iterations+1):
                print(f"Task 1 - {i}:")
                accuracies_task_1.append(
                    tasks.task_1(
                        MAX_EPOCHS=300,
                        PATIENCE=10,
                        SAVE_MODEL=save_models,
                        PATH_DETAILS=f"_{i}"
                        )
                    )
            mean_accuracy_task_1, confidence_interval_task_1 = estimate_mean_confidence_interval(accuracies_task_1)
            print(f"The mean accuracy in Task 1 is {mean_accuracy_task_1}+-{confidence_interval_task_1}% with {iterations} iterations and 95% confidence interval.")
            f.write(f"{accuracies_task_1}\n")
            f.write(f"The mean accuracy in Task 1 is {mean_accuracy_task_1}+-{confidence_interval_task_1}% with {iterations} iterations and 95% confidence interval.\n")

    # Task 2
    if task_2:
        with open("results.txt", "a") as f:
            for i in range(1,iterations+1):
                # Task 2 - Data Augmentation
                print(f"Task 2 - {i} (Data Augmentation):")
                accuracies_task_2_data_aug.append(
                    tasks.task_2(
                        NUMBER_OF_MODELS=1,
                        DROPOUT=0.0,
                        BATCH_NORMALIZATION=False,
                        CONV_KERNEL_SIZES=(3,3,3),
                        CONV_KERNEL_CHANNELS=(8,16,32),
                        MAX_EPOCHS=500,
                        PATIENCE=30,
                        RELU=True,
                        KAIMING_NORMAL=False,
                        LEARNING_RATE=0.0003,
                        SAVE_MODEL=save_models,
                        PATH_DETAILS=f"_data_aug_{i}"
                        )
                )
                # Task 2 - Regularization
                print(f"Task 2 - {i} (Regularization):")
                accuracies_task_2_regular.append(
                    tasks.task_2(
                        NUMBER_OF_MODELS=1,
                        DROPOUT=0.2,
                        CONV_KERNEL_SIZES=(3,3,3),
                        CONV_KERNEL_CHANNELS=(8,16,32),
                        MAX_EPOCHS=500,
                        PATIENCE=30,
                        RELU=True,
                        KAIMING_NORMAL=False,
                        SAVE_MODEL=save_models,
                        PATH_DETAILS=f"_regular_{i}"
                        )
                )

            mean_accuracy_task_2_data_aug, confidence_interval_task_2_data_aug = estimate_mean_confidence_interval(accuracies_task_2_data_aug)
            print(f"The mean accuracy in Task 2 (Data Augmentation) is {mean_accuracy_task_2_data_aug}+-{confidence_interval_task_2_data_aug}%")
            f.write(f"{accuracies_task_2_data_aug}\n")
            f.write(f"The mean accuracy in Task 2 (Data Augmentation) is {mean_accuracy_task_2_data_aug}+-{confidence_interval_task_2_data_aug}%\n")

            mean_accuracy_task_2_regular, confidence_interval_task_2_regular = estimate_mean_confidence_interval(accuracies_task_2_regular)
            print(f"The mean accuracy in Task 2 (Regularization) is {mean_accuracy_task_2_regular}+-{confidence_interval_task_2_regular}%")
            f.write(f"{accuracies_task_2_regular}\n")
            f.write(f"The mean accuracy in Task 2 (Regularization) is {mean_accuracy_task_2_regular}+-{confidence_interval_task_2_regular}%\n")

        # Task 2 - All
        with open("results.txt", "a") as f:
            accuracy_task_2_all = tasks.task_2(
                NUMBER_OF_MODELS=10,
                DROPOUT=0.2,
                CONV_KERNEL_SIZES=(7,5,3,3),
                CONV_KERNEL_CHANNELS=(8,16,32,64),
                MAX_EPOCHS=500,
                PATIENCE=50,
                PATH_DETAILS="_all",
                SWISH_BETA=0.8,
                SAVE_MODEL=save_models
            )
            print(f"The accuracy in Task 2 (All) is {accuracy_task_2_all}%")
            f.write(f"The accuracy in Task 2 (All) is {accuracy_task_2_all}%")
    
    # Task 3
    if task_3:
        with open("results.txt", "a") as f:
            for i in range(1,iterations+1):
                print(f"Task 3 - {i}:")
                accuracy_task_3 = tasks.task_3(
                    MAX_EPOCHS=300,
                    PATIENCE=50,
                    LOAD_SVM=False,
                    PATH_DETAILS=f"_{i}",
                    SAVE_MODEL=save_models,
                    EPOCHS_SVM=3,
                    SVM_C=1.0,
                    SVM_TYPE="linear"
                    )
                accuracies_task_3_classifier.append(accuracy_task_3[0])
                accuracies_task_3_svm.append(accuracy_task_3[1])

            mean_accuracy_task_3_classifier, confidence_interval_task_3_classifier = estimate_mean_confidence_interval(accuracies_task_3_classifier)
            print(f"The mean accuracy in Task 3 (Classifier) is {mean_accuracy_task_3_classifier}+-{confidence_interval_task_3_classifier}%")
            f.write(f"{accuracies_task_3_classifier}\n")
            f.write(f"The mean accuracy in Task 3 (Classifier) is {mean_accuracy_task_3_classifier}+-{confidence_interval_task_3_classifier}%\n")

            mean_accuracy_task_3_svm, confidence_interval_task_3_svm = estimate_mean_confidence_interval(accuracies_task_3_svm)
            print(f"The mean accuracy in Task 3 (SVM) is {mean_accuracy_task_3_svm}+-{confidence_interval_task_3_svm}%")
            f.write(f"{accuracies_task_3_svm}\n")
            f.write(f"The mean accuracy in Task 3 (SVM) is {mean_accuracy_task_3_svm}+-{confidence_interval_task_3_svm}%\n")