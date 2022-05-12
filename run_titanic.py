"""
CSCC11 - Introduction to Machine Learning, Winter 2022, Assignment 3
B. Chan
"""

import _pickle as pickle
import numpy as np

from experiments import run_experiment
from hyperparameters import TITANIC_HYPERPARAMETERS

def main(final_hyperparameters):
    with open("./datasets/titanic.pkl", "rb") as f:
        titanic_data =  pickle.load(f)

    train_X = titanic_data['train_X']
    train_y = titanic_data['train_y']
    
    validation_X = titanic_data['val_X']
    validation_y = titanic_data['val_y']

    test_X, test_y = None, None
    if final_hyperparameters:
        test_X = titanic_data['test_X']
        test_y = titanic_data['test_y']

    # You can try different seeds and check the model's performance!
    seeds = np.random.RandomState(0).randint(low=0, high=65536, size=(10))

    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    TITANIC_HYPERPARAMETERS["debug"] = False
    TITANIC_HYPERPARAMETERS["num_classes"] = 2
    for seed in seeds:
        TITANIC_HYPERPARAMETERS["rng"] = np.random.RandomState(seed)

        train_accuracy, validation_accuracy, test_accuracy = run_experiment(TITANIC_HYPERPARAMETERS,
                                                                            train_X,
                                                                            train_y,
                                                                            validation_X,
                                                                            validation_y,
                                                                            test_X,
                                                                            test_y)

        print(f"Seed: {seed} - Train Accuracy: {train_accuracy} - Validation Accuracy: {validation_accuracy} - Test Accuracy: {test_accuracy}")
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        test_accuracies.append(test_accuracy)

    print(f"Train Accuracies - Mean: {np.mean(train_accuracies)} - Standard Deviation: {np.std(train_accuracies, ddof=0)}")
    print(f"Validation Accuracies - Mean: {np.mean(validation_accuracies)} - Standard Deviation: {np.std(validation_accuracies, ddof=0)}")
    print(f"Test Accuracies - Mean: {np.mean(test_accuracies)} - Standard Deviation: {np.std(test_accuracies, ddof=0)}")


if __name__ == "__main__":
    final_hyperparameters = True
    main(final_hyperparameters=final_hyperparameters)
