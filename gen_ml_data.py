import numpy as np
import pandas as pd
from itertools import permutations
import random


def main():

    headers_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma",
                     "NumNotesInScale", "NumOctaves", "LastNoteIndex", "NextNote"]
    # Train & test data samples should be the same size!
    train_eeg_data_samples = [[0.021, 8.619, 92.24, 0.047, 0.197],
                              [0.005, 8.659, 93.362, 0.042, 0.196]]
    test_eeg_data_samples = [[93.307, 0.037, 0.011, 0.152, 8.552],
                             [0.013, 0.213, 0.046, 93.427, 8.648]]

    num_octaves = [1, 2, 3, 4, 5, 6, 7]
    num_notes_in_scale = [5, 7, 12]

    train_output = np.empty(shape=[1, 9], dtype=float)
    test_output = np.empty(shape=[1, 9], dtype=float)
    # train_output = np.vstack([train_output, headers_names])
    # test_output = np.vstack([test_output, headers_names])

    for i in range(len(train_eeg_data_samples)):
        # Grab current test and train sets of data
        train_current_eeg_data = train_eeg_data_samples[i]
        test_current_eeg_data = test_eeg_data_samples[i]

        # Get max & second max number:
        train_max_nums = getMaxAndSecondMaxNums(train_current_eeg_data)
        test_max_nums = getMaxAndSecondMaxNums(test_current_eeg_data)

        # Find all ordered combos of cur:
        train_combinations = getPermutations(train_current_eeg_data)
        test_combinations = getPermutations(test_current_eeg_data)

        for j in range(len(train_combinations)):
            train_current_combination = train_combinations[j]
            test_current_combination = test_combinations[j]

            # Training data: Get the max and second max
            train_max_index = getIndexFromNum(
                train_current_combination, train_max_nums[0])
            train_second_max_index = getIndexFromNum(
                train_current_combination, train_max_nums[1])

            # Training data: Get the max and second max
            test_max_index = getIndexFromNum(
                test_current_combination, test_max_nums[0])
            test_second_max_index = getIndexFromNum(
                test_current_combination, test_max_nums[1])

            # Iterate through all of the octaves
            for k in range(len(num_octaves)):
                current_num_octaves = num_octaves[k]

                # Iterate through all of the number of notes in a scale
                for l in range(len(num_notes_in_scale)):
                    notes_in_scale = num_notes_in_scale[l]

                    cur_range = notes_in_scale - 1
                    random_num_former = random.randrange(cur_range)
                    random_num_next = random.randrange(cur_range)

                    # “Delta”, “Theta”, “Alpha”, “Beta”, “Gamma”, “NumNotesInScale”, "NumOctaves", "LastNoteIndex”, “NewNote”
                    train_current_data = np.array([train_current_combination[0], train_current_combination[1], train_current_combination[2], train_current_combination[3],
                                                   train_current_combination[4], notes_in_scale, current_num_octaves, random_num_former, random_num_next])

                    # “Delta”, “Theta”, “Alpha”, “Beta”, “Gamma”, “NumNotesInScale”, "NumOctaves", "LastNoteIndex”, “NewNote”
                    test_current_data = np.array([test_current_combination[0], test_current_combination[1], test_current_combination[2], test_current_combination[3],
                                                  test_current_combination[4], notes_in_scale, current_num_octaves, random_num_former, random_num_next])

                    # print(train_current_data)
                    # Add it to the train and test output
                    train_output = np.vstack(
                        [train_output, train_current_data])
                    test_output = np.vstack([test_output, test_current_data])

    # Delete row
    train_output = np.delete(train_output, 0, 0)
    test_output = np.delete(test_output, 0, 0)

    train_output = pd.DataFrame(train_output, columns=headers_names)

    pd.DataFrame(train_output).to_csv(
        "train.csv", index=False)
    pd.DataFrame(test_output).to_csv(
        "test.csv", index=False)
    print(train_output)


def getMaxAndSecondMaxNums(array):
    array.sort()
    max_nums = [array[len(array) - 1], array[len(array) - 2]]
    return max_nums


def getIndexFromNum(array, max_num):
    return array.index(max_num)


def getPermutations(array):
    l = list(permutations(array))
    return l


if __name__ == "__main__":
    main()
