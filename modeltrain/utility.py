import numpy as np
import os
import pandas as pd


def get_sample_counts(output_dir, dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    counts = []
    for i in class_names:
        pathTmp = os.path.join(output_dir, i)
        files = [
            file
            for file in os.listdir(pathTmp)
            if os.path.isfile(os.path.join(pathTmp, file))
        ]
        counts.append(len(files))
    # df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
    total_count = sum(counts)
    # labels = df[class_names].as_matrix()
    positive_counts = counts
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts
