#!/usr/bin/python
'''
@File:           Metrics.py
@Description:    PySpark application for evaluating recommender system.
@Author:         Chetan Borse
@EMail:          cborse@uncc.edu
@Created on:     12/04/2016
@Usage:          python src/Metrics.py
                    -o "input/ActualRecommendations.csv"
                    -p "output/PredictedRecommendations.csv"
@python_version: 2.7
===============================================================================
'''


import argparse
import numpy as np


def AveragePrecisionAtK(actual, predicted, k=7):
    """
    Computes the average precision at k.

    @param: actual        => List of actual recommendations.
    @param: predicted     => List of predicted recommendations.
    @param: k             => Maximum number of predicted elements.

    @returns:             => Average precision at k.
    """
    # If size of 'predicted' is more than k,
    # then truncate it to size k only.
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # Calculate precision score
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    # Calculate average precision score and return it
    return score / min(len(actual), k)


def MeanAveragePrecisionAtK(actual, predicted, k=7):
    """
    Computes the mean average precision at k.

    @param: actual        => List of lists of actual recommendations.
    @param: predicted     => List of list of predicted recommendations.
    @param: k             => Maximum number of predicted elements.

    @returns:             => Mean average precision at k.
    """
    return np.mean([AveragePrecisionAtK(a, p, k) for a, p in zip(actual, predicted)])


def Metrics(**args):
    """
    Entry point for PySpark application to evaluate recommender system.
    """
    # Read arguments
    actual_csv    = args['actual_csv']
    predicted_csv = args['predicted_csv']
    
    # Read actual recommendations
    actual_recommendations = np.genfromtxt(actual_csv,
                                           dtype=np.str,
                                           delimiter=",",
                                           skip_header=1)
    actual_recommendations = map(lambda x: x[1].split(' '), 
                                 actual_recommendations)

    # Read predicted recommendations
    predicted_recommendations = np.genfromtxt(predicted_csv,
                                              dtype=np.str,
                                              delimiter=",",
                                              skip_header=1)
    predicted_recommendations = map(lambda x: x[1].split(' '), 
                                    predicted_recommendations)
    
    # Evaluate recommender system
    evaluation = MeanAveragePrecisionAtK(actual_recommendations, predicted_recommendations, 7)
    
    print evaluation


if __name__ == "__main__":
    """
    Entry point.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description='Application for evaluating recommender system',
                                     prog='python src/Metrics.py \
                                           -o <actual_csv> \
                                           -p <predicted_csv>')

    parser.add_argument("-o", "--actual_csv", type=str, required=True,
                        help="Actual recommendations.")
    parser.add_argument("-p", "--predicted_csv", type=str, required=True,
                        help="Predicted recommendations.")

    # Read user inputs
    args = vars(parser.parse_args())

    # Run application for evaluating recommender system
    Metrics(**args)

