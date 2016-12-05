#!/usr/bin/python
'''
@File:           RecordSorter.py
@Description:    PySpark application for sorting customer records based on
                 their IDs.
@Author:         Chetan Borse
@EMail:          cborse@uncc.edu
@Created on:     12/04/2016
@Usage:          spark-submit --master yarn --deploy-mode client 
                    src/RecordSorter.py
                    -i "output/Spark/RecommenderSystem/Recommendations"
                    -o "output/Spark/RecommenderSystem/CustomerRecords"
@python_version: 2.7
===============================================================================
'''


import argparse
import warnings

from pyspark import SparkContext, SparkConf

warnings.filterwarnings("ignore")


def RecordSorter(**args):
    """
    Entry point for PySpark application to sort customer records based on
    their IDs.
    """
    # Read arguments
    input  = args['input']
    output = args['output']

    # Create SparkContext object
    conf = SparkConf()
    conf.setAppName("RecordSorter")
    conf.set('spark.driver.memory', '4g')
#     conf.set('spark.executor.cores', '4')
#     conf.set('spark.executor.memory', '10g')
    conf.set('spark.kryoserializer.buffer.max', '2000')
#     conf.set('spark.shuffle.memoryFraction', '0.5')
    conf.set('spark.yarn.executor.memoryOverhead', '4096')
    
    sc = SparkContext(conf=conf)

    # Read product recommendations data
    input = sc.textFile(input)

    # Preprocess customer records before sorting it further
    input = input.filter(lambda x: x != None) \
                 .map(lambda x: (x.split(',', 1)[0], 
                                 x.split(',', 1)[1]))
    input = input.partitionBy(10).cache()
    
    # Sort customer records by IDs
    input = input.sortBy(lambda x: int(x[0]), \
                         ascending=True, \
                         numPartitions=1)
    
    # Save sorted customer records
    input = input.map(lambda x: x[0] + ',' + x[1])
    input.saveAsTextFile(output)

    # Shut down SparkContext
    sc.stop()


if __name__ == "__main__":
    """
    Entry point.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description='Customer Record Sorter Application',
                                     prog='spark-submit --master yarn --deploy-mode client \
                                           src/RecordSorter.py \
                                           -i <input> \
                                           -o <output>')

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Unordered customer records.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Sorted customer records.")

    # Read user inputs
    args = vars(parser.parse_args())

    # Run Customer Record Sorter Application
    RecordSorter(**args)

