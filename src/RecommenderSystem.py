#!/usr/bin/python
'''
@File:           RecommenderSystem.py
@Description:    Product Recommendation System application for recommending 
                 financial products to customer.
@Author:         Chetan Borse
@EMail:          cborse@uncc.edu
@Created on:     12/04/2016
@Usage:          spark-submit --master yarn --deploy-mode client 
                    src/RecommenderSystem.py
                    -i "input/RecommenderSystem/preprocessed_train.csv"
                    -o "input/RecommenderSystem/preprocessed_test.csv"
                    -d "output/Spark/RecommenderSystem/Recommendations"
                    -k 7
@python_version: 2.7
===============================================================================
'''


import argparse
import warnings
import numpy as np
import operator
import math

from pyspark import SparkContext, SparkConf

warnings.filterwarnings("ignore")


# Global variables
PRODUCT_FREQUENCY = {}
PRODUCT_PRIOR = {}
TOP_K = 7

# List of customer attributes
Customer = {0: "ncodpers",
            1: "ind_empleado",
            2: "pais_residencia",
            3: "sexo",
            4: "age",
            5: "ind_nuevo",
            6: "antiguedad",
            7: "indrel",
            8: "indrel_1mes",
            9: "tiprel_1mes",
            10: "indresi",
            11: "indext",
            12: "conyuemp",
            13: "canal_entrada",
            14: "indfall",
            15: "nomprov",
            16: "ind_actividad_cliente",
            17: "segmento"}

# List of products
Product = {0: "ind_ahor_fin_ult1",
           1: "ind_aval_fin_ult1",
           2: "ind_cco_fin_ult1",
           3: "ind_cder_fin_ult1",
           4: "ind_cno_fin_ult1",
           5: "ind_ctju_fin_ult1",
           6: "ind_ctma_fin_ult1",
           7: "ind_ctop_fin_ult1",
           8: "ind_ctpp_fin_ult1",
           9: "ind_deco_fin_ult1",
           10: "ind_deme_fin_ult1",
           11: "ind_dela_fin_ult1",
           12: "ind_ecue_fin_ult1",
           13: "ind_fond_fin_ult1",
           14: "ind_hip_fin_ult1",
           15: "ind_plan_fin_ult1",
           16: "ind_pres_fin_ult1",
           17: "ind_reca_fin_ult1",
           18: "ind_tjcr_fin_ult1",
           19: "ind_valo_fin_ult1",
           20: "ind_viv_fin_ult1",
           21: "ind_nomina_ult1",
           22: "ind_nom_pens_ult1",
           23: "ind_recibo_ult1"}


def Transform(customerEntry):
    '''
    Transform customer entry.

    @param: customerEntry    => Customer entry.

    @returns:                => Transformed customer entry.
    '''
    if not customerEntry:
        return
    
    if "ind_empleado" in customerEntry:
        return None
    
    customerEntry = customerEntry.split(',')
    customerEntry = map(lambda x: x.strip(), customerEntry)
    
    return customerEntry


def TransformTest(customerEntry):
    '''
    Transform testing data.

    @param: customerEntry    => Customer entry.

    @yields:                 => Tuple of (<Customer Information>, <Customer ID>)
    '''
    customerEntry = Transform(customerEntry)
    if not customerEntry:
        return
    
    customerID = int(customerEntry[0])
    
    for i, attribute in enumerate(customerEntry):
        for p in Product.values():
            try:
                yield (Customer[i]+"="+attribute+"@"+p, customerID)
            except KeyError, e:
                pass


def TransformCustomerInformation(customerEntry):
    '''
    Transform customer information.

    @param: customerEntry    => Customer entry.

    @returns:                => Tuple of (<Customer Information>, <Attribute Likelihood>)
    '''
    if not customerEntry:
        return
    
    product = customerEntry[0].split('@')[1]
    attributeLikelihood = customerEntry[1][0]
    customer = customerEntry[1][1]

    return (str(customer)+'@'+product, attributeLikelihood)


def GetCustomerShoppingHistory(customerEntry):
    '''
    Get customer shopping history.

    @param: customerEntry    => Customer entry for which product purchase history
                                is to be computed.

    @yields:                 => Tuple of (<Customer>, <Product>)
    '''
    customerEntry = Transform(customerEntry)
    if not customerEntry:
        return
    
    productPurchased = customerEntry[18:]
    
    # List all the products purchased by customer in the past
    for j, isPurchased in enumerate(productPurchased):
        if isPurchased == '1':
            yield (customerEntry[0], Product[j])


def CalculateProductFrequency(customerEntry):
    '''
    Calculate product purchase frequency.

    @param: customerEntry    => Customer entry for which product purchase frequency
                                is to be computed.

    @yields:                 => Tuple of (<Product>, <Frequency>)
    '''
    customerEntry = Transform(customerEntry)
    if not customerEntry:
        return
    
    productPurchased = customerEntry[18:]
    
    # If product is purchased, emit customer-frequency pair
    for j, isPurchased in enumerate(productPurchased):
        if isPurchased == '1':
            yield (Product[j], 1.0)


def CalculateAttributeFrequency(customerEntry):
    '''
    Calculate frequency of customer attribute/behaviour occurence.

    @param: customerEntry    => Customer entry for which attribute/behaviour frequency
                                is to be computed.

    @yields:                 => Tuple of (<Customer Attribute>, <Frequency>)
    '''
    customerEntry = Transform(customerEntry)
    if not customerEntry:
        return

    # Customer attributes
    customerAttribute = customerEntry[:18]
    
    # Product purchased
    productPurchased = customerEntry[18:]
    
    # Calculate customer attribute frequency
    for i, attribute in enumerate(customerAttribute):
        for j, isPurchased in enumerate(productPurchased):
            if isPurchased == '1':
                try:
                    yield (Customer[i]+"="+attribute+"@"+Product[j], 1.0)
                except KeyError, e:
                    pass


def CalculateAttributeLikelihood(customerEntry):
    '''
    Calculate likelihood of customer attribute/behaviour.

    @param: customerEntry    => Customer entry for which attribute/behaviour likelihood
                                is to be computed.

    @returns:                => Tuple of (<Customer Attribute>, <Likelihood>)
    '''
    product = customerEntry[0].split('@')[1]
    
    # Calculate customer attribute/behaviour likelihood.
    # Make sure that absence of entry for product purchase 
    # will not make the likelihood 0.
    attributeLikelihood = np.log((customerEntry[1]+1.0) / \
                                 (PRODUCT_FREQUENCY[product]+1.0))
    
    return (customerEntry[0], attributeLikelihood)


def CalculatePosterior(customerEntry):
    '''
    Calculate posterior probabilities.

    @param: customerEntry    => Customer entry for which posterior probability
                                is to be computed.

    @returns:                => Tuple of (<Customer>, <Posterior Probability>)
    '''
    if not customerEntry:
        return
    
    # Customer attribute/behaviour likelihood
    product = customerEntry[0].split('@')[1]
    likelihood = customerEntry[1]
    
    # Prior for purchasing certain product
    prior = PRODUCT_PRIOR[product]
    
    # Calculate posterior in log space
    posterior = likelihood + np.log(prior)
    
    return (customerEntry[0], posterior)


def PostprocessPosterior(customerEntry):
    '''
    Post-process posterior probabilities.

    @param: customerEntry    => Posterior probabilities of different products
                                for a given customer.

    @returns:                => Tuple of (<Customer>, 
                                          (<Product Entry>, <Posterior Probability>))
    '''
    if not customerEntry:
        return
    
    # Transform posterior from log space to original space
    customer = customerEntry[0].split('@')[0]
    product = customerEntry[0].split('@')[1]
    posteriorProbability = math.exp(customerEntry[1])
    
    return (customer, (product, posteriorProbability))


def RemoveOldShopping(customerEntry):
    '''
    For existing customers, remove products purchased in the past.

    @param: customerEntry    => Posterior probabilities of different products
                                and product purchase history for a given customer.

    @yields:                 => Tuple of (<Customer>, <Product Entry>)
    '''
    if not customerEntry:
        return
    
    customer = customerEntry[0]
    predictions = customerEntry[1][0]
    oldShopping = customerEntry [1][1]
    
    for prediction in predictions:
        if prediction[0] not in oldShopping:
            yield (customer, [prediction])


def GetTopRecommendation(customerEntry):
    '''
    Recommend top products based on its posterior probabilities.

    @param: customerEntry    => Posterior probabilities of different products
                                for a given customer.

    @returns:                => Tuple of (<Customer>, <Top Recommendations>)
    '''
    if not customerEntry:
        return
    
    customer = customerEntry[0]
    predictions = customerEntry[1]
    
    # Sort products based on posterior probabilities
    sortedPredictions = sorted(predictions, key=operator.itemgetter(1), reverse=True)
    
    # Pick top recommendations
    topPredictions = sortedPredictions[:TOP_K]
    
    # Remove posterior probabilities
    topRecommendations = map(lambda x: x[0], topPredictions)
    
    return (customer, topRecommendations)


def RecommenderSystem(**args):
    """
    Entry point for Product Recommendation System application.
    """
    # Global variables
    global PRODUCT_FREQUENCY
    global PRODUCT_PRIOR
    global TOP_K


    ###########################################################################
    #                         SPARK CONFIGURATION                             #
    ###########################################################################

    # Read arguments
    training_csv = args['training_csv']
    testing_csv  = args['testing_csv']
    output_dir   = args['output_dir']
    TOP_K         = args['top_k']

    # Create SparkContext object
    conf = SparkConf()
    conf.setAppName("RecommenderSystem")
    conf.set('spark.driver.memory', '4g')
#     conf.set('spark.executor.cores', '4')
#     conf.set('spark.executor.memory', '10g')
    conf.set('spark.kryoserializer.buffer.max', '2000')
#     conf.set('spark.shuffle.memoryFraction', '0.5')
    conf.set('spark.yarn.executor.memoryOverhead', '4096')
    
    sc = SparkContext(conf=conf)


    ###########################################################################
    #                                TRAINING                                 #
    ###########################################################################
    
    # Read product purchase data from the past
    training_data = sc.textFile(training_csv)

    # Find out what products customer purchased in the past
    customerShoppingHistory = training_data.flatMap(GetCustomerShoppingHistory) \
                                           .filter(lambda x: x != None) \
                                           .distinct() \
                                           .map(lambda x: (x[0], [x[1]])) \
                                           .reduceByKey(lambda x, y: x + y)
    customerShoppingHistory = customerShoppingHistory.partitionBy(10).cache()
    
    # Find total shopping instances
    totalShoppingInstances = training_data.count()
    
    # Find how many times each type of product is purchased, 
    # i.e. product purchase frequency
    productFrequency = training_data.flatMap(CalculateProductFrequency) \
                                    .filter(lambda x: x != None) \
                                    .reduceByKey(lambda x, y: x + y)
    PRODUCT_FREQUENCY = productFrequency.collectAsMap()
    
    # Calculate product purchase prior for every type of product
    prior = productFrequency.map(lambda x: (x[0], x[1]/totalShoppingInstances))
    PRODUCT_PRIOR = prior.collectAsMap()
    
    # Calculate likelihood of customer attribute/behaviour 
    # given certain type of product purchased
    attributeLikelihood = training_data.flatMap(CalculateAttributeFrequency) \
                                       .filter(lambda x: x != None) \
                                       .reduceByKey(lambda x, y: x + y) \
                                       .map(CalculateAttributeLikelihood)
    attributeLikelihood = attributeLikelihood.partitionBy(10).cache()


    ###########################################################################
    #                                 TESTING                                 #
    ###########################################################################

    # Read testing data of customers for predicting/recommending future purchases
    testing_data = sc.textFile(testing_csv)

    # Preprocess testing data for further predictions
    testing_data = testing_data.flatMap(TransformTest) \
                               .filter(lambda x: x != None)
    testing_data = testing_data.partitionBy(10).cache()

    # Calculate posterior probabilities for the purchase of every type of product
    # and for every customer.
    prediction = attributeLikelihood.join(testing_data) \
                                    .map(TransformCustomerInformation) \
                                    .filter(lambda x: x != None) \
                                    .reduceByKey(lambda x, y: x + y) \
                                    .map(CalculatePosterior) \
                                    .filter(lambda x: x != None) \
                                    .map(PostprocessPosterior) \
                                    .filter(lambda x: x != None) \
                                    .map(lambda x: (x[0], [x[1]])) \
                                    .reduceByKey(lambda x, y: x + y)
    prediction = prediction.partitionBy(10).cache()

    # Find the list of old customers and do not recommend the products 
    # that were already purchased.
    # As this recommendation system recommends financial products, 
    # it may not be beneficial for bank to recommend the same product again.
    # e.g. Saving Account, Home Loan, etc.
    oldCustomers = prediction.join(customerShoppingHistory) \
                             .flatMap(RemoveOldShopping) \
                             .filter(lambda x: x != None) \
                             .reduceByKey(lambda x, y: x + y)

    # Identify new customers.
    # We can recommend every possible product to new customers.
    newCustomers = prediction.subtractByKey(oldCustomers)

    # Merge both old and new customers
    allCustomers = oldCustomers.union(newCustomers)

    # Suggest only top recommendations
    recommendation = allCustomers.map(GetTopRecommendation) \
                                 .filter(lambda x: x != None)
    
    # Sort customers by customer id
    recommendation = recommendation.sortBy(lambda x: int(x[0]), \
                                           ascending=True, \
                                           numPartitions=1)
    
    # Save recommendations
    recommendation = recommendation.map(lambda x: x[0] + ',' + ' '.join(x[1]))
    recommendation.saveAsTextFile(output_dir)


    # Shut down SparkContext
    sc.stop()


if __name__ == "__main__":
    """
    Entry point.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description='Product Recommendation System Application',
                                     prog='spark-submit --master yarn --deploy-mode client \
                                           src/RecommenderSystem.py \
                                           -i <training_csv> \
                                           -o <testing_csv> \
                                           -d <output_dir>')

    parser.add_argument("-i", "--training_csv", type=str, required=True,
                        help="Training data of products purchased in the past.")
    parser.add_argument("-o", "--testing_csv", type=str, required=True,
                        help="Testing data of customers for whom product recommendations \
                              should be given.")
    parser.add_argument("-d", "--output_dir", type=str, required=True,
                        help="Output directory to store top K recommendations for every \
                              customer.")
    parser.add_argument("-k", "--top_k", type=int, default=7,
                        help="Top K recommendations to be suggested, default: 7")

    # Read user inputs
    args = vars(parser.parse_args())

    # Run Product Recommendation System Application
    RecommenderSystem(**args)

