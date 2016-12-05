# Create new input directory on HDFS and delete existing input 
# directory.
hadoop fs -rm -r input/RecommenderSystem
hadoop fs -mkdir -p input/RecommenderSystem


# Create new output directories on HDFS and delete existing output 
# directories generated by previous execution instance of 
# Spark application.
hadoop fs -mkdir -p output/Spark/RecommenderSystem
hadoop fs -rm -r output/Spark/RecommenderSystem


# Preprocess training and testing csv.
#python src/DataWrangler.py -i "input/train.csv" -o "input/preprocessed_train.csv"
#python src/DataWrangler.py -i "input/test.csv" -o "input/preprocessed_test.csv"


# Copy preprocessed traning and testing csv from local filesystem to HDFS filesystem.
hadoop fs -put input/preprocessed_train.csv input/RecommenderSystem
hadoop fs -put input/preprocessed_test.csv input/RecommenderSystem


# Train Recommender System model on provided training data and
# recommend products based on testing customer data.
spark-submit --master yarn --deploy-mode client --class SortByKeyDF src/RecommenderSystem.py -i "input/RecommenderSystem/preprocessed_train.csv" -o "input/RecommenderSystem/preprocessed_test.csv" -d "output/Spark/RecommenderSystem/Recommendations" -k 7
spark-submit --master yarn --deploy-mode client src/RecordSorter.py -i "output/Spark/RecommenderSystem/Recommendations" -o "output/Spark/RecommenderSystem/CustomerRecords"


# Fetch final output from HDFS.
rm -r output/*
hadoop fs -get output/Spark/RecommenderSystem/* output/
cp output/CustomerRecords/part-00000 output/PredictedRecommendations.csv


# Evaluate Recommendation System.
python src/Metrics.py -o "input/ActualRecommendations.csv" -p "output/PredictedRecommendations.csv"


