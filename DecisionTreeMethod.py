from spark.proj.scripts.twisted_server import sc

from pyspark.sql.types import *
import numpy as np
from pyspark.sql import Row
import pyspark.mllib
import pyspark.mllib.regression
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree

from pyspark.mllib.evaluation import RegressionMetrics

path = "file:///home/cloudera/Desktop/CS63/cars.csv"
raw_data = sc.textFile(path)
num_data = raw_data.count()
records = raw_data.map(lambda x: x.split(","))
first = records.first()
print first
print num_data
records.cache()


# function
def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()


mappings = [get_mapping(records, i) for i in range(8, 10)]
cat_len = np.sum(map(len, mappings))
num_len = len(records.first()[1:6])
total_len = num_len + cat_len

print "Feature vector length for categorical features: %d" % cat_len
print "Feature vector length for numerical features: %d" % num_len
print "Total feature vector length: %d" % total_len


def extract_features(record):
    cat_vec = np.zeros(cat_len)
    i = 0
    step = 0
    for field in record[8:10]:
        m = mappings[i]
        idx = m[field]
        cat_vec[idx + step] = 1
        i += 1
        step += len(m)
        num_vec = np.array([float(field) for field in record[5:6]])
        return np.concatenate((cat_vec, num_vec))


def extract_hp_label(record):
    return record[6]


def extract_acc_label(record):
    return record[5]


accData = records.map(lambda r: LabeledPoint(extract_acc_label(r), extract_features(r)))
hpData = records.map(lambda r: LabeledPoint(extract_hp_label(r), extract_features(r)))

acc_first_point = accData.first()
hp_first_point = hpData.first()


def extract_features_dt(record):
    return np.array(map(float, record[5:6]))


# Decision Tree Method

# Feature vector creation for acceleration
data_dt_acc = records.map(lambda r: LabeledPoint(extract_acc_label(r), extract_features_dt(r)))
first_point_dt_acc = data_dt_acc.first()
print "Decision Tree feature vector: " + str(first_point_dt_acc.features)
print "Decision Tree feature vector length: " + str(len(first_point_dt_acc.features))

# Feature vector creation for horsepower
data_dt_hp = records.map(lambda r: LabeledPoint(extract_hp_label(r), extract_features_dt(r)))
first_point_dt_hp = data_dt_hp.first()
print "Decision Tree feature vector: " + str(first_point_dt_hp.features)
print "Decision Tree feature vector length: " + str(len(first_point_dt_hp.features))

# Acceleration predictions
dt_model_acc = DecisionTree.trainRegressor(data_dt_acc, {})
acc_preds = dt_model_acc.predict(data_dt_acc.map(lambda p: p.features))
actual = accData.map(lambda p: p.label)
true_vs_predicted_dt_acc = actual.zip(acc_preds)
print "Acceleration Predictions ############"
print "Decision Tree predictions: " + str(true_vs_predicted_dt_acc.take(5))
print "Decision Tree depth: " + str(dt_model_acc.depth())
print "Decision Tree number of nodes: " + str(dt_model_acc.numNodes())

# Horsepower predictions
dt_model_hp = DecisionTree.trainRegressor(data_dt_hp, {})
hp_preds = dt_model_hp.predict(data_dt_hp.map(lambda p: p.features))
actual = hpData.map(lambda p: p.label)
true_vs_predicted_dt_hp = actual.zip(hp_preds)
print "Horsepower Predictions ############"
print "Decision Tree predictions: " + str(true_vs_predicted_dt_hp.take(5))
print "Decision Tree depth: " + str(dt_model_hp.depth())
print "Decision Tree number of nodes: " + str(dt_model_hp.numNodes())


def squared_error(actual, pred):
    return (pred - actual) ** 2


def abs_error(actual, pred):
    return np.abs(pred - actual)


def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1)) ** 2


mse = true_vs_predicted_dt_acc.map(lambda (t, p): squared_error(t, p)).mean()
mae = true_vs_predicted_dt_acc.map(lambda (t, p): abs_error(t, p)).mean()
rmsle = np.sqrt(true_vs_predicted_dt_acc.map(lambda (t, p): squared_log_error(t, p)).mean())

print "Horsepower Errors"
print "-----------------------------------"
print "Linear Model - Mean Squared Error: %2.4f" % mse
print "Linear Model - Mean Absolute Error: %2.4f" % mae
print "Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle

mse = true_vs_predicted_dt_hp.map(lambda (t, p): squared_error(t, p)).mean()
mae = true_vs_predicted_dt_hp.map(lambda (t, p): abs_error(t, p)).mean()
rmsle = np.sqrt(true_vs_predicted_dt_hp.map(lambda (t, p): squared_log_error(t, p)).mean())

print "Acceleration Errors"
print "-----------------------------------"
print "Linear Model - Mean Squared Error: %2.4f" % mse
print "Linear Model - Mean Absolute Error: %2.4f" % mae
print "Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle
