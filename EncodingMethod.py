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
num_len = len(records.first()[1:7])
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

print "Raw data: " + str(first[0:])
print "Acceleration Label: " + str(acc_first_point.label)
print "Linear Model feature vector:\n" + str(acc_first_point.features)
print "Linear Model feature vector length: " + str(len(acc_first_point.features))

print "Raw data: " + str(first[0:])
print "Horsepower Label: " + str(hp_first_point.label)
print "Linear Model feature vector:\n" + str(hp_first_point.features)
print "Linear Model feature vector length: " + str(len(hp_first_point.features))

# Acceleration Predictions
acc_linear_model = LinearRegressionWithSGD.train(accData, iterations=10, step=0.01, intercept=False)
true_vs_predicted_acc = accData.map(lambda p: (p.label, acc_linear_model.predict(p.features)))
print "Linear Model predictions for Acceleration: " + str(true_vs_predicted_acc.take(5))

# Horsepower Predictions
hp_linear_model = LinearRegressionWithSGD.train(hpData, iterations=10, step=0.01, intercept=False)
true_vs_predicted_hp = hpData.map(lambda p: (p.label, hp_linear_model.predict(p.features)))
print "Linear Model predictions for Horsepower: " + str(true_vs_predicted_hp.take(5))


def squared_error(actual, pred):
    return (pred - actual) ** 2


def abs_error(actual, pred):
    return np.abs(pred - actual)


def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1)) ** 2


mse = true_vs_predicted_hp.map(lambda (t, p): squared_error(t, p)).mean()
mae = true_vs_predicted_hp.map(lambda (t, p): abs_error(t, p)).mean()
rmsle = np.sqrt(true_vs_predicted_hp.map(lambda (t, p): squared_log_error(t, p)).mean())

print "Horsepower Errors"
print "-----------------------------------"
print "Linear Model - Mean Squared Error: %2.4f" % mse
print "Linear Model - Mean Absolute Error: %2.4f" % mae
print "Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle


mse = true_vs_predicted_acc.map(lambda (t, p): squared_error(t, p)).mean()
mae = true_vs_predicted_acc.map(lambda (t, p): abs_error(t, p)).mean()
rmsle = np.sqrt(true_vs_predicted_acc.map(lambda (t, p): squared_log_error(t, p)).mean())

print "Acceleration Errors"
print "-----------------------------------"
print "Linear Model - Mean Squared Error: %2.4f" % mse
print "Linear Model - Mean Absolute Error: %2.4f" % mae
print "Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle
