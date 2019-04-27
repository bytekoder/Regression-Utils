from spark.proj.scripts.twisted_server import sc

import numpy as np
from pyspark.sql.types import *
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
raw_car_data = sc.textFile(path)
num_data = raw_car_data.count()

records = raw_car_data.map(lambda x: x.split(","))
first = records.first()
print first

print num_data

data_with_idx = records.zipWithIndex().map(lambda (k, v): (v, k))
print data_with_idx

test_data = data_with_idx.sample(False, 0.1233, 100)
train = data_with_idx.subtractByKey(test_data)
train_data = train.map(lambda (idx, p): p)
test_data_extracted = test_data.map(lambda (idx, p): p)
train_size = train_data.count()
test_size = test_data_extracted.count()

print "Training data size: %d" % train_size
print "Test data size: %d" % test_size
print "Total data size: %d" % num_data
print "Train + Test size: %d" % (test_size + train_size)

df = records.map(lambda line: Row(Displacement=line[2], Horsepower=line[6])).toDF()
df.show(10)

df = df.select('Horsepower', 'Displacement')
df = df[df.Displacement > 0]
df = df[df.Horsepower > 0]
df.describe(['Horsepower', 'Displacement']).show()
temp = df.map(lambda line: LabeledPoint(line[0], [line[1:]]))
temp.take(5)

linearModel = LinearRegressionWithSGD.train(temp, 10000, 0.0001, intercept=False)
linearModel.weights

test_data.take(10)

true_vs_predicted = temp.map(lambda p: (p.label, linearModel.predict(p.features)))
print "Linear Model predictions: " + str(true_vs_predicted.take(100))


def squared_error(actual, pred):
    return (pred - actual) ** 2


def abs_error(actual, pred):
    return np.abs(pred - actual)


def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1)) ** 2


mse = true_vs_predicted.map(lambda (t, p): squared_error(t, p)).mean()
mae = true_vs_predicted.map(lambda (t, p): abs_error(t, p)).mean()
rmsle = np.sqrt(true_vs_predicted.map(lambda (t, p): squared_log_error(t, p)).mean())
print "Linear Model - Mean Squared Error: %2.4f" % mse
print "Linear Model - Mean Absolute Error: %2.4f" % mae
print "Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle
