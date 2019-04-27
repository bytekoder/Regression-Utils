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

path = "file:///home/cloudera/Desktop/CS63/Small_Car_Data.csv"

user_data = sc.textFile(path)
user_fields = user_data.map(lambda line: line.split(","))
all_countries = user_fields.map(lambda fields: fields[9]).distinct().collect()
all_years = user_fields.map(lambda fields: fields[7]).distinct().collect()
all_countries = [x.strip(' ') for x in all_countries]
all_years = [x.strip(' ') for x in all_years]

print all_countries.sort()
print all_years.sort()

idx = 0
all_countries_dict = {}
for o in all_countries.collect():
    all_countries_dict[o] = idx
    idx += 1
print all_countries_dict['France']

# print all_countries_dict['France']


# print "Encoding of 'USA': %d" % all_countries_dict['USA']
# print "Encoding of 'Germany': %d" % all_countries_dict['Germany']

K = len(all_countries_dict)
binary_x = np.zeros(K)
k_programmer = all_countries_dict['Germany']
binary_x[k_programmer] = 1
print "Binary feature vector: %s" % binary_x
print "Length of binary vector: %d" % K

