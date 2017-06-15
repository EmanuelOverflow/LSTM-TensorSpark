# LSTM-TensorSpark

Implementation of a LSTM with [TensorFlow](https://www.tensorflow.org/) and distributed on [Apache Spark](http://spark.apache.org/) 

There are provided three different implementations:

- Distributed on Spark;
- Standalone;
- Standalone with speed improvements;

Detailed explanation here: [Distributed implementation of a LSTM on Spark and Tensorflow](http://www.slideshare.net/emanueldinardo/distributed-implementation-of-a-lstm-on-spark-and-tensorflow-69787635)

### Important

Developed for academic purpose


## Dependencies

Distributed model needs:
- Python 2.6+
- Pyspark
- TensorFlow
- Numpy
- Argparse

Standalone models need:
- Python 2.6+
- TensorFlow
- Numpy
- Argparse


## Usage

### Example using Spark: 

From src directory

```
pyspark rnn.py -e 10 -hl 5 -i ../dataset/iris.data -t ../dataset/data.config -p 5 
```

* -e: number of epochs;
* -hl: number of hidden layer;
* -i: input data;
* -t: class configuration file;
* -p: number of partitions;
* -mb: number of mini batch;

### Example without Spark:

From src directory

```
python rnn-no-spark.py -e 10 -hl 5 -i ../dataset/iris.data -t dataset/data.config
```
```
python rnn-speed.py -e 10 -hl 5 -i ../dataset/iris.data -t dataset/data.config
```

### Using minibatch:

data.config

```
python rnn-no-spark.py -e 10 -hl 5 -i ../dataset/iris.data -t ../dataset/data.config -mb 10
```

**For full documentation use -h option**