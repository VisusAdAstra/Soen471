# BIG DATA

## Topic

Predictive Maintenance: Failure Prediction

## Abstract

Our goal is to predict when different components of CNC machines will fail. We use time-series telemetry data obtained from sensors which include failure history, maintenance history, machine conditions and features such as the machine’s age and model. The dataset was published on [GitHub](https://github.com/DeeptiChevvuri/Predictive-Maintenance-Modelling-Datasets) in 2018. It is around 77 Mb with 876100 samples of telemetry values for 100 machines for 1 year on an hourly basis. We will perform EDA and feature engineering to eventually implement 3 algorithms: decision trees, random forests and GBTClassifier. We will then evaluate the results based on different metrics such as accuracy, recall and F1 scores.

## I. Introduction

## II. Materials and Methods

### Technologies

* <b>Apache Spark/PySpark: </b> 3.0.0+ - It is a unified analytics engine for large-scale data processing. We will use Dataframe API and Resilient Distributed Dataset (RDD) for data processing and parallelization.
* <b>Pandas: </b> 1.2.0+ - We will use Pandas for data manipulation, analysis, sorting, handling missing values, cleaning, and visualization.
* <b>Scikit-Learn: </b> 0.24.1+ Software machine learning library for the Python programming language. We will use it for data analysis and utilize isolation forest from its sklearn.ensemble module.
* <b>PySpark ML: </b> It is DataFrame-based machine learning APIs to let users quickly assemble and configure practical machine learning pipelines.
* <b>Python/Pytest: </b> 3.6+ - Programming language and test framework.

### Algorithms

## Installation 
Required pakages:

- Python version 3.6+
- `numpy` version 2.0 or later: https://www.numpy.org/
- `scikit-learn` version 0.23 or later: https://scikit-learn.org
- `spark` version 3.1.1 or later: https://spark.apache.org/
...

## Usage

...

### Readme

```
$ conda install numpy scipy matplotlib scikit-learn ipython-notebook seaborn

$ jupyter notebook --notebook-dir='<project folder>'
```

