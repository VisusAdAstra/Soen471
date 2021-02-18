# BIG DATA

## Topic

Predictive Maintenance: Failure Prediction

## Abstract

Our goal is to predict when different components of CNC machines will fail. We use time-series telemetry data obtained from sensors which include failure history, maintenance history, machine conditions and features such as the machine’s age and model. The dataset was published on [GitHub](https://github.com/DeeptiChevvuri/Predictive-Maintenance-Modelling-Datasets) in 2018 with 5 csv files and is around 77 Mb. We will perform EDA and feature engineering to eventually implement 3 algorithms: decision trees, random forests and GBTClassifier. We will then evaluate the results based on different metrics such as accuracy, recall and F1 scores.

## I. Introduction

Machines needs regular servicing and maintenance. In a manufacturing factory, we want to avoid failure as much as we can. To do so, we can predict when a particular machine is likely to fail before it actually fails. From automated data capture softwares, multiple data on the machines are obtained like the error logs, machine configurations and telemetry like voltage or pressure. From a history of captured data and failures, we can learn from it to predict upcoming failures.

Our objective is to be able to predict which component of which machine will fail and when the latter will fail. We plan to use various machine learning algorithms to achieve the best results. To get a better grasp of the data, we will first perform an Explorary Data Analysis to have an idea of the data’s balance. It might be helpful to find out what are the most determining factors in our tasks. The presentation of the problem is a time series one. We will train on the least recent data and will test on the most recent data. 

Our success criteria would be to achieve a high predictive rate of 90% and to have discovered meaningful properties on the machines during EDA. Related work has been done by the Sogeti team on GitHub with R. They created the predictive model later hosted it in Azure Cloud. More information on their work can be found [here](https://github.com/DeeptiChevvuri/Predictive-Maintenance-Modelling-Datasets), which includes a presentation on the data.

## II. Materials and Methods

### Datasets

The dataset is around 77 Mb with 876100 samples of telemetry values for 100 machines for 1 year on an hourly basis. It includes 5 files: errors.csv, failures.csv, machines.csv, maint.csv and telemetry.csv. All of the files include a datetime and machine ID column except machines.csv which does not have datetime. Individually,

1. errors.csv contains the error ID
2. failures.csv contains the component failing
3. machines.csv contains the model and age
4. maint.csv contains the component maintained
5. telemetry.csv contains the voltage, rotation, pressure and vibration.

### Technologies

* <b>Apache Spark/PySpark: </b> 3.0.0+ - It is a unified analytics engine for large-scale data processing. We will use Dataframe API and Resilient Distributed Dataset (RDD) for data processing and parallelization.
* <b>Pandas: </b> 1.2.0+ - We will use Pandas for data manipulation, analysis, sorting, handling missing values, cleaning, and visualization.
* <b>PySpark ML: </b> It is DataFrame-based machine learning APIs to let users quickly assemble and configure practical machine learning pipelines.
* <b>Scikit-Learn: </b> 0.24.1+ - Machine learning library for the Python programming language. We will use it for data analysis and utilize isolation forest from its sklearn.ensemble module.
* <b>XGBoost: </b> 1.3.3+ - eXtreme Gradient Boosting, provides a parallel tree boosting (also known as GBDT, GBM). XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.

### Algorithms
Traditionally, we make use of Decision Tree, Random Forest and GBT Classifier which are native tools of PySpark ML library. However, due to many articles about their limitation, we will also experiment Python native library XGBoost: XGB Classifier and Isolation Forest Classifier from Scikit-Learn to retrieve the best training scenario.

* <b>Feature Engineering:</b> Perform rolling computation for various features, create time windows and aggregate statistics.

  * <b>Lag features: </b> Divide the duration of data collection into time units where each record belongs to a single point in time for each asset. Once we set the frequency of observations, we want to look for trends within measures, over time/rolling windows in order to predict performance degradation. We create features for these trends within each record using time lags over previous observations to check for these performance changes. 
  
       <p align="center"><img src="https://github.com/VisusAdAstra/Soen471/blob/master/images/window1.png" width="500"></p>

       <p align="center"><img src="https://github.com/VisusAdAstra/Soen471/blob/master/images/window2.png" width="500"></p>
  
  * <b>Label construction: </b> Label all observation cycles within the failure warning window as failed. The prediction problem then becomes estimating the probability of failure within this window.

* <b>Model Training:</b> The purpose is to predict maintenance within the next N cycles and the failure type. Decision Trees are among the most popular and versatile classification methods, used typically for this kind of tasks. Predictive analytics algorithms try to achieve the lowest error possible by either using “boosting” or “bagging”.

  * <b>Random Forest Classifier: </b> Random Forest uses bagging mechanism.The individual decision tree might be “weak learners,” multiple trees reduce the variance and bias of a smaller set or single tree.
  
  * <b>GBT Classifier: </b> The Gradient Boosted Model produces a prediction model composed of an ensemble of decision trees. Each new tree helps to correct errors made by the previously trained tree.
  
  * <b>Isolation Forest Classifier: </b> Isolation Forest is similar in principle to Random Forest, however, identifies anomalies or outliers rather than profiling normal data points. The idea being that isolated observations, or anomalies, are easier to isolate because there are fewer conditions necessary to distinguish them from the normal cases.
  
  * <b>XGBClassifier: </b> Xgboost does regularization of the tree as well to avoid overfitting and deals with the missing values efficiently. Xgboost implementation enhances various parts make a big difference in performance speed, memory efficience and parallelization.

* <b>Evaluation:</b> Confusion matrix, precision, recall and F1 score. Enhancement is conducted by Hyper-Parameter Tuning & Cross Validation.

* <b>Streaming:</b> Apache Spark Streaming enables scalable, high-throughput, fault-tolerant stream processing of live data streams, creating self-updated model.

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

