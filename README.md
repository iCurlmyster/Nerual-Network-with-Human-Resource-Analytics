# Nerual-Network-with-Human-Resource-Analytics
Creating a neural network in tensorflow to predict whether an employee will leave or not based on a Human Resource Analytics data set. The original data set can be found [here](https://www.kaggle.com/ludobenistant/hr-analytics).
To view a walkthrough of analyzing the data and creating and training the model you can view the [markdown python notebook](https://github.com/iCurlmyster/Nerual-Network-with-Human-Resource-Analytics/blob/master/Neural%20Network%20with%20Human%20Resource%20Analytics/2017-02-15-jm-predict-employee-leaving.md) or you can view it on my [Kaggle profile](https://www.kaggle.com/icurlmyster/d/ludobenistant/hr-analytics/tf-neural-net-with-hr-analytics).

### Neural net results
The neural network has 2 hidden layers using Elu activation functions and one output layer using sigmoid activation function. When fully trained its metrics are:
- Precision of ~92% 
- Recall of ~92%
- F1 Score of ~92%
- Total Accuracy of ~96%.

The Test set has comparable results.

### Run program
Main file is [main.py](https://github.com/iCurlmyster/Nerual-Network-with-Human-Resource-Analytics/blob/master/main.py).

This file accepts optional command line arguments.

- '-v' for verbose output of information about the data
- '-plot' for plotting the loss values in a graph
- '-relu' for testing the model with the 2 hidden layers as Relu activation functions
- '-full' for testing the model with full batch training

#### Requirements
- python3 (you might can run with python 2)
- tensorflow
- pandas
- numpy
- matplotlib
