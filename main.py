import tensorflow as tf
import pandas as pd
import numpy as np
import sys


init_data = pd.read_csv("./HR_comma_sep.csv")

# take a look at the info
# we see no rows are missing in any column. We also see "sales" and "salary" are object types.
print(init_data.info())

# lets take a look at the first 5 entries of object data
print("Sales: {0}".format(init_data["sales"][:5]))
print("Salary: {0}".format(init_data["salary"][:5]))

# We can see that sales looks like nominal data and salary looks like ordinal data
# Lets see how many unique entries there are
sales_unique_n = init_data["sales"].nunique()
salary_unique_n = init_data["salary"].nunique()

## 10 unique
print("Unique sale categories: {0}".format(sales_unique_n))
## 3 unique
print("Unique salary categories: {0}".format(salary_unique_n))

# Now lets break these categories out into easily digestible features for our model
sales_unique_feature_names = init_data["sales"].unique()
salary_unique_feature_names = init_data["salary"].unique()

# Function to breakdown a category into individual binary features
def break_down_features(feature_list, category, orig_data):
    for name in feature_list:
        orig_data[category+"_"+name] = [1 if x == name else 0 for _, x in enumerate(orig_data[category])]

break_down_features(sales_unique_feature_names, "sales", init_data)
break_down_features(salary_unique_feature_names, "salary", init_data)

# Now that we have added our new categories we can drop our old object ones.
init_data = init_data.drop(["sales", "salary"], axis=1)

# Lets look at the first 5 entries of what these float values are in are data
print(init_data["satisfaction_level"][:5])
print(init_data["last_evaluation"][:5])

# They seem to be just percentages but lets make sure
## 1.0
print("Max value satisfaction_level: {0}".format(init_data["satisfaction_level"].max()))
## 0.0899
print("Min value satisfaction_level: {0}".format(init_data["satisfaction_level"].min()))
## 1.0
print("Max value last_evaluation: {0}".format(init_data["last_evaluation"].max()))
## 0.3599
print("Min value last_evaluation: {0}".format(init_data["last_evaluation"].min()))

# So the values are bounded between 0 and 1


# Now lets create our split method and split the data
def split_data(data, ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_data(init_data, 0.2)





