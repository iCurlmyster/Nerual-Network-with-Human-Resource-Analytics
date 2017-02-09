from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import sys

np.random.seed(7)


init_data = pd.read_csv("./HR_comma_sep.csv")
sales_unique_feature_names = init_data["sales"].unique()
salary_unique_feature_names = init_data["salary"].unique()
def break_down_features(feature_list, category, orig_data):
    for name in feature_list:
        orig_data[category+"_"+name] = [1 if x == name else 0 for _, x in enumerate(orig_data[category])]

break_down_features(sales_unique_feature_names, "sales", init_data)
break_down_features(salary_unique_feature_names, "salary", init_data)
# Now that we have added our new categories we can drop our old object ones.

init_data = init_data.drop(["sales", "salary"], axis=1)
def stratified_split_data(data, ratio):
    # Grab the data into its own category
    stayed_data = data.loc[data["left"] == 0]
    left_data = data.loc[data["left"] == 1]
    # mix up the data
    stayed_data = stayed_data.iloc[np.random.permutation(len(stayed_data))]
    left_data = left_data.iloc[np.random.permutation(len(left_data))]
    test_stayed_set_size = int(len(stayed_data) * ratio)
    test_left_set_size = int(len(left_data) * ratio)
    # Concatenate the partitioned data
    train_set = pd.concat([stayed_data[test_stayed_set_size:], left_data[test_left_set_size:]], ignore_index=True)
    test_set = pd.concat([stayed_data[:test_stayed_set_size], left_data[:test_left_set_size]], ignore_index=True)
    # Now mix up the concatenated data
    train_shuffled_indices = np.random.permutation(len(train_set))
    test_shuffled_indices = np.random.permutation(len(test_set))
    return train_set.iloc[train_shuffled_indices], test_set.iloc[test_shuffled_indices]

train_set, test_set = stratified_split_data(init_data, 0.2)
# Now lets make sure we still have the same percentages

data = (train_set.drop("left", axis=1)).values
data_labels = train_set["left"].values
data_labels = data_labels.reshape([len(data_labels), 1])
num_features = data.shape[1]
n_samples = data.shape[0]


model = Sequential()

# allow first layer to use sigmoid. (gives better results)
model.add(Dense(10, input_dim=num_features, init="uniform", activation="relu"))

#perform dropout of neurons at 0.4 threshold
model.add(Dropout(0.2))

# only use relu in middle layer
model.add(Dense(6, init="uniform", activation="sigmoid"))

# also dropout neurons if below 0.3
#model.add(Dropout(0.3))

model.add(Dense(1, init="uniform", activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(data, data_labels, nb_epoch=50, batch_size=10)


print("\nevaluating data that the model was trained on.")
scores = model.evaluate(data, data_labels, batch_size=32)
print("\n{0}: {1}".format(model.metrics_names[1], scores[1]*100))
print("{0}: {1}\n".format(model.metrics_names[0], scores[0]*100))

