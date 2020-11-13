from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_digits

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

plt.rcParams.update({'font.size': 12})

data = pd.read_csv("featuresEmeralds.csv")
data = data.sample(frac=1, random_state=42)

index = dataset[dataset.columns[0:1]]
label = dataset[dataset.columns[dataset.shape[1]-1:dataset.shape[1]]]

mean = data.mean()
std = data.std()
norm = (data - data.mean())/data.std()
normalizado = pd.concat([norm, label], axis=1)

X = norm
y = label

# Load data
digits = load_digits()

# Create feature matrix and target vector
#X, y = digits.data, digits.target

# Create range of values for parameter
param_range = np.arange(1, 200)  # 100


# Calculate accuracy on training and test set using range of parameter values

train_scores2, test_scores2 = validation_curve(RandomForestClassifier(max_depth=5),
                                               X,
                                               y,
                                               param_name="n_estimators",
                                               param_range=param_range,
                                               cv=4,
                                               scoring="accuracy",
                                               n_jobs=-1)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

train_mean2 = np.mean(train_scores2, axis=1)
train_std2 = np.std(train_scores2, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean2 = np.mean(test_scores2, axis=1)
test_std2 = np.std(test_scores2, axis=1)

train_mean3 = np.mean(train_scores3, axis=1)
train_std3 = np.std(train_scores3, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean3 = np.mean(test_scores3, axis=1)
test_std3 = np.std(test_scores3, axis=1)

# Plot mean accuracy scores for training and test sets

# Plot accurancy bands for training and test sets
#plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
#plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

print(stats.ttest_ind(test_scores[298], test_scores2[98], equal_var=False))
print(stats.ttest_ind(test_scores[298], test_scores3[98], equal_var=False))
print(stats.ttest_ind(test_scores2[98], test_scores3[98], equal_var=False))


# Create plot

fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)


ax3.plot(param_range3, train_mean3, label="Training accuracy", color="gray")
ax3.plot(param_range3, test_mean3, label="Validation accuracy", color="purple")

# Major ticks every 20, minor ticks every 5
major_ticks0 = np.arange(0, 301, 50)
minor_ticks0 = np.arange(0, 301, 10)
major_ticks = np.arange(0, 201, 20)
minor_ticks = np.arange(0, 201, 5)
major_ticks2 = np.arange(0, 1.1, .2)
minor_ticks2 = np.arange(0, 1.1, .1)
major_ticks3 = np.arange(0, 1.1, .2)
minor_ticks3 = np.arange(0, 1.1, .1)

ax.set_xticks(major_ticks0)
ax.set_xticks(minor_ticks0, minor=True)
ax.set_yticks(major_ticks3)
ax.set_yticks(minor_ticks3, minor=True)


ax2.set_xticks(major_ticks)
ax2.set_xticks(minor_ticks, minor=True)
ax2.set_yticks(major_ticks2)
ax2.set_yticks(minor_ticks2, minor=True)

ax3.set_xticks(major_ticks)
ax3.set_xticks(minor_ticks, minor=True)
ax3.set_yticks(major_ticks2)
ax3.set_yticks(minor_ticks2, minor=True)


# And a corresponding grid

ax3.grid(which='both')
ax3.grid(which='minor', alpha=0.2)
ax3.grid(which='major', alpha=0.5)


ax3.set_title(
    "Extremely Randomized Trees \n training and validation accuracy curve")
ax3.set_xlabel("Number of trees")
ax3.set_ylabel("Accuracy")

ax3.legend(loc='lower right')

# plt.tight_layout()
plt.show()
