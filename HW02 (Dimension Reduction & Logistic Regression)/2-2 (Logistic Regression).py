import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def softmax(t, w):
    ak = np.dot(t, w)
    ak = np.exp(ak)
    ak = ak / np.sum(ak, axis=1, keepdims=True)
    return ak


def plot_learning_curve(info):
    plt.plot(info[:, 0], info[:, 1])


ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

Lambda = 1
learning_rate = 0.6
cross_entropy = 0

df_train = pd.read_csv("kdd99_training_data.csv")
classes = len(df_train["Class"].value_counts())

df_test = pd.read_csv("kdd99_testing_data.csv")

# read source data
train = df_train.values

oneks = np.zeros((np.shape(train)[0], classes))
oneks[np.arange(np.shape(train)[0]), np.asarray(train[:, -1], dtype=int)] = 1

test = df_test.values

# Gradient Descent Algotirhm
print("[Gradient Descent Algotirhm]")

iter_time = 10000
n_samples, n_features = np.shape(train[:, :-1])
w = np.zeros((n_features, classes))
info = np.zeros(iter_time)

# train parameters
for i in range(iter_time):
    phi = train[:, :-1]
    y = softmax(phi, w)

    w = w - learning_rate * np.dot(phi.T, y - oneks)
    cross_entropy = -np.multiply(oneks, np.log(y))
    cross_entropy = - oneks * np.log(y)

    info[i] = np.sum(cross_entropy)

# testing accuracy
y = softmax(test[:, :-1], w)
correct = np.argmax(y, axis=1) == test[:, -1]
print("Gradient Descent miss classification rate ", 1 - np.sum(correct) / np.shape(test)[0])

plt.sca(ax1)
# plot
plt.plot(info)
plt.title("[Gradient Descent] Learning Curve")
plt.xlabel("Iteration Times")
plt.ylabel("Cross Entropy")

# Newton-Raphson Algorithm

print("[Newton-Raphson Algorithm]")

iter_time = 10000
w = np.zeros((n_features, classes))
info = np.zeros(iter_time)

# train parameters
for i in range(iter_time):
    phi = train[:, :-1]
    y = softmax(phi, w)
    dw = np.dot(phi.T, y - oneks)

    for idx in range(classes):
        tmp = y[:, idx]
        tmp = np.dot(tmp, (1 - tmp))
        H = tmp * np.dot(phi.T, phi)
        w[:, idx] = w[:, idx] - Lambda * np.dot(np.linalg.inv(H), dw[:, idx])

    print(H)
    cross_entropy = -np.multiply(oneks, np.log(y))

    info[i] = np.sum(cross_entropy)

# testing accuracy
y = softmax(test[:, :-1], w)
correct = np.argmax(y, axis=1) == test[:, -1]
print("Newton-Raphson miss classification rate ", 1 - np.sum(correct) / np.shape(test)[0])

# plot
plt.sca(ax2)
plt.plot(info)
plt.title("[Newton-Raphson] Learning Curve")
plt.xlabel("Iteration Times")
plt.ylabel("Cross Entropy")

# only need to plot the figure once
plt.show()
