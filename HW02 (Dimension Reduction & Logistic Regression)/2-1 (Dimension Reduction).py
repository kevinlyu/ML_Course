import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


# Lab Version

def plot_scatter(data, title):
    fig = plt.figure()
    ax = Axes3D(fig)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    types = data[:, 3]
    color = ['r', 'g', 'b']

    ax.set_title(title)

    SET = mpatches.Patch(color="red", label="SET")
    VIR = mpatches.Patch(color="green", label="VIR")
    VER = mpatches.Patch(color="blue", label="VER")

    plt.legend(handles=[SET, VIR, VER])
    ax.legend()

    for i in range(np.shape(data)[0]):
        ax.scatter(x[i], y[i], z[i], c=color[(int)(types[i])], marker='o')

    plt.show()


def LDA(X, k):
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])

    norm_X = X - np.asarray([mean])

    S_T = np.dot(norm_X[:, 0:-1].T, norm_X[:, 0:-1])

    C = [X[X[:, -1] == i] for i in range(3)]
    M = [np.mean(c[:, 0:-1], axis=0) for c in C]

    S_W = np.dot((C[0][:, 0:-1] - np.asarray([M[0]])).T, C[0][:, 0:-1] - np.asarray([M[0]])) + np.dot(
        (C[1][:, 0:-1] - np.asarray([M[1]])).T, C[1][:, 0:-1] - np.asarray([M[1]])) + np.dot(
        (C[2][:, 0:-1] - np.asarray([M[2]])).T, C[2][:, 0:-1] - np.asarray([M[2]]))

    S_B = S_T - S_W

    eig_val, eig_vec = np.linalg.eig(np.dot(np.linalg.inv(S_W), S_B))
    # print(np.dot(np.linalg.inv(S_W), S_B))

    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features - 1)]
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X[:, 0:-1], np.transpose(feature))

    return data


def PCA(X, k):  # k is the components you want

    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the first k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))

    return data


# QDA
def g(mean, cov, prior, x):
    Wi = -0.5 * np.linalg.inv(cov)
    wi = np.dot(np.linalg.inv(cov), mean)
    wio = -0.5 * np.log(np.linalg.det(cov)) - 0.5 * np.dot(np.dot(mean.T, np.linalg.inv(cov)), mean) + np.log2(
        prior)

    return np.dot(np.dot(x.T, Wi), x) + np.dot(wi.T, x) + wio


def prediction(r, c, mean, cov, prior, x):
    table = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            table[i][j] = g(mean[j], cov[j], prior[j], x[i][:-1])
    return table


def fill_chart(r, data, table):
    chart = np.zeros((3, 3))

    for i in range(r):
        x = (int)(data[i, -1])
        y = np.where(table[i, :] == np.max(table[i]))[0]
        chart[x][y] += 1
    return chart


def display_result(chart, num):
    print(chart)
    print("Accuracy = ", (chart[0][0] + chart[1][1] + chart[2][2]) / num)


df = pd.read_excel('Irisdat .xls', sheetname='Sheet1')

df["IRISTYPE Three types of iris"] = df["IRISTYPE Three types of iris"].map({'SETOSA': 0, 'VIRGINIC': 1, 'VERSICOL': 2})
data = df.values
train = data[0:120, :]
test = data[120:, :]

C = [train[train[:, -1] == i] for i in range(3)]
mean = [np.mean(c[:, 0:4], axis=0) for c in C]
cov = [np.cov(c[:, 0:-1].T, ddof=0) for c in C]
num_C = [np.shape(c)[0] for c in C]
prior = [c / np.shape(train)[0] for c in num_C]

####################################### Training Stage, QDM #######################################
print("############## QDA ##############")

print("Training")

predict_train = prediction(120, 3, mean, cov, prior, train)
train_chart = fill_chart(120, train, predict_train)
display_result(train_chart, 120)

####################################### Testing Stage #######################################
print("Testing")

predict_test = prediction(30, 3, mean, cov, prior, test)
test_chart = fill_chart(30, test, predict_test)
display_result(test_chart, 30)

##################################### PCA  #####################################
print("############## PCA ##############")
for k in range(1, 4, 1):
    print("k = ", k)

    append = np.reshape(train[:, -1], (120, 1))
    train_pca = np.hstack((PCA(train[:, 0:-1], k), append))

    append = np.reshape(test[:, -1], (30, 1))
    test_pca = np.hstack((PCA(test[:, 0:-1], k), append))

    C = [train_pca[train[:, -1] == i] for i in range(3)]
    mean = [np.mean(c[:, 0:k], axis=0) for c in C]
    cov = [np.cov(c[:, 0:k].T, ddof=0) for c in C]

    if k == 1:
        cov = np.reshape(cov, (3, 1, 1))

    print("Training Stage")
    predict_train = prediction(120, 3, mean, cov, prior, train_pca)
    train_chart = fill_chart(120, train_pca, predict_train)
    display_result(train_chart, 120)

    print("Testing Stage")
    predict_test = prediction(30, 3, mean, cov, prior, test_pca)
    test_chart = fill_chart(30, test_pca, predict_test)
    display_result(test_chart, 30)

    if k == 3:
        plot_scatter(train_pca, "PCA Train")
        plot_scatter(test_pca, "PCA Test")

#################### LDA ####################

print("############## LDA ##############")
for k in range(1, 4, 1):
    print("k = ", k)

    append = np.reshape(train[:, -1], (120, 1))
    train_lda = np.hstack((LDA(train, k), append))

    append = np.reshape(test[:, -1], (30, 1))
    test_lda = np.hstack((LDA(test, k), append))

    C = [train_lda[train[:, -1] == i] for i in range(3)]
    mean = [np.mean(c[:, 0:k], axis=0) for c in C]
    cov = [np.cov(c[:, 0:k].T, ddof=0) for c in C]

    if k == 1:
        cov = np.reshape(cov, (3, 1, 1))

    print("Training Stage")
    predict_train = prediction(120, 3, mean, cov, prior, train_lda)
    train_chart = fill_chart(120, train_lda, predict_train)
    display_result(train_chart, 120)

    print("Testing Stage")
    S = num_C[0] / np.shape(test)[0] * cov[0] + num_C[1] / np.shape(test)[0] * cov[1] + num_C[2] / np.shape(test)[
        0] * cov[2]
    predict_test = prediction(30, 3, mean, cov, prior, test_lda)
    test_chart = fill_chart(30, test_lda, predict_test)
    display_result(test_chart, 30)

    if k == 3:
        plot_scatter(train_lda, "LDA Train")
        plot_scatter(test_lda, "LDA Test")
