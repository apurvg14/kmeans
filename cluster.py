# Numpy import
import numpy as np

# Other utility imports
import argparse
import copy
import itertools
import matplotlib.pyplot as plt
import random

colors = ["red", "green", "blue", "pink", "brown", "magenta", "black", "gold", "cyan", "hotpink", "lime"]

'''
Parser arguments
'''
parser = argparse.ArgumentParser(description='Info about cluster.py parameters.')
parser.add_argument('-p', '--norm_number', help='Specify the norm that you want to use.', required=False, default=2)
parser.add_argument('-k', '--num_clusters', help='Specify the number of clusters you want to make.', required=False,
                    default=3)
parser.add_argument('-m', '--data_transform', help='Specify the kind of data transform you want to apply.',
                    required=False, choices=["unchanged", "normalised", "standardised"], default="unchanged")
parser.add_argument('-i', '--init_mode', help='Specify the kind of cluster initialisation you want to use.',
                    required=False, choices=["plusplus", "forgy"], default="plusplus")
parser.add_argument('-s', '--perform_silhouette', help='Use Silhouette score test to find optimal number of clusters.',
                    action='store_true', required=False)

args = vars(parser.parse_args())
print("Chosen arguments:", args)
norm_number = args["norm_number"]
num_clusters = args["num_clusters"]
init_mode = args["init_mode"]
perform_silhouette = args["perform_silhouette"]

'''
Set random seed, reproducible tag will ensure that the same results are generated each time
'''
seed_init = 0
reproducible = True


# Standardise the 2D array by subtracting mean and dividing by standard deviation, column wise
def standardise_ndarray(a):
    return (a - np.mean(a, axis=0)) / np.std(a, axis=0)


# Normalise the 2D array by dividing by sum to bring values between 0 and 1, column wise
def normalise_ndarray(a):
    col_sums = a.sum(axis=0)
    return a / col_sums[np.newaxis, :]


# Custom prediction function for k-means
def kmeans_predict(instance, means):
    global norm_number
    return np.argmin([np.linalg.norm((i - instance), ord=norm_number) for i in means])


# Function to return clusters obtained from kmeansplusplus initialisation
def plus_plus(data):
    global norm_number
    global num_clusters
    if reproducible:
        random.seed(seed_init)
    new_data = copy.deepcopy(data)
    cluster_means = []
    latest_cluster_idx = random.choices(np.arange(new_data.shape[0]), k=1)[0]
    latest_cluster = new_data[latest_cluster_idx]
    cluster_means.append(latest_cluster)
    new_data = np.delete(new_data, latest_cluster_idx, 0)
    for k_n in range(num_clusters - 1):
        distances = [min([np.linalg.norm((cluster - i), ord=norm_number) for cluster in cluster_means]) for i in
                     new_data]
        latest_cluster_idx = random.choices(np.arange(new_data.shape[0]), k=1, weights=distances)[0]
        latest_cluster = new_data[latest_cluster_idx]
        cluster_means.append(latest_cluster)
        new_data = np.delete(new_data, latest_cluster_idx, 0)

    return cluster_means


'''
The following function permutes the prediction labels till the maximum accuracy is achieved.
Note that this doesnt change the train model or its predictions. Since kmeans is an unsupervised algorithm,
we can rename cluster 1 to cluster 3, cluster 2 to cluster 1, and cluster 3 to cluster 2, and so on.
Since accuracy depends on the exact cluster numbering, the following function tries all such numberings, or
permutations, to find out which gets the max accuracy. 
'''


def get_max_acc(curr_predictions, ground_truth):
    global num_clusters
    preds = curr_predictions
    if reproducible:
        random.seed(seed_init)
    max_acc = 0
    perm_list = list(itertools.permutations(np.arange(num_clusters)))
    new_pred = curr_predictions
    for perm in perm_list:
        for i in range(len(perm)):
            new_pred[curr_predictions == i] = perm[i]
        acc = ((np.array(ground_truth) == np.array(new_pred)).astype(int)).mean()
        if acc > max_acc:
            max_acc = acc
            preds = new_pred
    return preds, max_acc


# Function to implement k-means clustering with custom norm and custom initialisation.
def normed_kmeans(x_train):
    global y_train_num
    global norm_number
    global num_clusters
    global reproducible
    global init_mode

    cluster_means = []
    if reproducible:
        random.seed(seed_init)
    if init_mode == "forgy":
        idx = random.choices(range(x_train.shape[0]), k=num_clusters)
        cluster_means = x_train[idx, :]
    elif init_mode == "plusplus":
        cluster_means = plus_plus(x_train)

    error = 1
    while error != 0:
        new_means = np.zeros_like(cluster_means)
        num_points = np.array([0] * num_clusters)
        for instance in x_train:
            assigned_cluster = kmeans_predict(instance, cluster_means)
            new_means[assigned_cluster] += instance
            num_points[assigned_cluster] += 1

        new_means = np.array([new_means[i] / num_points[i] for i in range(len(num_points))])
        error = np.linalg.norm(new_means - cluster_means)
        cluster_means = new_means
    return cluster_means


# Function to find the silhouette score for a given number of cluster means 'k' for data
def silhouette_score(data, k):
    global norm_number
    global init_mode
    cluster_means = normed_kmeans(data)
    predictions = [kmeans_predict(i, cluster_means) for i in data]
    silhouette_scores = []
    for i in range(data.shape[0]):
        distances_same_cluster = [np.linalg.norm((data[i] - data[j]), ord=norm_number) for j in range(len(predictions))
                                  if (predictions[j] == predictions[i]) and (i != j)]
        a_i = 0.0
        if len(distances_same_cluster):
            a_i = np.mean(distances_same_cluster)
        min_val = np.inf
        for cluster in range(k):
            if predictions[i] == cluster:
                continue
            distances = [np.linalg.norm((data[i] - data[j]), ord=norm_number) for j in range(len(predictions)) if
                         predictions[j] == cluster]
            min_val = min(min_val, np.mean(distances))

        num_same_cluster = len(distances_same_cluster) + 1
        if num_same_cluster == 1:
            silhouette_scores.append(0)
        else:
            silhouette_scores.append((min_val - a_i) / max(min_val, a_i))
    return np.mean(silhouette_scores)


# Load the csv files using Numpy
train_data = np.loadtxt("./data/iris_train.csv", delimiter=",", dtype="str")
test_data = np.loadtxt("./data/iris_test.csv", delimiter=",", dtype="str")

# Divide the data into features (x) and labels (y) and convert all value to Numpy floats
x_train, y_train = train_data[:, 0:4].astype(np.float), train_data[:, 4]
x_test, y_test = test_data[:, 0:4].astype(np.float), test_data[:, 4]

# Create label dictionary
name_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_train_num = [name_dict[i] for i in y_train]
y_test_num = [name_dict[i] for i in y_test]

classes = list(set(y_train))
num_clusters_original = len(classes)
print("Number of labels in the train set:", num_clusters_original)

# Apply chosen transform on data
if args["data_transform"] == "normalised":
    x_train = normalise_ndarray(x_train)
    x_test = normalise_ndarray(x_test)
elif args["data_transform"] == "standardised":
    x_train = standardise_ndarray(x_train)
    x_test = standardise_ndarray(x_test)

# Get the cluster means
cluster_means = normed_kmeans(x_train)

# Predict on test data
predictions = [kmeans_predict(i, cluster_means) for i in x_test]
predictions_test, accuracy_test = get_max_acc(predictions, y_test_num)
print("Test accuracy by taking", args["data_transform"], "data:", accuracy_test, "using", num_clusters, "clusters and", norm_number, "norm.")

# Predict on train data
predictions = [kmeans_predict(i, cluster_means) for i in x_train]
predictions_train, accuracy_train = get_max_acc(predictions, y_train_num)
print("Train accuracy by taking", args["data_transform"], "data:", accuracy_test, "using", num_clusters, "clusters and", norm_number, "norm.")

# Plot the kmeans clustering results along with cluster means, for first two features
cluster_labels = {'cluster 1': 0, 'cluster 2': 1, 'cluster 3': 2}
for i in range(len(cluster_means)):
    plt.scatter(cluster_means[i][0], cluster_means[i][1], color=colors[i], label=cluster_labels.keys()[i], marker="^", s=20)

for i in x_test_standardised:
    plt.scatter(i[0], i[1], color=colors[kmeans_predict(i, cluster_means)], label=cluster_labels.keys()[i],s=5)

plt.xlabel('First feature', fontsize = 18)
plt.ylabel('Second feature', fontsize = 18)
plt.title('Cluster labels for points')
plt.legend(bbox_to_anchor=(0.0, 1), loc='upper left', fontsize = 12, ncol=3)
plt.rc('grid', linestyle="dotted", color='gray')
plt.grid(True)
plt.savefig('./outputs/feature12_norm' + str(norm_number) + '_clusters' + num_clusters + '_init' + init_mode + '_' + args["data_transform"] + '_output.jpg', format='jpg', dpi=600)

print("Plot saved in the outputs folder. :)")

if perform_silhouette:
    print("Performing silhouette score analysis to determine best k value for k-means...")
    y_vals = [silhouette_score(x_train, k) for k in range(2, 11)]
    plt.plot(np.arange(2, 11), y_vals)
    print("Plot saved in the outputs folder. :)")

