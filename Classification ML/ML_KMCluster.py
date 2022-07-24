import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits_df = load_digits()
data_df = scale(digits_df.data)  # scales the rgb values in digits_df to -1 tp +1
# print(digits_df.data, '\n', data_df)

y = digits_df.target
k = len(np.unique(y))


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


model = KMeans(n_clusters=k, init='k-means++', n_init=10)
bench_k_means(model, '1', data_df)
