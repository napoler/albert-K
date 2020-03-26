import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans,kmeans_predict
import os
# set device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# data
data_size, dims, num_clusters = 10000, 2, 100
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)


print('x',x )
# kmeans
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
)

print('cluster_ids_x',cluster_ids_x)
print("cluster_centers",cluster_centers)

output_dir='./'
torch.save(cluster_centers, os.path.join(output_dir, 'pytroch_model.bin'))


y = np.random.randn(5, dims) / 6
y = torch.from_numpy(y)



# predict cluster ids for y
cluster_ids_y = kmeans_predict(
    y, cluster_centers, 'euclidean', device=device
)

print("cluster_ids_y",cluster_ids_y)














# plot
plt.figure(figsize=(4, 3), dpi=160)
plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x, cmap='cool')
plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1],
    c='white',
    alpha=0.6,
    edgecolors='black',
    linewidths=2
)
plt.axis([-1, 1, -1, 1])
plt.tight_layout()
plt.show()