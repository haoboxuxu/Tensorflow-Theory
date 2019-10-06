from skimage import io
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('test_img.jpg')

rows = image.shape[0]
cols = image.shape[1]

print(rows)
print(cols)

image = image.reshape(image.shape[0] * image.shape[1], 3)
kmeans = KMeans(n_clusters=128, n_init=10, max_iter=200)
kmeans.fit(image)

print('finish1')

clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
labels = np.asarray(kmeans.labels_, dtype=np.uint8)
labels = labels.reshape(rows, cols)

print('finish2')

print(clusters.shape)
np.save('codebook_test.npy', clusters)
io.imsave('zip_testimg.jpg', labels)

print('finish3')

image = io.imread('compressed_test.jpg')
io.imshow(image)
io.show()

print('finish4')


