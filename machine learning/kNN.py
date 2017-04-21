import cv2
import numpy as np
import matplotlib.pyplot as plt

# feature set : 25 tuples of (x, y) values of KNOWN date
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

# Labels each one with 0 (red) or 1(blue)
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

# Plot red family
red = trainData[responses.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')

# Plot blue family
blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

# Here comes a new challenger
newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')

# Initialize kNN algorithm
knn = cv2.ml.KNearest_create();
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 5)

print "result :", results, "\n"
print "neighbours: ", neighbours, "\n"
print "distance: ", dist

plt.show()
