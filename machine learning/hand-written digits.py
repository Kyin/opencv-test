import numpy as np
import cv2

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# split the image into 5000 20x20 cells
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

# make it into a 50x100 numpy array
x = np.array(cells)

# now we prepare train_data and test data

train = x[:, :50].reshape(-1, 400).astype(np.float32)  # array 2500x400
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # array 2500x400

# create labels
k = np.arange(10)  # k =  [1, 2, 3, 4, 5, 6, 7, 8, 9]
train_labels = np.repeat(k, 250)[:, np.newaxis]  # train_labels = [k, k, k, k, k ..., k]
test_labels = train_labels.copy()  # test_labels = train_labels

# Initiate kNN
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, results, neighbours, dist = knn.findNearest(test, k=5)

# check accuracy
matches = results == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100 / results.size

np.set_printoptions(threshold=np.nan)
print results
print accuracy

# save the data
np.savez('knn_data.npz', train=train, train_labels=train_labels)
