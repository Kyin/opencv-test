import numpy as np
import cv2

data = np.loadtxt('letter-recognition.data', dtype='float32', delimiter=',',
                  converters={0: lambda ch: ord(ch) - ord('A')})

# split the data in 2, 1000 values for train an test
train, test = np.vsplit(data, 2)

# split trainData and testData to feature and response
responses, trainData = np.hsplit(train, [1])
labels, testData = np.hsplit(test, [1])

# Initiate the kNN, classify
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

ret, results, neighbours, dist = knn.findNearest(testData, k=5)

# measure accuracy
correct = np.count_nonzero(results == labels)
accuracy = correct * 100 / results.size
print accuracy
