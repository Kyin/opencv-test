import numpy as np
import cv2
# local modules
from common import clock, mosaic

SIZE = 20
affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
bin_n = 16  # number of bins
CLASS_N = 10


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err) * 100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[..., :2] = 0

        vis.append(img)
    return mosaic(25, vis)

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells


def load_digits(fn):
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (SIZE, SIZE))
    labels = np.repeat(np.arange(CLASS_N), len(digits) / CLASS_N)
    return digits, labels


# We want to deskew the images to help the algorith
def deskew(image):
    # second order moment ??
    m = cv2.moments(image)  # We can use the moments of the image as implemented in OpenCV

    if abs(m['mu02']) < 1e-2:
        return image.copy()  # No deskewing needed

    skew = m['mu11'] / m['mu02']  # The measure of skewness is given by ration of the 2 central moments :
    # mu11 and mu02

    m = np.float32([[1, skew, -0.5 * SIZE * skew], [0, 1, 0]])  # Calculate correction
    image = cv2.warpAffine(image, m, (SIZE, SIZE), flags=affine_flags)  # Apply correction
    return image


# Histogram of Oriented Gradients (HOG) descriptor
# Converts a greyscale image to a feature vector
def hog(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantiziong binvalues in (0....16)
    bins = np.int32(bin_n * ang / (2 * np.pi))

    # Divide into 4 subsquares
    bin_cells = bins[:10, :10], bins[:10, :10], bins[:10, :10], bins[:10, :10]

    # magnitude cells weight bin cells
    mag_cells = mag[:10, :10], mag[:10, :10], mag[:10, :10], mag[:10, :10]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]

    hist = np.hstack(hists)

    return hist


#
# STEP 1 : LOAD AND PREPROCESS IMAGE
#
digits, labels = load_digits("digits.png")
#img = cv2.imread('digits.png', 0)

# split image into cells
#cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

# split data into train an test

#train_cells = [i[:50] for i in cells]
#test_cells = [i[50:] for i in cells]

#
# STEP 2 : CALCULATE HISTOGRAM OF ORIENTED GRADIENTS (HOG) DESCRIPTOR
#

# (Converts a greyscale image to a feature vector)

winSize = (20, 20)  # The size of the digit images is 20x20
blockSize = (10, 10)
blockStride = (5, 5)  # Overlap betwee, neighboring blocks. Degree of contrast normalization. Typically 50% of blockSize

cellSize = (10, 10)  # The size of a feature. Can be changed.
nbins = 9  # Number of bins in the histogram of gradients. A value of 9 is recommended by HOG authors. can change
deriveAperture = 1  # don't change
winSigma = -1  # don't change
histogramNormType = 0  # don't change
L2HysThreshold = 0.2  # don't change
gammaCorrection = 1  # don't change
nLevels = 64  # don't change
signedGradients = True  # can change

# We can use the OpenCV implementation of HogDescriptor instead of building our own
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, deriveAperture, winSigma
                        , histogramNormType, L2HysThreshold, gammaCorrection, nLevels, signedGradients)

# prepare training data
deskewed = list(map(deskew, digits))

# Compute feature decriptor
hog_descriptor = []
for image in deskewed:
    hog_descriptor.append(hog.compute(image))
hog_descriptor = np.squeeze(hog_descriptor)

# split into training data (90%) and test set (10%)

train_n = int(0.9 * len(hog_descriptor))  # 90% des descripteurs
digits_train, digits_test = np.split(deskewed, [train_n])
hog_descriptor_train, hog_descriptor_test = np.split(hog_descriptor, [train_n])

# because we have only digits, we can label the training data automatically. Usually, it needs to be labelled by hand

labels_train, labels_test = np.split(labels, [train_n])

#
# STEP 3 : TRAIN A MODEL
#

# Set up SVM
svm = cv2.ml.SVM_create()

# Set SVM type
svm.setType(cv2.ml.SVM_C_SVC)

# Set SVM Kernel
svm.setKernel(cv2.ml.SVM_RBF)  # RBF = Radial Basis Function

# THE TWO FOLLOWING PARAMETERS NEEDS TO BE ADJUSTED TO THE TASK

# Set parameter C. A smaller C gives better separation but some points may be missclassified
svm.setC(12.5)

# Set Gamma. Gamma controls the stretching of the parameters in an extra dimension.
svm.setGamma(0.50625)

# train svm and save data
svm.train(hog_descriptor_train, cv2.ml.ROW_SAMPLE, labels_train)
svm.save('svm_data.data')

# Evaluate model

vis = evaluate_model(svm, digits_test, hog_descriptor_test, labels_test)
cv2.imwrite("digit-classification.jpg", vis)
cv2.imshow('Vis', vis)
cv2.waitKey(0)

