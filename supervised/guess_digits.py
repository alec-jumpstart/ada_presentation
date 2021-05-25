# Import datasets, classifiers and performance metrics
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# load the MNIST dataset
digits = datasets.load_digits()

# create a 1x10 grid of numbers to show examples of training data
_, axes = plt.subplots(nrows=1, ncols=10, figsize=(20, 3))

# plots the images by converting values between 0 and 1 to grayscale boxes
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Training: {label}')

# flatten the images (turns each of them from an 8x8 grid to a 64x1 list)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a model (in this case, a support vector classifier)
# gamma controls how sensitive the classifier is to changes
classifier = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
x_train, x_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=True)

# Learn the digits on the train subset
classifier.fit(x_train, y_train)

# Predict the value of the digit on the test subset
predicted = classifier.predict(x_test)

_, axes = plt.subplots(nrows=1, ncols=10, figsize=(20, 3))
for ax, image, prediction in zip(axes, x_test, predicted):
    # turn the image back into an 8x8 matrix
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

# the classification report shows how good our classifier was
print(f"Classification report for classifier {classifier}:")
print(f"{metrics.classification_report(y_test, predicted)}")
print()

# the confusions matrix shows how often our classifier thought
# each number was another number
disp = metrics.plot_confusion_matrix(classifier, x_test, y_test)
disp.figure_.suptitle("Confusion Matrix")

plt.show()
