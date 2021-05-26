import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=10, figsize=(20, 3))

for axis, image, label in zip(axes, digits.images, digits.target):
    axis.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    axis.set_title(f'Training: {label}')

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

max_performance = ___
for gamma in range(0.0001, 1, .0001):
    classifier = svm.SVC(gamma=0.00001)

    x_train, x_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.9, shuffle=True,
    )

    classifier.fit(x_train, y_train)

    predicted = classifier.predict(x_test)

    _, axes = plt.subplots(nrows=1, ncols=10, figsize=(20, 3))
    for axis, image, prediction in zip(axes, x_test, predicted):
        image = image.reshape(8, 8)
        axis.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        axis.set_title(f'Prediction: {prediction}')

    print(f'Classification report for classifier {classifier}')
    print(f'{metrics.classification_report(y_test, predicted)}')
    print()

    disp = metrics.plot_confusion_matrix(classifier, x_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")

    plt.show()
