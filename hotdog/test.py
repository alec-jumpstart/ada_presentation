import os
import time

from fastai.imports import plt
from matplotlib.pyplot import plt

hotdogs = os.listdir('./data/testing/hotdog')[:10]
for i in range(5):
    img = plt.imread(f'./data/testing/hotdog/{hotdogs[i]}')
    plt.imshow(img)
    plt.show()


not_hotdogs = os.listdir('./data/testing/not_hotdog')[:10]
for i in range(5):
    img = plt.imread(f'./data/testing/not_hotdog/{not_hotdogs[i]}')
    plt.imshow(img)
    plt.show()
