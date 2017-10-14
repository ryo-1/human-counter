import os, glob
import numpy as np
from sklearn import cross_validation
from keras.preprocessing.image import load_img, img_to_array

root_dir = "./image/"
categories = ["red_apple", "green_apple"]
nb_classes = len(categories)
image_size = 32

X = []
Y = []
for idx, cat in enumerate(categories):
    files = glob.glob(root_dir + "/" + cat + "/*")
    print("---", cat, "‚ğˆ—’†")
    for i, f in enumerate(files):
        img = load_img(f, target_size=(image_size,image_size))
        data = img_to_array(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./image/apple.npy", xy)
print("ok,", len(Y))