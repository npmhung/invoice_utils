import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def show_img(img, figsize=None, cmap='gray'):
    plt.figure(figsize=figsize)
    plt.show(plt.imshow(img, cmap=cmap))

def load_img(path, to_gray=True):
    img = cv.imread(path)
    if to_gray:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def draw_mask(boxes, shape):
    tmp = np.zeros(shape)
    cv.fillPoly(tmp, np.array([np.array(b, dtype=np.int32).reshape(-1,2) for b in boxes]),1)
    return tmp.astype(np.uint8)