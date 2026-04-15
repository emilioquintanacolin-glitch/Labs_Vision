import cv2
img = cv2.imread('borpa.png')
cv2.imshow('borpa',img)
cv2.waitKey(0)

import matplotlib.pyplot as plt
img = cv2.imread('borpa.png')
plt.imshow(img)
plt.show()