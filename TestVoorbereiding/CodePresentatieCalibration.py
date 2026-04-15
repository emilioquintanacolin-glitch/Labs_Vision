import cv2
from datetime import datetime

current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
print("Current date & time : ", current_datetime)


img_name = "borpa.png"
img = cv2.imread(img_name)

#Flip image
#img_h = cv2.flip(img, 1)
#img_v = cv2.flip(img, 0)

#Rotate image
#img_cw_180 = cv2.rotate(img, cv2.ROTATE_180)

#Convert to color image
#color_img = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2RGB)
#color_img = cv2.circle(color_img, (10, 10), 5, (0, 0, 255), 1)
cv2.putText(img, "text", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2, cv2.LINE_AA)


cv2.imshow('borpa',img)
cv2.waitKey(0)

file_name = "test result "+current_datetime+".jpg"
cv2.imwrite(file_name,img)

'''
import matplotlib.pyplot as plt
img = cv2.imread('borpa.png')
plt.imshow(img)
plt.show()
'''