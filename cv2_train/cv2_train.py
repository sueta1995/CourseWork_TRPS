import cv2
import numpy as np

# img = cv2.imread('images/viridis.jpg')
#
# img = cv2.resize(img, (img.shape[1], img.shape[0]))
# img = cv2.GaussianBlur(img, (9, 9), 0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# img = cv2.Canny(img, 100, 100)
#
# kernel = np.ones((5, 5), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)
#
# img = cv2.erode(img, kernel, iterations=1)
#
# cv2.rectangle(img, (img.shape[1] // 2, img.shape[0] // 2), (100, 100), (119, 201, 105), thickness=2)
#
# cv2.imshow('Dragonfly', img)
#
# print(img.shape)
#
# cv2.waitKey(0)

# cap = cv2.VideoCapture('videos/buyanov.mp4')
#
# while 1:
#     success, img = cap.read()
#
#     img = cv2.resize(img, (img.shape[1], img.shape[0]))
#     img = cv2.GaussianBlur(img, (9, 9), 0)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     img = cv2.Canny(img, 30, 30)
#
#     kernel = np.ones((5, 5), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#
#     img = cv2.erode(img, kernel, iterations=1)
#
#     cv2.imshow('Result', img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

img = cv2.imread('images/viridis.jpg')

new_img = np.zeros(img.shape, dtype='uint8')

# img = cv2.flip(img, 1)


def rotate(img, angle):
    height, width = img.shape[:2]
    point = (width // 2, height // 2)
    mat = cv2.getRotationMatrix2D(point, angle, 1)

    return cv2.warpAffine(img, mat, (width, height))


def transform(img, x, y):
    mat = np.float32([[1, 0, x], [0, 1, y]])

    return cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.Canny(img, 100, 140)

con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(new_img, con, -1, (230, 111, 148), 1)

print(con)
cv2.imshow('Result', new_img)
cv2.waitKey(0)
