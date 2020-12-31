import cv2
import os

for filename in os.listdir('/experiments/accuraccyTest'):
    img = cv2.imread(os.path.join('/experiments/accuraccyTest', filename))
    print('Original Dimensions : ', img.shape)
    cv2.imwrite('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/croppedAndResized/' + str(filename) + '.jpg', cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA))

# scale_percent = 60  # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# # resize image
# resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
#
# print('Resized Dimensions : ', resized.shape)
#
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
cv2.destroyAllWindows()
