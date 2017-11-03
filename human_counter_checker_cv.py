# -*- coding: utf-8 -*-
import apple_keras as apple
import numpy as np
#import cv2
from keras.preprocessing.image import load_img, img_to_array

#cam = cv2.VideoCapture(0)
image_size = 32
categories = ["一人", "二人","三人"]

def main():

#    while(True):
#        ret, frame = cam.read()
#        cv2.imshow("Show FLAME Image", frame)

#        k = cv2.waitKey(1)
#        if k == ord('s'):
#            cv2.imwrite("output.png", frame)
#            cv2.imread("output.png")

            X = []
            img = load_img("./output.png", target_size=(image_size,image_size))
            in_data = img_to_array(img)
            X.append(in_data)
            X = np.array(X)
            X  = X.astype("float")  / 255

            model = apple.build_model(X.shape[1:])
            model.load_weights("./image/three_check.h5")

            pre = model.predict(X)
            print(pre)
            # todo なんかイケてない
            if pre[0][0] > pre[0][1]:
                if pre[0][0] > pre[0][2]:
                    text = 'これは' + categories[0]+ 'だよ'
                    print(text)
                else:
                    text = 'これは' + categories[2]+ 'だよ'
                    print(text)
            elif pre[0][1] > pre[0][2]:              
                text = 'これは' + categories[1]+ 'だよ'
                print(text)
            else:
                text = 'これは' + categories[2]+ 'だよ'
                print(text)

#        elif k == ord('q'):
#            break

#    cam.release()
#    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()