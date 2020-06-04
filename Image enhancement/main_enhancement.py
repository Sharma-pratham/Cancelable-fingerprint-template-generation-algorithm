
import numpy as np
import cv2
import sys

from image_enhance import image_enhance

if __name__ == '__main__':

    if(len(sys.argv)<2):
        print('loading sample image')
        img_name = 'a.tif'
        img = cv2.imread(r'C:\Users\User\Desktop\DB1_B\101_1.tif' )
    elif(len(sys.argv) >= 2):
        img_name = sys.argv[1]
        img = cv2.imread(r'C:\Users\User\Desktop\DB1_B\101_1.tif')
    if(len(img.shape)>2):
         img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    rows,cols = np.shape(img)
    aspect_ratio = np.double(rows)/np.double(cols)

    new_rows = 350      # randomly selected number
    new_cols = new_rows/aspect_ratio

    img = cv2.resize(img,(np.int(new_cols),np.int(new_rows)))

    enhanced_img = image_enhance(img)

    print('saving the image')
    cv2.imwrite(r'C:\Users\User\Desktop\DB1_B\A\enh1' + img_name, (255*enhanced_img))


