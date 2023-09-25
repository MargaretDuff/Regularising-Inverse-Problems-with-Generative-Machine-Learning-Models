# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:58:16 2019

@author: magd21
"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#%%
#train_images=np.zeros((60000,56,56))
test_images=np.zeros((10000, 56, 56))
test_images_table=np.zeros((10000, 11))
# for i in range(60000):
#     a=np.random.randint( 28,42, 2)
#     b=np.random.randint(6,14)
#     cv2.circle(train_images[i,:,:], tuple(a), b, 1,-1)
#     #cv2.circle(train_images[i,:,:], tuple((a+np.random.uniform(-0.5,0.5,2)*b).astype(int)), 2, 2,-1)
#     c=np.random.randint(0,14,2)
#     d=np.random.randint(6,14,2)
#     cv2.rectangle(train_images[i,:,:], tuple(c), tuple(c+d),  1,-1)
    #cv2.circle(train_images[i,:,:], tuple((c+np.random.uniform(0.2,0.8,2)*d).astype(int)), 2, 2,-1)
    

for i in range(10000):
    a=np.random.randint( 28,42, 2)
    b=np.random.randint(6,14)
    cv2.circle(test_images[i,:,:], tuple(a), b, 1,-1)
    e=(a+np.random.uniform(-0.5,0.5,2)*b).astype(int)
    cv2.circle(test_images[i,:,:], tuple(e), 2, 2,-1)
    c=np.random.randint(0,14,2)
    d=np.random.randint(6,14,2)
    cv2.rectangle(test_images[i,:,:], tuple(c), tuple(c+d),  1,-1)
    #f=(c+np.random.uniform(0.2,0.8,2)*d).astype(int)
    f=[None,None]
    #cv2.circle(test_images[i,:,:], tuple(f), 2, 2,-1)
    test_images_table[i,:]=[*a,b,*c,*d,*e,*f]

    
    
table= pd.DataFrame(test_images_table, columns=['circle_centre_x', 'circle_centre_y', 'circle_radius', 'circle_bright_spot_x', 'circle_bright_spot_y', 'rect_top_left_x', 'rect_top_left_y', 'rect_bottom_right_x','rect_bottom_right_x', 'rect_bright_spot_x', 'rect_bright_spot_y'])
# np.save('./2d_shapes_train_images_none.npy', (train_images/2).astype('float32') )
np.save('./2d_shapes_test_images2_circle.npy', (test_images/2).astype('float32') )
table.to_csv('2d_shapes_test_images2_circle.csv')


#%%
# for i in range(10):
#     plt.imshow(test_images[200+i,:])
#     plt.show()
#     plt.close()

#%%
#img=np.zeros((56,56))
#
#img+=cv2.circle(np.zeros((56,56)), tuple(np.random.randint( 20,36, 2)), np.random.randint(8,20), 1,-1)
#a=np.random.randint(20,36,2)
#img+=cv2.rectangle(np.zeros((56,56)), tuple(a), tuple(a+np.random.randint(8,20,2)),  1,-1)
#b=np.random.randint(12,44,2)
##img+=cv2.fillConvexPoly(np.zeros((56,56)), np.array([b, b+np.random.randint(8,20,2),b+np.array([1,-1])*np.random.randint(2,12,2)]),  color=1)
#img+=cv2.putText(np.zeros((56,56)), 'M', tuple(np.random.randint(20,36,2)),cv2.FONT_HERSHEY_SIMPLEX,0.5, 1)
#plt.imshow(img, cmap='gray')
