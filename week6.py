import matplotlib.pyplot as plt
import numpy as np

im_1=plt.imread('test.png')
im_2= np.zeros((341,306),dtype=float) #normalde dtype=np.uint8
im_2 = im_1[:,:,0]
plt.imshow(im_2,cmap='gray')
plt.show()

m,n = im_2.shape

im_3 = im_1[:,:,0]   #im_1 gerçek foto , im_2 işlenen foto, im_3 filtre uygulanacak foto
im_3= np.zeros((341,306),dtype=float)  #normalde dtype=np.uint8

for i in range(1,m-1):
    for j in range(1,n-1):
        s= \
        im_2[i-1,j-1]/9+ \
        im_2[i-1,j]/9+ \
        im_2[i-1,j+1]/9+ \
        im_2[i,j-1]/9+ \
        im_2[i,j]/9+ \
        im_2[i,j+1]/9+ \
        im_2[i+1,j-1]/9+ \
        im_2[i+1,j]/9+ \
        im_2[i+1,j+1]/9
        #s=int(s)
        im_3[i,j]=s

plt.subplot(1,2,1)  # bir satir iki satırlık bölge oluştur demek
plt.imshow(im_2,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(im_3,cmap='gray')
plt.show()