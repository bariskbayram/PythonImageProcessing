import numpy as np
import matplotlib.pyplot as plt

img_1 = plt.imread('ITCrowd.png')

print(type(img_1))
print(img_1.ndim)
print(img_1.shape)
m, n, p = img_1.shape
print(m, n, p)
print("size(m x n x dimension): ", img_1.size)

plt.imshow(img_1)
plt.show()

print(img_1[200, 100, :])

new_image = np.zeros((m, n), dtype=float)

for i in range(m):
    for j in range(n):
        s = (img_1[i, j, 0] + img_1[i, j, 1] + img_1[i, j, 2])/3
        new_image[i, j] =s

plt.imshow(new_image, cmap='gray')
plt.show()
plt.imsave('ITCrowdGray.png', new_image, cmap='gray')


#Transpoze için m ve n değişkenlerinin yeri değişir..

transpoze_image = np.zeros((n, m), dtype=float)

for i in range(m):
    for j in range(n):
        s = (img_1[i, j, 0] + img_1[i, j, 1] + img_1[i, j, 2]) / 3
        transpoze_image[j, i] = s

plt.imshow(transpoze_image, cmap='gray')
plt.show()
plt.imsave('ITCrowd2.png', transpoze_image, cmap='gray')
