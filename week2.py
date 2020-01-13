import numpy as np
import matplotlib.pyplot as plt

image1 = plt.imread('ITCrowd.png')
plt.imshow(image1)
plt.show()

image1 = image1 + 0.25
plt.imshow(image1)
plt.show()

image1[:,:,0] = image1[:,:,0] - 0.25
plt.imshow(image1)
plt.show()

image2 = plt.imread('ITCrowd.png')

#Fotoğrafın renk değerlerinin her birini belli sayıda arttırma Yoğunluk değiştirme
def RGBPlus(image, s=0.5):
    m, n, p = image.shape
    newImage = np.zeros((m, n, 3), dtype=float)

    for m in range(image.shape[0]):
        for n in range(image.shape[1]):
            newImage[m, n, 0] = image[m, n, 0] + s
            newImage[m, n, 1] = image[m, n, 1] - s
            newImage[m, n, 2] = image[m, n, 2] + s

    return newImage

image3 = RGBPlus(image2)
plt.imshow(image3)
plt.show()

#Fotoğrafın pixellerini 1/4  oranında küçültme
def pixelOneToFour1 (image):
    m, n, p = image.shape
    new_m = int(m/2)
    new_n = int(n/2)

    image2 = np.zeros((new_m, new_n), dtype=float) #eğer new_m ve new_n değil de m,n yaparsan orjinal boyutun 1/4 oranına
                                                  # küçük halini koyar.

    for m in range(new_m):
        for n in range(new_n):
            s0 = (image[m * 2, n * 2, 0] + image[m * 2, n * 2, 1] + image[m * 2, n * 2, 2]) / 3
            image2[m, n] = s0

    return image2

image0 = plt.imread('ITCrowd.png')
plt.imshow(image0)
image4 = pixelOneToFour1(image0)
plt.imshow(image4)
plt.show()
plt.imsave('ITCrowdOneToFour1.png', image4)

#Fotoğrafın orjinal boyutunun 1/4 kısmına fotoğrafın küçük hali konur
def pixelOneToFour2(image):
    m, n, p = image.shape
    new_m = int(m/2)
    new_n = int(n/2)

    image2 = np.zeros((m,n,3), dtype=float)

    for m in range(new_m):
        for n in range(new_n):
            image2[m, n, 0] = image[m * 2, n * 2, 0]
            image2[m, n, 1] = image[m * 2, n * 2, 1]
            image2[m, n, 2] = image[m * 2, n * 2, 2]

    return image2

image5 = pixelOneToFour2(image0)
plt.imshow(image5)
plt.show()
plt.imsave('ITCrowdOneToFour2.png', image5)

#Fotoğrafın renk değerlerinin(yoğunluklarının) tersini alma. Unutma sen 1- yapıyorsun hoca 255-
def RGBReverse(image, s=0.5):
    m, n, p = image.shape
    newImage = np.zeros((m, n, 3), dtype=float)

    for m in range(image.shape[0]):
        for n in range(image.shape[1]):
            newImage[m, n, 0] = 1 - image[m, n, 0]
            newImage[m, n, 1] = 1 - image[m, n, 1]
            newImage[m, n, 2] = 1 - image[m, n, 2]

    return newImage

image5 = RGBReverse(image0)
plt.imshow(image5)
plt.show()
plt.imsave('ITCrowdRGBReverse.png', image5)
