import numpy as np
import matplotlib.pyplot as plt
import math
 
myList=[2,4,3,4,5,6,3,3,2,1]
def calculatingMeanAndVaryans (myList=[2,4,3,4,5,6,3,3,2,1]):
    t = 0
    s = 0

    #calculating mean
    for i in myList:
         s = s + 1
         t = t + i
    mean = t / s

    t = 0
    s = 0
    #calculating varyans
    for i in myList:
        s = s + 1
        t = t + (i-mean) * (i-mean)
    varyans = t / (s-1)
    varyans = math.sqrt(varyans) # bu satırdan emin değilim

    return mean, varyans

print(calculatingMeanAndVaryans())

def creatingHistogram (myList):
    for i in myList:
        if i in histogram.keys():
            histogram[i] = histogram[i] + 1
        else:
            histogram[i] = 1

    print(histogram)

histogram = {}
creatingHistogram(myList)

image = plt.imread('ITCrowd.png')
plt.imshow(image)
plt.show()

def creatingPhotoHistogram (image):
    m, n, p = image.shape
    photoHistoram = {}
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if image[i,j,k] in photoHistoram.keys():
                    photoHistoram[image[i,j,k]] = photoHistoram[image[i,j,k]] + 1
                else:
                     photoHistoram[image[i,j,k]] = 1
    return photoHistoram

print(creatingPhotoHistogram(image))

def covertToRealHistogram (histogram):
    x=[]
    y=[]
    for key in histogram.keys():
        x.append(key)
        y.append(histogram[key])
    return x, y, histogram

x, y, histogram2 = covertToRealHistogram(creatingPhotoHistogram(image))
plt.bar(x,y)
plt.show()
