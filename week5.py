import numpy as np
import matplotlib.pyplot as plt
import math

data_path ="/home/bariskbayram/Masaüstü/PythonIP"
#train_data = np.loadtxt(data_path + "/mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "/mnist_test.csv", delimiter=",")

eps = np.finfo(float).eps
def my_pdf_1(x,mu=0.0,sigma=1.0):
    x = float(x-mu)/(sigma+eps)
    return math.exp(((-x*x)/2.0)/(math.sqrt(2.0*math.pi)/(sigma+eps)))

def get_my_mean_and_std(k=0,l=350):
    s=0
    t=0
    for i in range(10000):  #m train olursa 60000 test olursa 10000
        if(test_data[i,0]==k):
            s=s+1
            t=t+test_data[i,l+1]
    mean_1=t/s

    s=0
    t=0
    for i in range(10000):
        if(test_data[i,0]==k):
            s=s+1
            diff_1=test_data[i,l+1]-mean_1
            t=t+diff_1*diff_1
    std_1=np.sqrt(t/(s-1))

    return mean_1,std_1

im_1=plt.imread('uc.png')
plt.imshow(im_1)
plt.show()

im_2=im_1[:,:,0] #im_1 i iki boyutlu hale getirmek için

im_5=im_2.reshape(1,784) #im_2 resmini düzleştirdi.

def hesapla(im_5):
    liste = list()  # pdf değerlerini tutmak için
    for i in range(10):
        pdf_t = 0
        for j in range(784):
            x = im_5[0, j]
            m_1, std_1 = get_my_mean_and_std(i, j)
            pdf_deger = my_pdf_1(x, m_1, std_1)
            if (math.isnan(pdf_deger) == False):
                pdf_t = pdf_t + pdf_deger
    liste.append(pdf_t)
    return liste

listem = hesapla(im_5)
print(listem)

m = len(listem)
maxNumber = 0
for i in range(m):  # listedeki en büyük pdf degerini bulmak için
    if maxNumber < listem[i]:
        maxNumber = listem[i]
print(maxNumber)
