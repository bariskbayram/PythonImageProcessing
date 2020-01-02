# Ortalama ve Varyans, pdf değeri
import matplotlib.pyplot as plt
import numpy as np

data_path ="/home/bariskbayram/Masaüstü/PythonIP"
train_data = np.loadtxt(data_path + "/mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "/mnist_test.csv", delimiter=",")

image_size = 28 # width and length
no_of_different_labels = 10 # 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size #784
print(test_data[:10]) #tüm satırları ve 10. sutuna kadar gösterir.
print(train_data[10,15])
print(train_data[10,:])

image = train_data[10, :]
print(image.ndim)
print(image.shape)

image1 = image[1:] #ilk satırda sayılar gösterildiği için o hariç alıyoruz
print(image1.shape)

image2 = image1.reshape(28,28) # 28'e 28 olarak yeniden boyutlandırdık
plt.imshow(image2, cmap="gray")
plt.show()

m,n = train_data.shape #60000 785 boyutunda
print(m,n)

#train data içinde hangi rakamdan kaç tane var
def counterTrainData(k=0):  #default 0 buluyor ama sen teker teker rakamları yollayarak bulcaksın
    s=0
    for i in range(m):
        if(train_data[i,0]==k):
            s=s+1
    return s

for i in range(10): #sırayla hepsini buluyoruz
    c = counterTrainData(i)
    print(i, " ", c)

#rakamların bulunma olasılıkları
import math
def pdf1 (x, mu=0.0, sigma=1.0):
    x = float(x - mu)/ sigma
    return math.exp(-x*x/2.0) / math.sqrt(2.0*math.pi) / sigma

print(pdf1(5,1,3))

def get_my_mean_and_std(k=0,l=350):
    s=0 #kactane sıfır var onu saysın//kac digit oldugu
    t=0 #intersitiy degeri pixeldeki
    #k=0 #sınfı bilgisi yani digitin
    #l=350  #location'ı belirtiyor.classın pixel degeri
    for i in range(m):  #ortalamayı buldurdu
        if(train_data[i,0]==k):
            s=s+1
            t=t+train_data[i,l+1] #l+1 olcak çünkü 0.'da digit bilgisi oluyor
            #digit_class=train_data[i,0]
            #top_left=train_data[i,1]
            #bottom_right=train_data[i,784]
    mean_1=t/s

    s=0
    t=0
    for i in range(m):
        if(train_data[i,0]==k):
            s=s+1
            diff_1=train_data[i,l+1]-mean_1
            t=t+diff_1*diff_1
    #var_1=t/(s-1)
    std_1=np.sqrt(t/(s-1))

    print(mean_1,std_1)
    return mean_1,std_1
        # train_data[i,0] #label
        # train_data[i,1] #sol üstteki deger
        # train_data[i,784] #en alt kosedeki deger


m1,std1=get_my_mean_and_std(4,200)
print("Mean: ", m1)
print("STD: ", std1)
print(pdf1(40,m1,std1))

image3=plt.imread("uc.png")
plt.imshow(image3)
plt.show()
test_value= image3[0,0,0]

print(test_value)

