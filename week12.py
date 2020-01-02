import numpy as np
import matlotlib.pyplot as plt
import mnist
from conv import Conv3x3

image = plt.imread('ITCrowd.png')

def __init__(self, num_filters):
    self.num_filters = num_filters
    # num_filters max boyutu 8,4.. gibi

    self.filters = np.random.randn(num_filters, 3, 3) / 9  # 8 tane (3,3) luk matris üretti.

    # mask in değerlerini random üretti ilerideki aşamalarda güncelleyecek.

    def iterate_regions(self, image):  # resim geldi.
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array
        '''

    h, w = image.shape  # en,boy ,değerleri 28x8

    # resim üzerindeki son iki satırları ilerlemiyor. mask 3x3 olduğu için -2 oldu.
    for i in range(h - 2):
        for j in range(w - 2):
            im_region = image[i:(i + 3), j:(j + 3)]
            yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        '''
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        # !!önemli!!!#
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output



    # The mnist package handles the MNIST dataset for us!
    # Learn more at https://github.com/datapythonista/mnist
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    conv = Conv3x3(8)
    output = conv.forward(train_images[0])
    print(output.shape)  # (26, 26, 8)