# Universidad del Valle de Guatemala
# Cifrado de informacion
# Lab 3
# Julio Herrera
# Bryann Alfaro
# Diego Arredondo

import cifrados
import matplotlib.pyplot as plt
from PIL import Image
import random

def show_image(I):
    plt.figure()
    plt.imshow(I,cmap='gray')
    plt.show()

# load image as pixel array
img = Image.open('camera.png').convert('L') # L = 8-bit pixels, black and white
#img.show()

imgBits = cifrados.read_image(img)

I = cifrados.write_image(imgBits, img)
show_image(I)

# algorithms

resultLGC_test = cifrados.lgc(a=3, b=3, N=7, seed=5, size=16) # Prueba de LGC retornando 16 bits
print(resultLGC_test)

print('Now LGC')
a= int(input('Ingrese a: '))
b= int(input('Ingrese b: '))
N= int(input('Ingrese N: '))
resultLGC = cifrados.lgc(a=a, b=b, N=N, seed=53, size=len(imgBits))

lgc_xor = cifrados.xor(imgBits, resultLGC)
show_image(cifrados.write_image(lgc_xor, img))

print('Now wichman')
s1 = random.randint(0,30000)
s2 = random.randint(0,30000)
s3 = random.randint(0,30000)
resultWichman = cifrados.wichman(s1,s2,s3,len(imgBits))

wichman_xor = cifrados.xor(imgBits, resultWichman)
show_image(cifrados.write_image(wichman_xor, img))

resultLFSR = cifrados.lfsr(seed=format('110100011011011010111101'), taps=[2,5,7,10,15,11], nbits=len(imgBits))
lfsr_xor = cifrados.xor(imgBits, resultLFSR)
print('now LFSR')
show_image(cifrados.write_image(lfsr_xor, img))
