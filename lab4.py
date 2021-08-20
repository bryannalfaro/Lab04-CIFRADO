# Universidad del Valle de Guatemala
# Cifrado de informacion
# Lab 4
# Julio Herrera
# Bryann Alfaro
# Diego Arredondo

import cifrados
import matplotlib.pyplot as plt
from PIL import Image
import random
import testLab
import json

img = Image.open('camera.png').convert('L') # L = 8-bit pixels, black and white
imgBits = cifrados.read_image(img)
size = len(imgBits)

examplesData = json.load(open('examplesData.json'))

#3 ejemplos buenos y malos lgc
for example in examplesData['lgc']:
    print(example)
    resultLGC = cifrados.lgc(a=example['a'], b=example['b'], N=example['N'], seed=example['seed'], size=size)
    testsResults = testLab.successTable(resultLGC, True)

#3 ejemplos buenos y malos wichman
for example in examplesData['wichmaN']:
    print(example)
    resultWichman = cifrados.wichman(example['s1'], example['s2'], example['s3'], size)
    testsResults = testLab.successTable(resultWichman)

#3 ejemplos buenos y malos lfsr
for example in examplesData['lfsr']:
    print(example)
    resultLFSR = cifrados.lfsr(seed=example['seed'], taps=example['taps'], nbits=size)
    lfsr_xor = cifrados.xor(imgBits, resultLFSR)
    testsResults = testLab.successTable(lfsr_xor, True)


#200 iteraciones de cada generador para histograma
testLab.generateLGC(iterations=200)

testLab.generateWichman(iterations=200)

testLab.generateLFSR(iterations=200)

