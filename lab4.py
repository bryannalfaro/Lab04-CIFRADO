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
import testLab
import json

img = Image.open('camera.png').convert('L') # L = 8-bit pixels, black and white
imgBits = cifrados.read_image(img)
size = len(imgBits)

examplesData = json.load(open('examplesData.json'))

'''for example in examplesData['lgc']:
    print(example)
    resultLGC = cifrados.lgc(a=example['a'], b=example['b'], N=example['N'], seed=example['seed'], size=size)
    testsResults = testLab.successTable(resultLGC, True)
    #testLab.getHistogram(testsResults)

for example in examplesData['wichman']:
    resultWichman = cifrados.wichman(example['s1'], example['s2'], example['s3'], size)
    testsResults = testLab.successTable(resultWichman)
    #testLab.getHistogram(testsResults)

for example in examplesData['lfsr']:
    resultLFSR = cifrados.lfsr(seed=example['seed'], taps=example['taps'], nbits=size)
    lfsr_xor = cifrados.xor(imgBits, resultLFSR)
    testsResults = testLab.successTable(lfsr_xor, True)
    #testLab.getHistogram(testsResults)'''


#testLab.generateLGC(iterations=20)

testLab.generateWichman(iterations=50)

#testLab.generateLFSR(iterations=200)

'''print('now LFSR')
resultLFSR = cifrados.lfsr(seed=format('110100011011011010111101'), taps=[2,5,7,10,15,11], nbits=size)
lfsr_xor = cifrados.xor(imgBits, resultLFSR)
testLab.successTable(lfsr_xor, True)'''
