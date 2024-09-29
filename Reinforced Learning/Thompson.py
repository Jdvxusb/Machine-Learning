import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
prizes = [0] * d
tiklamalar = [0] * d
toplam = 0
secilenler = []
birler = [0]*d
sifirlar = [0]*d

for i in range(1,N):
    ad = 0
    max_th = 0
    for j in range(0,d):
        rasbeta = random.betavariate(birler[j]+1, sifirlar[j]+1)
        if(rasbeta > max_th):
            max_th = rasbeta
            ad = j
    
    secilenler.append(ad)
    odul = veriler.values[i,ad]
    if(odul==1):
        birler[ad] = birler[ad]+1
    else:
        sifirlar[ad] = sifirlar[ad]+1
    prizes[ad] = prizes[ad] + odul
    toplam = odul + toplam

print(toplam)

plt.hist(secilenler)
plt.show()