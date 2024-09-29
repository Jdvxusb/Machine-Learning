import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
prizes = [0] * d
tiklamalar = [0] * d
toplam = 0
secilenler = []

for i in range(1,N):
    ad = 0
    max_ucb = 0
    for j in range(0,d):
        if(tiklamalar[j]>0):
            ortalama = prizes[j] / tiklamalar[j]
            delta = math.sqrt(3/2* math.log(i)/tiklamalar[j])
            ucb = ortalama + delta
        else:
            ucb = N * 10
        
        if(max_ucb < ucb):
            max_ucb = ucb
            ad = j
    
    secilenler.append(ad)
    tiklamalar[ad]+=1
    odul = veriler.values[i,ad]
    prizes[ad] = prizes[ad] + odul
    toplam = odul + toplam

print(toplam)

plt.hist(secilenler)
plt.show()