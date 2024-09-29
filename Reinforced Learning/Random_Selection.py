import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
toplam = 0
secilenler = []

for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    prize = veriler.values[n,ad]
    toplam = toplam + prize
print(toplam)

plt.hist(secilenler)
plt.show()