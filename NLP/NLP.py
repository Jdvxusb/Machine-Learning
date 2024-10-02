import pandas as pd
import numpy as np
import nltk
import re

yorumlar = pd.read_csv("yelp_academic_dataset_review.csv")
yorumlar = yorumlar.drop(["review_id","user_id","business_id","cool","useful","funny","date"],axis  =1)
yorumlar = yorumlar.rename(columns={'text':'Review', 'stars':'liked'}, )

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

derlem = []
for i in range(0,len(yorumlar)):
    if(yorumlar['liked'][i]>=3):
        yorumlar['liked'][i] = 1
    else:
        yorumlar['liked'][i] = 0
    
    yorum = re.sub('^[a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not  kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
x = cv.fit_transform(derlem).toarray()
y = yorumlar.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.2, random_state=4)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)




