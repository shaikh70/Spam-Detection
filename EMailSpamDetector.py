import pandas as pd
import re
df=pd.read_csv('D:\Projects\EMail Spam Detection NLP\spam_ham_dataset.csv')

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
corpus=[]
for i in range(len(df)):
  WB=re.sub('[^a-zA-Z]',' ',df['text'][i])
  WB=WB.lower()
  WB=WB.split()
  WB=[word for word in WB if not word in stopwords.words('english') and word!='subject']
  WB=' '.join(WB)
  corpus.append(WB)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()

Y=df['label_num']

Y=Y.iloc[:].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import MultinomialNB
Spam_Detect_Model=MultinomialNB().fit(X_train,Y_train)  

Y_pred=Spam_Detect_Model.predict(X_test)

df=pd.DataFrame([input('paste your mail here \n')],columns=['text'])

corpus=[]

for i in range(0,len(df)):
  review=re.sub('[^a-zA-Z]',' ',df['text'][i])
  review=review.lower()
  review=review.split()

  review=[word for word in review if word not in stopwords.words('english')]
  review=' '.join(review)

  corpus.append(review)

df=cv.transform(corpus).toarray()

pred=Spam_Detect_Model.predict(df)

label=pred[0]

if label==1:
  print('Spam')
else:
  print('Ham')