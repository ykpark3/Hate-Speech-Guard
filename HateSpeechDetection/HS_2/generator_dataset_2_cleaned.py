import pickle
from collections import Counter
import pandas as pd
import re
import nltk
import math
from keras_preprocessing.text import Tokenizer
from matplotlib import pyplot
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import numpy as np
from Utils.Utils import data_text_cleaning

# csv3 파일 load
data = pd.read_csv("csv2/hs_data_2.csv", encoding="ISO-8859-1")


result = []
text = data['Contents']
class_text = data['Class']

for i in range(len(data)):
    temp = []
    temp.append(data_text_cleaning(text[i]))
    temp.append(class_text[i])
    result.append(temp)

print(result)

# 전처리된 npy 파일 생성
np.save('csv2/hs_total_data_cleanded.npy', result)

# fit_on_texts 는 단어 인덱스를 구축
# texts_to_sequences 은 정수 인덱스를 리스트로 변환
# pad_sequences 길이가 같지 않고 적거나 많을 때 일정한 길이로 맞춰 줄 때 사용
X = []
for i in range(len(result)):
    X.append(result[i][0])

token = Tokenizer()
token.fit_on_texts(X)

# 데이타 토큰 생성
# 분석한 token을 저장해서 모델을 사용할때 같이 불러줘야한다.그렇지 않으면 모델로 사용할 때 예측을 하고픈 문장으로
# 토크나이징이 되기 때문에 학습의 결과가 아무 의미가 없다.
with open('csv2/hs_total_data_cleanded.pickle', 'wb') as handle:
    pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)
