# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 12:27:47 2024

@author: E16348
"""

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import os
import json
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
nltk.download('stopwords')

print('marcando la ruta donde se encuentra el archivo')
archivo = os.path.join(os.getcwd(), 'Gift_Cards_reviews.jsonl')

print('Cargando las reseñas y normalizando el texto de las mismas.')
texts = []

with open(archivo, 'r') as file:
    for line in file:
        review = json.loads(line.strip())  # Eliminar el salto de línea
        text = review["text"].upper().translate(str.maketrans('', '', string.punctuation + '¡¿'))
        texts.append(text)

df = pd.DataFrame(texts, columns=['text'])        

print('Tokenizando las reseñas')
tokens = [word_tokenize(i) for i in texts]

print('Tomando consideración de las palabras que solo sirven para dar estructura a la oración')
sin_stopwords = []

for words in tokens:
  if not words in stopwords.words('english'):
    sin_stopwords.append(words)
    
print('Cargando el modelo VADER')    
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

print('Generamos los scores para cada oración del archivo')
df['scores'] = df['text'].apply(lambda text: sid.polarity_scores(text))
df = pd.concat([df.drop(['scores'], axis=1), df['scores'].apply(pd.Series)], axis=1)

print('Añadiendo los scores y la categoria al DF')
result = []
for value in df['compound']:
    if value >= 0 and value <=0.3:
        result.append("Insatisfecho")
    elif value >0.3 and value <=0.5:
        result.append("Neutral")
    else:
        result.append("Satisfecho")

df["Categoria"] = result

print('Generando la gráfica de las categorias del análisis de sentimientos')    
sent = sns.barplot(data=df, x='Categoria', y='compound')
plt.ylabel('Scores')
plt.title('Análisis de Sentimientos')

print('Guardando la gráfica en la ruta establecida')
plt.savefig('grafica_VADER.png')