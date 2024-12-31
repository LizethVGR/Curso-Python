# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:31:22 2024

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
os.environ['CURL_CA_BUNDLE'] = ''

print('Cargar el modelo y el tokenizador BERT preentrenado')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

print('Crear un pipeline de clasificación de texto'')
clasificador = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

print('marcando la ruta donde se encuentra el archivo')
archivo = os.path.join(os.getcwd(), 'Gift_Cards_reviews.jsonl')

print('Cargando las reseñas y normalizando el texto de las mismas.')
texts = []

with open(archivo, 'r') as file:
    for line in file:
        review = json.loads(line.strip())  # Eliminar el salto de línea
        # Convert the 'review' dictionary to a JSON string and then to uppercase
        #texts.append(review["text"].upper())
        text = review["text"].upper().translate(str.maketrans('', '', string.punctuation + '¡¿'))
        texts.append(text)

print('Obteniendo el resultado del modelo Bert')
resultado = clasificador(texts)

print('Categorizando el resultado del modelo Bert')
result = []
for val in resultado:
    if val['score'] >= 0 and val['score'] <=0.3:
        result.append("Insatisfecho")
    elif val['score'] >0.3 and val['score'] <=0.5:
        result.append("Neutral")
    else:
        result.append("Satisfecho")

df = pd.DataFrame(texts, columns=['text'])
df["Categoria"] = result
df['Score'] = [val['score'] for val in resultado]

print('Generando la gráfica con los Scores del modelo Bert')
sent = sns.barplot(data=df, x='Categoria', y='Score')
plt.title('Análisis de Sentimientos')

print('Guardando la gráfica en la ruta establecida')
plt.savefig('grafica_Bert.png')