<h1 align="center"> Análisis de Sentimientos basados en Reseñas </h1>
<h3 align="left"> Índice </h1>

*[Descripción del proyecto](#descripción-del-proyecto)

A continuacion se presentan los desarrollos realizados para la obtención de un Score, basado en análisis de sentimientos, para 100 reseñas de tarjetas de regalo. Esto con la finalidad de conocer cuales sueron las opiniones de estas 100 referencias, satisfecho, regular o insatisfecho.

Para poder iniciar es necesario cargar el archivo en formato json <em> Gift_Cards_reviews.jsonl <em>

*[Archivo para generar entorno virtual y línea de comando](#Título-e-imagen-de-portada)

El primer paso a considerar dentro de este proyecto es la creación de un nuevo entorno virtual con las caracteristicas necesarias para poder desaroollar las tareas solicitadas.
Para ello procedemos a cargar el archivo env_curso_BSG.yml con la línea de comando:

<em> conda env create -f  env_curso_BSG.yml</em>

*[Archivos .py y .ipynb (Modelos Bert y Vader)](#insignias)

En este apartado, se genraron tanto el notebook de colab como el scritp en spyder para poder obtener los archivos .ipynb y .py solicitados.
Cabe mencionar que como el análisis con el modelo Bert parecia ser no del todo esclarecedor, se proceedio a realizar las pruebas con el modelo VADER.
De esta parte obtuvimos los archivos:

Lineas de Comandos.ipynb (Google colar y/o Jupyter)

Analisis_sentimientos_Bert.py (Spyder)

Analisis_sentimientos_VADER.py (Spyder)

Cabe mencionar que los archivos .py pueden ejecutados desde la consola de anaconda con los comandos: 

<em>python Analisis_sentimientos_Bert.py<em> y/o 

<em>python Analisis_sentimientos_VADER.py<em>

*[Gráficos de barras para la representación de resultados.](#conclusión)

Finalmente como parte de las instrucciones de los scripts se genera una gráfica, en cada uno de ellos, mostrando los resultados por categoria (Satisfecho, Regular e Insatisfecho) para cada Modelo según corresponda.
Esta gráfica se guarda de manera automatica dentro del entorno donde estes guardando las referencias.
