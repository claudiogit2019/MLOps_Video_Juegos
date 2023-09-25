# Machine Learning Operations (MLOps)

           ¡Bienvenidos al primer proyecto individual de la etapa de labs!


**Propuesta de trabajo (requerimientos de aprobación)**

**Transformaciones** 
La limpieza de los DataSet se dividio en tres grandes partes:

                    # **Lectura y limpieza de steam_games.json.gz**

                    # **lectura y Limpieza de australian_users_items.json**

                    # **Lectura y Limpieza de australian_users_reviews.json**

Se Utilizo sentencias como:

**Para la lectura**

file_path = "/content/drive/MyDrive/steam_games.json.gz"

data = []

with gzip.open(file_path, "rt", encoding="utf-8") as f:

        for line in f:
 
            data.append(json.loads(line))


**Para manipular en un DataFrame**

df_gral = pd.DataFrame(data)

df_gral.info()

**Para la reduccion de columnas y renombre**


limpieza_steam_games = df_gral.drop(columns = ['user_id','steam_id','items','items_count'])

limpieza_steam_gamesX = limpieza_steam_gamesX.rename(columns={'id':'item_id'})

**En el momento de guardar** 

**# Guardar el DataFrame en un archivo CSV**

limpieza_steam_gamesX.to_csv("/content/drive/MyDrive/limpieza_steam_games.csv", index=False)

**Ademas se trabajo
**
 
                **con la columna genres de limpieza_steam_games.csv**
                **Top 5 usuarios**



**Feature Engineering**

En el dataset user_reviews se incluyen reseñas de juegos hechos por distintos usuarios. Debes crear la columna 'sentiment_analysis' aplicando análisis de sentimiento con NLP

En el Proyecto_de_Video_Juego.ipynb, se realiza 

**Analsis de Sentimiento** tomada del siguiente articulo, https://neuraldojo.org/proyectos/analisis-de-sentimiento/guia-basica-de-analisis-de-sentimiento-en-python/



**Desarrollo API**: Propones disponibilizar los datos de la empresa usando el framework FastAPI. Las consultas que propones son las siguientes:

def userdata( User_id : str )

def countreviews( YYYY-MM-DD y YYYY-MM-DD : str )

def genre( género : str )

def userforgenre( género : str )

def developer( desarrollador : str )

def sentiment_analysis( año : int )

Los mismos de podran probar en Render.com


### Análisis exploratorio de los datos: (Exploratory Data Analysis-EDA)

Ya los datos están limpios, ahora es tiempo de investigar las relaciones que hay entre las variables del dataset, ver si hay outliers o anomalías (que no tienen que ser errores necesariamente 👀 ), y ver si hay algún patrón interesante que valga la pena explorar en un análisis posterior. Las nubes de palabras dan una buena idea de cuáles palabras son más frecuentes en los títulos, ¡podría ayudar al sistema de predicción! En esta ocasión vamos a pedirte que no uses librerías para hacer EDA automático ya que queremos que pongas en práctica los conceptos y tareas involucrados en el mismo.

                                **Empezamos a indagar sobre los datos**

**contemos, Empresa publicadora del contenido**

**contemos, app_name, nombre de contenido**

**contemos, cantidad de developer y genero**

**contemos, user_id y item_name**

**contemos las columnas recommend y posted**

### Ademas

***Muestra Media***

**Varianza y Desviacion  de la Muestra**

**Muestra Mediana**

**Histograma**


### Modelo de aprendizaje automático

Una vez que toda la data es consumible por la API, está lista para consumir por los departamentos de Analytics y Machine Learning, y nuestro EDA nos permite entender bien los datos a los que tenemos acceso, es hora de entrenar nuestro modelo de machine learning para armar un sistema de recomendación. Para ello, te ofrecen dos propuestas de trabajo: En la primera, el modelo deberá tener una relación ítem-ítem, esto es se toma un item, en base a que tan similar esa ese ítem al resto, se recomiendan similares. Aquí el input es un juego y el output es una lista de juegos recomendados, para ello recomendamos aplicar la similitud del coseno.

def recomendacion_juego( id de producto )

La misma de podra probar en render.com
## Enlaces de Video:

Enlace de Render

https://sistema-recomendacion-voideo-juego2023.onrender.com

Enlace de Video

https://drive.google.com/file/d/1uRObXiNxvlyWjD5LBFE9JqDxHEewPYkZ/view?usp=drive_link

Enlace de Github

https://github.com/claudiogit2019/MLOps_Video_Juegos.git

Enlace de Colaboratory Google-Proyecto_de_Video_Juego.ipynb 

https://colab.research.google.com/drive/1rnQv0O-R5lI7H_Xy4GD7A_qDdVJpERG6?usp=sharing




