import pandas as pd
from fastapi import FastAPI, HTTPException
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#////////////////////////////////////////////////////////////////////////////////////////
#def userdata( User_id : str )
df_user_items = pd.read_csv('limpieza_user_items.csv')
df_steam_games = pd.read_csv('limpieza_steam_games.csv')
df_user_reviews = pd.read_csv('limpieza_user_reviews.csv')

app = FastAPI()

@app.get('/userdata/{user_id}')
def userdata(user_id: str):
    
    user_items = df_user_items[df_user_items['user_id'] == user_id]
    
    
    user_reviews = df_user_reviews[df_user_reviews['user_id'] == user_id]
    
    
    total_money_spent = (user_items['items_count'] * df_steam_games['price']).sum()
    
    
    total_reviews = len(user_reviews)
    positive_reviews = user_reviews['recommend'].sum()
    recommendation_percentage = (positive_reviews / total_reviews) * 100 if total_reviews > 0 else 0
    
    
    total_items = len(user_items)
    
    
    user_data = {
        'user_id': user_id,
        'total_money_spent': total_money_spent,
        'recommendation_percentage': recommendation_percentage,
        'total_items': total_items
    }
    
    return user_data

#//////////////////////////////////////////////////////////////////////////////////////////

#/////////////////////////////////////////////////////////////////////////////////////////
#def countreviews( YYYY-MM-DD y YYYY-MM-DD : str )
# Cargar 
user_reviews = pd.read_csv('limpieza_user_reviews.csv')



@app.get('/countreviews/{start_date}/{end_date}')
async def countreviews(start_date: str, end_date: str):
    
    user_reviews['posted'] = pd.to_datetime(user_reviews['posted'], format='%d-%m-%Y')
    mask = (user_reviews['posted'] >= start_date) & (user_reviews['posted'] <= end_date)
    filtered_reviews = user_reviews[mask]
    
    
    unique_users = filtered_reviews['user_id'].nunique()
    avg_recommendation = filtered_reviews['recommend'].mean() * 100
    
    return {
        "cantidad_usuarios": unique_users,
        "porcentaje_recomendacion": avg_recommendation
    }

#/////////////////////////////////////////////////////////////////////////////////////



#/////////////////////////////////////////////////////////////////////////////////////
#def genre( género : str )
# Carga de datos 
generos_df = pd.read_csv('generos_steam_gamesX.csv')
user_items_df = pd.read_csv('limpieza_user_items.csv')


@app.get('/genre/{genre}')
def genre(genre: str):
    
    genre_df = generos_df[generos_df['genre'] == genre]
    
    if not genre_df.empty:
        
        merged_df = pd.merge(genre_df, user_items_df, left_on='item_id', right_on='item_id')
        ranking_position = merged_df['playtime_forever'].rank(ascending=False, method='min').iloc[0]
        
        
        genre_data = {
            'genre': genre,
            'position_in_ranking': int(ranking_position),
            'total_items': len(merged_df)
        }
    else:
        genre_data = {
            'genre': genre,
            'position_in_ranking': None,
            'total_items': 0
        }
    
    return genre_data

#////////////////////////////////////////////////////////////////////////////////


#////////////////////////////////////////////////////////////////////////////////
#def userforgenre( género : str )
# Carga de datos 
generos_df = pd.read_csv('generos_steam_gamesX.csv')
user_items_df = pd.read_csv('limpieza_user_items.csv')


@app.get('/userforgenre/{genre}')
def userforgenre(genre: str):
    
    genre_df = generos_df[generos_df['genre'] == genre]
    
    if not genre_df.empty:
       
        merged_df = pd.merge(genre_df, user_items_df, left_on='item_id', right_on='item_id')
        
        
        user_playtime = merged_df.groupby(['user_id', 'user_url'])['playtime_forever'].sum().reset_index()
        
        
        top_users = user_playtime.sort_values(by='playtime_forever', ascending=False).head(5)
        
       
        top_users_list = []
        for index, row in top_users.iterrows():
            user_data = {
                'user_id': row['user_id'],
                'user_url': row['user_url'],
                'playtime_forever': int(row['playtime_forever'])
            }
            top_users_list.append(user_data)
    else:
        top_users_list = []
    
    return top_users_list
#//////////////////////////////////////////////////////////////////////////////////////////


#/////////////////////////////////////////////////////////////////////////////////////////
#def developer( desarrollador : str )
release_date_genre_df = pd.read_csv('release_date_genre.csv')
user_items_df = pd.read_csv('limpieza_user_items.csv')

@app.get('/developer/{developer}')
def developer(developer: str):
   
    release_date_genre_df['release_date'] = pd.to_datetime(release_date_genre_df['release_date'])

   
    developer_df = release_date_genre_df[release_date_genre_df['developer'] == developer]
    
    if not developer_df.empty:
        
        merged_df = pd.merge(developer_df, user_items_df, left_on='item_id', right_on='item_id')
        
        
        free_to_play_items = merged_df[merged_df['genre'] == 'Free to Play']
        
        
        free_items_per_year = free_to_play_items.groupby(free_to_play_items['release_date'].dt.year)['item_id'].count()
        
        
        total_items_per_year = merged_df.groupby(merged_df['release_date'].dt.year)['item_id'].count()
        percentage_free_per_year = (free_items_per_year / total_items_per_year) * 100
        
        
        developer_data = {
            'developer': developer,
            'free_items_per_year': free_items_per_year.to_dict(),
            'percentage_free_per_year': percentage_free_per_year.to_dict()
        }
    else:
        developer_data = {
            'developer': developer,
            'free_items_per_year': {},
            'percentage_free_per_year': {}
        }
    
    return developer_data

#//////////////////////////////////////////////////////////////////////////////////////////



#/////////////////////////////////////////////////////////////////////////////////////////
#def sentiment_analysis( año : int ):
sentimiento_df = pd.read_csv('fun_sentimiento.csv')

@app.get('/sentiment_analysis/{year}')
def sentiment_analysis(year:int):
    
    filtered_df = sentimiento_df[sentimiento_df['release_date'] == year]
    
    if not filtered_df.empty:
        
        positive_count = (filtered_df['sentiment'] == 2).sum()
        negative_count = (filtered_df['sentiment'] == 0).sum()
        neutral_count = len(filtered_df) - (positive_count + negative_count)
        
        
        sentiment_data = {
            'Positive': int(positive_count),
            'Negative': int(negative_count),
            'Neutral': int(neutral_count)
        }
    else:
        sentiment_data = {
            'Positive': 0,
            'Negative': 0,
            'Neutral': 0
        }
    
    return sentiment_data

#/////////////////////////////////////////////////////////////////////////////////////////////


#/////////////////////////////////////////////////////////////////////////////////////////////
#def recomendacion_juego( id de producto )
# Cargar
df_juegos = pd.read_csv('def_muestra2_recomendacion.csv')

# Combinar 
df_juegos['combined_features'] = df_juegos['tag'] + ' ' + df_juegos['spec'] + ' ' + df_juegos['genre']

# Asegurar
df_juegos['combined_features'] = df_juegos['combined_features'].apply(lambda x: ' '.join(x.split()))

# Matriz
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(df_juegos['combined_features'])

# Similitud 
cosine_sim = cosine_similarity(count_matrix, count_matrix)

@app.get('/recomendacion_juego/{item_id}')
def recomendacion_juego(item_id: int):
    try:
        
        idx = df_juegos[df_juegos['item_id'] == item_id].index[0]

        
        sim_scores = list(enumerate(cosine_sim[idx]))

        
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        
        top_juegos = sim_scores[1:6]

        
        recommended_juegos = [df_juegos.iloc[i[0]]['app_name'] for i in top_juegos]

        return recommended_juegos
    except IndexError:
        return {"message": "Juego no encontrado"}

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////