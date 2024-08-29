import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = {
    'title': [
        'The Matrix', 'Inception', 'The Avengers', 'The Dark Knight', 'Interstellar',
        'The Lord of the Rings', 'Avatar', 'Pulp Fiction', 'Fight Club', 'Forrest Gump'
    ],
    'genre': [
        'Action|Sci-Fi', 'Action|Sci-Fi|Thriller', 'Action|Adventure|Sci-Fi', 'Action|Crime|Drama', 'Adventure|Drama|Sci-Fi',
        'Action|Adventure|Drama', 'Action|Adventure|Sci-Fi', 'Crime|Drama', 'Drama|Fight', 'Drama|Romance'
    ],
    'rating': [8.7, 8.8, 8.0, 9.0, 8.6, 8.8, 7.8, 8.9, 8.8, 8.8],
    'release_date': [
        '1999-03-31', '2010-07-16', '2012-05-04', '2008-07-18', '2014-11-07',
        '2001-12-19', '2009-12-18', '1994-10-14', '1999-10-15', '1994-07-06'
    ]
}

df = pd.DataFrame(data)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['genre'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_movies(title, num_recommendations=3):
    idx = df.index[df['title'] == title].tolist()[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = df.iloc[movie_indices][['title', 'rating', 'release_date']]
    return recommendations

# Test the recommendation system
movie_title = 'The Matrix'
recommended_movies = recommend_movies(movie_title)
print(f"Recommended movies similar to '{movie_title}':")
print(recommended_movies)
