import pandas as pd,sys,ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
try:
    movies=pd.read_csv("tmdb_5000_movies.csv");print("✅ TMDB Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ Error: Please place 'tmdb_5000_movies.csv' in the same folder as this script.");sys.exit()
movies=movies[['title','overview','genres','vote_average']].dropna()
movies['genres']=movies['genres'].apply(lambda x:' '.join([g['name'] for g in ast.literal_eval(x)]))
movies['content']=movies['overview']+" "+movies['genres']
tfidf=TfidfVectorizer(stop_words='english')
tfidf_matrix=tfidf.fit_transform(movies['content'])
cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
def recommend_movie(title):
    if title not in movies['title'].values:
        print(f"\n❌ Movie '{title}' not found in dataset.")
        similar=[m for m in movies['title'].values if title.lower() in m.lower()]
        if similar:print("👉 Did you mean:",', '.join(similar[:5]))
        return
    idx=movies[movies['title']==title].index[0]
    sim_scores=sorted(list(enumerate(cosine_sim[idx])),key=lambda x:x[1],reverse=True)[1:6]
    print(f"\n🎬 Because you liked **{title}**, you may also enjoy:\n")
    titles=[];scores=[];genres=[];ratings=[]
    for i,(movie_index,score) in enumerate(sim_scores):
        m=movies.iloc[movie_index]
        titles.append(m['title']);scores.append(score);genres.append(m['genres']);ratings.append(m['vote_average'])
        print(f"{i+1}. 🎞️ {m['title']}");print(f"    🔹 Similarity Score: {score:.2f}");print(f"    ⭐ IMDb Rating: {m['vote_average']}/10");print(f"    🎭 Genres: {m['genres']}\n")
    open("recommendation_log.txt","a",encoding="utf-8").write(f"\nUser liked: {title}\nRecommended: {', '.join(titles)}\n")
    plt.figure(figsize=(8,4));sns.barplot(x=scores,y=titles);plt.xlabel("Similarity Score");plt.ylabel("Recommended Movies");plt.title(f"Top 5 Movies Similar to '{title}'");plt.tight_layout();plt.show()
    all_genres=' '.join(genres).split();genre_counts=pd.Series(all_genres).value_counts().head(6)
    plt.figure(figsize=(5,5));plt.pie(genre_counts.values,labels=genre_counts.index,autopct='%1.1f%%',startangle=140);plt.title("Genre Distribution of Recommendations");plt.show()
    plt.figure(figsize=(7,4));sns.scatterplot(x=scores,y=ratings,hue=titles,s=120)
    [plt.text(scores[i]+0.002,ratings[i]+0.02,titles[i],fontsize=9) for i in range(len(titles))]
    plt.xlabel("Similarity Score");plt.ylabel("IMDb Rating");plt.title("Similarity Score vs IMDb Rating");plt.legend(bbox_to_anchor=(1.05,1));plt.tight_layout();plt.show()
if __name__=="__main__":
    print("🎥 AI-POWERED MOVIE RECOMMENDATION SYSTEM 🎥");print("Dataset: TMDB 5000 Movies");print("-"*50)
    recommend_movie(input("\nEnter a movie title you like: "))
