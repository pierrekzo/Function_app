# app.py
import streamlit as st
import pandas as pd
from scipy.spatial import distance
import pickle
import json
import numpy as np

# Charger les données et le modèle au démarrage de l'application
with open("C:\\Users\\pcasaux\\OneDrive - Castelis\\Documents\\OCR\\P9\\dataset\\news-portal-user-interactions-by-globocom\\articles_embeddings.pickle", 'rb') as file_pi:
    articles_embeddings = pickle.load(file_pi)

clicks = pd.read_csv('C:\\Users\\pcasaux\\OneDrive - Castelis\\Documents\\OCR\\P9\\dataset\\news-portal-user-interactions-by-globocom\\clicks_tot.csv')

def top5(articles_embeddings, userId, clicks_final):
    # (votre code existant)
    
     # get all articles read by user
    var = clicks_final.loc[clicks_final['user_id']==userId]['click_article_id'].tolist()
    
    # chose last one --> le plus proche en terme de préférence
    value = var[-1]
    print(value)

    # get 5 articles the most similar to the selected one
    distances = distance.cdist([articles_embeddings[value]], articles_embeddings, "cosine")[0]
    
    # find indexes except the one selected
    result = np.argsort(distances)[1:6]
    

    # similarity value between selected article and 5 top proposed articles (the smaller the better!)
    similarite = distance.cdist([articles_embeddings[value]], articles_embeddings[result], "cosine")[0]
    return result, similarite

def get_recommendation(userID):
    result, similarite = top5(articles_embeddings, int(userID), clicks)
    return result.tolist()

# Définir l'interface utilisateur Streamlit
def main():
    st.title("Recommandation d'articles")

    # Obtenir le numéro d'utilisateur à partir de l'interface utilisateur
    userID = st.text_input("Entrez votre numéro d'utilisateur:")

    if st.button("Obtenir des recommandations"):
        if userID and userID.isdigit():
            recommendation_list = get_recommendation(userID)
            st.success("Recommandations: {}".format(recommendation_list))
        else:
            st.error("Veuillez entrer un numéro d'utilisateur valide.")

if __name__ == '__main__':
    main()
