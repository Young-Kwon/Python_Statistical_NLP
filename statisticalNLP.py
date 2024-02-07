"""
Statistical NLP

Description: A chatbot that provides answers to user questions using fuzzy pattern matching.
Function: load_articles
          vectorize_documents
          calculate_similarity
          generate_recommendations          
          display_recommendations
          display_article
          main
Author: Young Sang Kwon
Date: Nov 17th, 2023
Version: 1.0
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from csv import DictReader
from random import sample, randint
import json
import numpy as np

NUM_RECS = 100 # number of recommendations to return to the user

def load_articles(filename, num=None, filetype="csv"):
    """
    Load a list of articles from a specified file.

    Args:
        filename (str): Name of the file to load.
        num (int, optional): Number of articles to load. Loads all if None.
        filetype (str): Type of file - 'csv' or 'json'.

    Returns:
        list: A list of articles, each represented as a dictionary.
    """
    articles = []
    if filetype=="csv":
        with open(filename, encoding="utf-8") as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                articles.append(row)
    elif filetype=="json":
        with open(filename, encoding="utf-8") as jsonfile:
            articles = json.loads(jsonfile.read())
    for row in articles:
        if row["title"] == None:
            row["title"] = row["text"][:30]
    if num:
        shuffle(articles)
        articles = articles[:num]
    print(len(articles),"articles loaded")
    return articles

def vectorize_documents(articles):
    """
    Vectorize the documents using TfidfVectorizer.

    Args:
        articles (list): List of articles to be vectorized.

    Returns:
        sparse matrix: TfidfVectorizer output of vectorized documents.
    """
    vectorizer = TfidfVectorizer()
    docs = [article['text'] for article in articles]
    return vectorizer.fit_transform(docs)

def calculate_similarity(doc_vectors):
    """
    Calculate cosine similarity among document vectors.

    Args:
        doc_vectors (sparse matrix): Vectorized documents.

    Returns:
        ndarray: Matrix of cosine similarities.
    """
    return cosine_similarity(doc_vectors)

def generate_recommendations(last_choice, n, articles, similarity_matrix):
    """
    Generate document recommendations based on similarity.

    Args:
        last_choice (int): Index of the last article read.
        n (int): Number of recommendations.
        articles (list): List of articles.
        similarity_matrix (ndarray): Matrix of cosine similarities.

    Returns:
        list: Indices of recommended articles.
    """
    similar_articles = np.argsort(-similarity_matrix[last_choice])[:n + 10]
    
    filtered_articles = [i for i in similar_articles if i != last_choice and articles[i]['title'] != articles[last_choice]['title']][:n]

    dissimilar_articles = [i for i in range(len(articles)) if i not in filtered_articles][:2]
    return filtered_articles + dissimilar_articles

def display_recommendations(recommendations, articles):
    """
    Display a list of recommended articles to the user.

    Args:
        recommendations (list): List of indices representing recommended articles.
        articles (list): List of all articles.

    Returns:
        None: This function only prints the recommended articles' titles to the console.
    """
    print("\n\n\nHere are some new recommendations for you:\n")
    for i in range(len(recommendations)):
        art_num = recommendations[i]
        print(str(i+1)+".",articles[art_num]["title"])

def display_article(art_num, articles):
    """
    Display the details of a specific article, including its title and text.

    Args:
        art_num (int): Index number of the article to be displayed.
        articles (list): List of all articles.

    Returns:
        None: This function only prints the details of the article to the console.
    """
    print("\n\n")
    print("article",art_num)
    print("=========================================")
    print(articles[art_num]["title"])
    print()
    print(articles[art_num]["text"])
    print("=========================================")
    print("\n\n")

def main():
    """
    Main function to run the article recommendation system.

    Allows user to select a dataset and displays article recommendations.
    """
    print("Choose a dataset:")
    print("1. BBC News Articles (bbc_news.csv)")
    print("2. Wikipedia Sample (wikipedia_sample.json)")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        filename = 'data/bbc_news.csv'
        filetype = 'csv'
    elif choice == '2':
        filename = 'data/wikipedia_sample.json'
        filetype = 'json'
    else:
        print("Invalid choice. Exiting program.")
        return

    articles = load_articles(filename, filetype=filetype)
    doc_vectors = vectorize_documents(articles)
    similarity_matrix = calculate_similarity(doc_vectors)
    
    recs = sample(range(len(articles)), NUM_RECS)
    while True:
        display_recommendations(recs, articles)
        user_choice = input("\nYour choice? (Enter 'q' to quit) ")
        if user_choice.lower() == 'q':
            print("Goodbye!")
            break

        try:
            user_choice = int(user_choice) - 1
            if user_choice < 0 or user_choice >= len(recs):
                print("Invalid choice. Please try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        article_index = recs[user_choice]
        if article_index < 0 or article_index >= len(articles):
            print("Article number out of range. Please try again.")
            continue

        display_article(article_index, articles)
        input("Press Enter")
        recs = generate_recommendations(article_index, NUM_RECS, articles, similarity_matrix)

if __name__ == "__main__":
    main()