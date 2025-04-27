import streamlit as st
import pandas as pd
import pickle
import re
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.graph_objs as go
from reviewscrapper import *
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
text = "Finally my issue of nltk is resolved"
file_path = Path("./notebooks/models.p")
tokens = word_tokenize(text,language='english', preserve_line=True)

def preprocess_text(text):
    # Make text lowercase and remove links, text in square brackets, punctuation, and words containing numbers
    text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', ' ', text)
    text = re.sub(r'\n', ' ', text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words).strip()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lem_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(lem_tokens)

def display_result(result):
    if result[0]=="Positive":
        st.subheader(result[0]+":smile:")
    elif result[0]=="Negative":
        st.subheader(result[0]+":pensive:")
    else:
        st.subheader(result[0]+":neutral_face:")

def classify_multiple(dataframe):
    st.write(f"There are a total of {dataframe.shape[0]} reviews given")

    dataframe.columns = ["Review"]
    data = dataframe.copy()
    data["Review"].apply(preprocess_text)
    count_positive = 0
    count_negative = 0
    count_neutral = 0
    sentiments = []
    for i in range(dataframe.shape[0]):
        rev = data.iloc[i]["Review"]
        rev_vectorized = vect.transform([rev])
        res = model.predict(rev_vectorized)
        sentiments.append(res[0])
        if res[0]=='Positive':
            count_positive+=1
        elif res[0]=='Negative':
            count_negative+=1
        else:
            count_neutral+=1 

    x = ["Positive", "Negative", "Neutral"]
    y = [count_positive, count_negative, count_neutral]

    fig = go.Figure()
    layout = go.Layout(
        title='Product Reviews Analysis',
        xaxis=dict(title='Sentiment Category'),
        yaxis=dict(title='Number of reviews'),
        paper_bgcolor='#f6f5f6',  # Background color
        font=dict(color='#0e0d0e')  # Text color
    )

    fig.update_layout(layout)
    fig.add_trace(go.Bar(name='Multi Reviews', x=x, y=y, marker_color='#8d7995'))  # Bar color
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Positive: {count_positive}, Negative: {count_negative}, Neutral: {count_neutral}")
    
    # Word Cloud
    wordcloud_data = " ".join(dataframe["Review"].astype(str))
    
    if wordcloud_data.strip() == "":
       st.write("No reviews available to generate a word cloud.")
    else:
       wordcloud = WordCloud(width=800, height=400, max_words=100, background_color="#f6f5f6", colormap='viridis').generate(wordcloud_data)
       # Set the color scheme of the Word Cloud
       wordcloud.recolor(color_func=lambda *args, **kwargs: "#8d7995")
       fig_wordcloud = plt.figure(figsize=(8, 4), facecolor="#f6f5f6")
       plt.imshow(wordcloud, interpolation="bilinear")
       plt.axis("off")
       plt.title('Word Cloud - Most Frequent Words', color='#0e0d0e')  # Set text color
       plt.gca().set_facecolor("#f6f5f6")  # Set background color for the entire plot
       st.pyplot(fig_wordcloud, use_container_width=True)


    dataframe["Sentiment"] = sentiments
    st.dataframe(dataframe, use_container_width=True)

def web_scraper(url, max_pages=2):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in background
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    service = Service(executable_path="C:/Users/vedant/chromedriver.exe")  # Replace with full path if not in PATH
    driver = webdriver.Chrome(service=service, options=options)
    
    all_reviews = []

    for page in range(1, max_pages + 1):
        paginated_url = f"{url}&pageNumber={page}"
        print(f"Scraping Page {page}")
        driver.get(paginated_url)
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        reviews = soup.find_all("span", {"data-hook": "review-body"})

        for review in reviews:
            text = review.get_text(strip=True)
            if text:
                all_reviews.append(text)

    driver.quit()
    
    if not all_reviews:
        print("No reviews found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_reviews, columns=["Review"])
    return df

if __name__ == "__main__":
    st.title('Sentiment Analysis of Amazon product reviews!')
    st.divider()
    classifier = st.radio(
        "Which classifier do you want to use?",
        ["Logistic Regression", "Support Vector Machine (SVM)"])
    if classifier == 'Logistic Regression':
        st.write('You selected Logistic Regression')
    else:
        st.write("You selected SVM")
    st.divider()
    
    with open(file_path, 'rb') as mod:
            data = pickle.load(mod)
    vect = data['vectorizer']
    print(type(data))
    print(data.keys())

    if classifier=="Logistic Regression":
        model = data["logreg"]
    else:
        model = data["svm"]


    st.subheader('Check sentiments of a single review:')
    single_review = st.text_area("Enter review:")
    if st.button('Check the sentiment!'):
        review = preprocess_text(single_review)
        inp_test = vect.transform([single_review])
        result = model.predict(inp_test)
        print(result)
        display_result(result)
        
    else:
        st.write('')

    st.divider()
    st.subheader('Check sentiments of an Amazon product:')
    url_review = st.text_input("Enter the URL to the product:")
    if st.button('Check the reviews!'):
        try:
           df_reviews = web_scraper(url_review)
           if df_reviews.empty:
              st.warning("No reviews were found.")
           else:
            st.success(f"Fetched {len(df_reviews)} reviews.")
            st.dataframe(df_reviews)
            classify_multiple(df_reviews)  # Your sentiment classification function
        except Exception as e:
            st.error(f"Error: {e}")
    else:
       st.write('')
        
    st.divider()
    st.subheader('Check sentiments of multiple reviews:')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        if dataframe.shape[1]!=1:
            st.write("Wrong CSV format!")
        else:
            classify_multiple(dataframe)