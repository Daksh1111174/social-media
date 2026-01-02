import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Reddit
import praw

# Twitter scraping
import snscrape.modules.twitter as sntwitter

# ---- Reddit Auth Config ----
reddit = praw.Reddit(
    client_id="YOUR_ID",
    client_secret="YOUR_SECRET",
    user_agent="trendAnalyzer"
)

# ---- TFIDF & Wordcloud Function ----
def compute_tfidf_wordcloud(texts):

    vec = TfidfVectorizer(stop_words='english', max_features=100)
    X = vec.fit_transform(texts)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
    freq = df.sum().sort_values(ascending=False)

    # Wordcloud
    wc = WordCloud(width=800, height=400).generate_from_frequencies(freq.to_dict())
    return freq, wc

# ---- UI ----
st.title("Social Media Big Data Analyzer")

tab1, tab2, tab3 = st.tabs(["Twitter", "Facebook", "Reddit"])

# ---------------- TWITTER ----------------
with tab1:
    keyword = st.text_input("Enter Trending Topic for Twitter", key="tw")
    if st.button("Fetch Twitter Data"):
        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()):
            if i >= 500:
                break
            tweets.append(tweet.content)

        df = pd.DataFrame({"text": tweets})
        st.write("Twitter Data", df)

        freq, wc = compute_tfidf_wordcloud(tweets)
        st.write("TF-IDF Frequencies", freq)

        st.pyplot(plt.figure(figsize=(10,5)))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

# ---------------- FACEBOOK ----------------
with tab2:
    fb_keyword = st.text_input("Enter Facebook Search Keyword", key="fb")
    if st.button("Fetch Facebook Data"):
        st.write("Note: Facebook API access is required.")
        st.write("Use Graph API with page/search endpoints.")

# ---------------- REDDIT ----------------
with tab3:
    reddit_keyword = st.text_input("Enter Reddit Topic or Subreddit", key="rd")
    if st.button("Fetch Reddit Data"):
        posts = []
        for submission in reddit.subreddit("all").search(reddit_keyword, limit=500):
            posts.append(submission.title + " " + submission.selftext)

        df = pd.DataFrame({"text": posts})
        st.write("Reddit Data", df)

        freq, wc = compute_tfidf_wordcloud(posts)
        st.write("TF-IDF Frequencies", freq)

        st.pyplot(plt.figure(figsize=(10,5)))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
