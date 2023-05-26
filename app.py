import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from textblob import TextBlob
import cleantext

def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

def analyze(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

st.title('Customer Review Sentiment Analysis')

st.markdown('This Application is to anlyze the sentiment of customer review')

st.sidebar.title('Review Analysis')

df = pd.read_csv('sample_us.tsv', delimiter='\t')
df['score'] = df['review_body'].apply(score)
df['analysis'] = df['score'].apply(analyze)
df = df[['product_category', 'star_rating', 'review_headline', 'review_body', 'score', 'analysis']]

if st.checkbox('Show 20 Data Sample'):
    st.write(df.sample(20))

st.sidebar.subheader('Review Analyzer')
review = st.sidebar.radio('Sentiment Sample', ('Positive', 'Negative', 'Neutral'))
st.write(df.query('analysis==@review')[['review_body']].sample(1).iat[0,0])
st.write(df.query('analysis==@review')[['review_body']].sample(1).iat[0,0])
st.write(df.query('analysis==@review')[['review_body']].sample(1).iat[0,0])

select = st.sidebar.selectbox('Visualisation', ['Histogram', 'Pie Chart'], key=1)

sentiment = df['analysis'].value_counts()
sentiment = pd.DataFrame({'Sentiment': sentiment.index, 'Reviews': sentiment.values})
st.markdown('Sentiment Count')
if select == 'Histogram':
    fig = px.bar(sentiment, x='Sentiment', y='Reviews', color='Reviews', height=500)
    st.plotly_chart(fig)
else:
    fig = px.pie(sentiment, values='Reviews', names='Sentiment')
    st.plotly_chart(fig)


