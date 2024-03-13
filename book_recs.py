import streamlit as st
from streamlit import session_state as session
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSag40XZ-i6WjD6taPwiwbniWLp8mR3NCJIh6Q-8_3OarBQarC8xpRxInmWOVm3JD93554&usqp=CAU");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

# https://i.postimg.cc/4xgNnkfX/Untitled-design.png

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    body {
        background-color: #feefe8;
        font-family: Arial, sans-serif;
        color: #973aa8;
    }
    .stButton>button {
        background-color: #973aa8;
        color: #white;
    }
    
    .stSelectbox div[data-baseweb="select"] > div:first-child {
        background-color: white;
        border-color: #2d408d;
        color: #973aa8;
    }

    .stSelectbox [data-testid='stMarkdownContainer'] {
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

def string_to_array(s):
    s = s[1:-1]
    return np.fromstring(s, sep=' ')

f = (lambda x: x if x != '\n' else '')
def delete_newline(sentence):
    words = sentence.split()
    result = [f(word) for word in words]
    return ' '.join(result)

def recommend_books_st(user_book, book_df, top_n = 5):
    user_embeddings = book_df[book_df['title_without_series'] == user_book]['embedding_avg'].iloc[0]
    other_embeddings = np.array(book_df[book_df['title_without_series'] != user_book]['embedding'])
    cos_sim_st = cosine_similarity(user_embeddings.reshape((1, -1)), np.vstack(other_embeddings))
    indices_st = np.argsort(cos_sim_st.flatten())[::-1]
    return book_df.iloc[indices_st]['title_without_series'].unique()[:top_n]

def recommend_books_tfidf(user_book, book_df, top_n = 5):
    user_array = book_df[book_df['title_without_series'] == user_book]['tfidf_array'].iloc[0]
    other_array = np.array(book_df[book_df['title_without_series'] != user_book]['tfidf_array'])
    cos_sim_tfidf = cosine_similarity(user_array.reshape((1, -1)), np.vstack(other_array))
    indices_tfidf = np.argsort(cos_sim_tfidf.flatten())[::-1]
    return book_df.iloc[indices_tfidf]['title_without_series'].unique()[:top_n]

import textwrap

def get_top_neg_reviews(title, book_df):  
    neg_df = book_df[(book_df['title_without_series'] == title) & (book_df['rating'] <= 2)].sort_values('roberta_neg', ascending=False)
               
    for i in range(min(3, len(neg_df))):
        result = neg_df['review_text'].values[i]
        formatted_result = f"{i + 1}.\n{textwrap.fill(result, width=80, initial_indent='   ', subsequent_indent='   ')}\n"
        st.text(formatted_result)
        st.text("")
        
def get_top_pos_reviews(title, df):
    filtered_df = book_df[(book_df['title_without_series'] == title) & (book_df['rating'] >= 4)]
    
    if len(filtered_df[filtered_df['rating'] == 5]) < 3:
        filtered_df = book_df[(book_df['title_without_series'] == title) & (book_df['rating'] >= 3)]

    sorted_df = filtered_df.sort_values('roberta_pos', ascending=False)

    for i in range(min(3, len(sorted_df))):
        result = sorted_df['review_text'].values[i]
        formatted_result = f"{i + 1}.\n{textwrap.fill(result, width=80, initial_indent='   ', subsequent_indent='   ')}\n"
        st.text(formatted_result)
        st.text("")

@st.cache_data
def load_data():
    book_data = pd.read_csv('book_df.csv', index_col=0)
    
    book_data['embedding_avg'] = book_data['embedding_avg'].apply(delete_newline)
    book_data['embedding'] = book_data['embedding'].apply(delete_newline)

    book_data['embedding_avg'] = book_data['embedding_avg'].apply(lambda x: string_to_array(x))
    book_data['embedding'] = book_data['embedding'].apply(lambda x: string_to_array(x))

    book_data['tfidf_array'] = book_data['tfidf_array'].apply(delete_newline)
    book_data['tfidf_array'] = book_data['tfidf_array'].apply(lambda x: string_to_array(x))
    
    return book_data

book_df = load_data()

st.title('Goodreads Recommendations :books::heart_eyes:')

st.text("")
st.text("")
st.text("")
st.text("")

session.options = st.selectbox(label="Select Books", options=book_df.sort_values(by='ratings_count', ascending = False)['title_without_series'].unique())

buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

rec_button = col1.button(label="Recommend")

st.text("")
st.text("")
st.text("")

if rec_button:
    # st.table(recommend_books(session.options, book_df, 5))
    st_books = recommend_books_st(session.options, book_df, 5)
    tfidf_books = recommend_books_tfidf(session.options, book_df, 5)
    st.header('Recommended Books')
    st.table(pd.DataFrame({'SentenceTransformers': st_books, 
                           'TF-IDF': tfidf_books}))

    row1 = st.columns(5)
    row2 = st.columns(5)

    i = 0
    for col in row1:
        url = book_df.loc[book_df['title_without_series'] == st_books[i]]['image_url'].iloc[0]
        with col:
            st.image(url)
        i = i + 1

    i = 0
    for col in row2:
        url = book_df.loc[book_df['title_without_series'] == tfidf_books[i]]['image_url'].iloc[0]
        with col:
            st.image(url)
        i = i + 1

    st.text("")
    st.text("")

    st.header('Top Positive Reviews')
    get_top_pos_reviews(session.options, book_df)
    st.header('Top Negative Reviews')
    get_top_neg_reviews(session.options, book_df)



    