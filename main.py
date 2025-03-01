import streamlit as st
import pandas as pd
import base64
from PIL import Image
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import gensim

nltk.download('stopwords')
nltk.download('punkt_tab')


stop_words=set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Stopwords in English
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens


 # Function to convert text to a vector (average word embeddings)
def text_to_vector(text, model):
    vector = np.zeros(100)  # 100 is the vector size we defined in Word2Vec
    num_words = 0
    for word in text:
        if word in model.wv:
            vector += model.wv[word]
            num_words += 1
    if num_words > 0:
        vector /= num_words  # Average the word vectors
    return vector   

#-------------------

if __name__ == '__main__':

    logo = Image.open("insurance_image.webp")  # Replace with your logo path

    # Adjust the size: Increase width and reduce height
    new_width = 600  # Set your desired width
    new_height = 200  # Set your desired height
    logo_resized = logo.resize((new_width, new_height))

    st.image(logo_resized)
    
    # Path to your image (either local or from a URL)

    st.header("Insurance Claims Prediction :books:")
    with st.sidebar:
        st.markdown('''
        ## About
        This Insurance Claims Prediction app is built using:
        - Streamlit Application
        ''')

    uploaded_file = st.sidebar.file_uploader("Upload your Excel/CSV here:", accept_multiple_files=False)
 

    if st.sidebar.button("Submit(Predictions using Random Forest)"):
        try:

            #read xls or xlsx
            if uploaded_file is not None:
                data=pd.read_excel(uploaded_file)
                st.write("Text & Predicted Result:")
                #st.write(data[['content']])

            word2vec = gensim.models.Word2Vec.load("word2vec_model.model")
            
            with open("label_encoder.pkl", "rb") as le_file:
                label_encoder = pickle.load(le_file)
	        
            #Apply the function on the dataset to perform the preprocessing data.
            unseen_data_processed = [preprocess_text(text) for text in data['content']]
            unseen_vectors = np.array([text_to_vector(text, word2vec) for text in unseen_data_processed])

                        # Load the pickled NLP model
            with open("rf_model_tunned.pkl", "rb") as file:
                model = pickle.load(file)
            	        
            
            # Make predictions on the unseen data
            predictions = model.predict(unseen_vectors)
            predicted_labels = label_encoder.inverse_transform(predictions)
            
            # Display predicted labels for the unseen data
            for i, text in enumerate(data['content']):
                #print(f"Text: '{text}' --> Predicted Label: {predicted_labels[i]}")

                st.markdown(f"**Text:** '{user_text}' → **Predicted Label:** <span style='color: darkgreen; font-weight: bold;'>{predicted_label}</span>", unsafe_allow_html=True)
		




        except Exception as e:
            st.write("Error :",e)

    
            
			
			
