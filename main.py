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
import tensorflow_hub as hub
import tensorflow_text as text

nltk.download('stopwords')

stop_words=set(stopwords.words('english'))

def return_only_words(text):
    text=re.sub(r'[^a-zA-Z0-9-\s]','',text)
    text=text.lower()
    # Replace hyphens with spaces
    text = re.sub(r'-', ' ', text)
    text=word_tokenize(text)

    return ' '.join([word for word in text if word not in stop_words])

# Create sentence embeddings by averaging word embeddings
def sentence_to_vector(sentence, model, embedding_size):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)  # Average word vectors
    else:
        return np.zeros(embedding_size)  # Return a zero vector if no words are found

# #Creation Functions to create embeddings
# def get_text_embeddings(text):
#   preprocess=bert_preprocess(text)
#   encoder   =bert_encoder(preprocess)

#   return encoder["pooled_output"]


def get_text_embeddings(text_series):
    """Returns BERT embeddings for a Pandas Series of text."""
    text_list = text_series.tolist()  # Convert Pandas Series to list
    preprocessed_text = bert_preprocess(text_list)  # Preprocess text
    encoder_output = bert_encoder(preprocessed_text)  # Encode with BERT
    return encoder_output["pooled_output"].numpy()  # Convert to NumPy for readability




if __name__ == '__main__':

    logo = Image.open("insurance_image.webp")  # Replace with your logo path

    # Adjust the size: Increase width and reduce height
    new_width = 600  # Set your desired width
    new_height = 200  # Set your desired height
    logo_resized = logo.resize((new_width, new_height))

    st.image(logo_resized)
    
    # Path to your image (either local or from a URL)

    st.header("Insurance Claims Prediction :books:")

    uploaded_file = st.sidebar.file_uploader("Upload your Excel/CSV here:", accept_multiple_files=False)
 

    if st.button("Submit(Predictions using Random Forest)"):
        try:

            #read xls or xlsx
            if uploaded_file is not None:
                data=pd.read_excel(uploaded_file,engine="openpyxl")
                st.write("Uploaded Data:")
                st.write(data[['content']])
	        
            #Apply the function on the dataset to perform the preprocessing data.
            data['updated_content']=data['content'].apply(return_only_words)
	        
            # Function to count words in a dataframe column
            max_words_size=max(data['updated_content'].apply(lambda x: len(x.split())))
            #print(max_words_size)
	        
            # Preprocessing: Tokenize the sentences in the 'text_column'
            sentences = [sentence.split() for sentence in data['updated_content']]
            #sentences
	        
            embedding_size=max_words_size+20
            word2vec_model=Word2Vec(sentences,vector_size=embedding_size, window=3, min_count=1, workers=2)
	        
            # Create feature matrix X by applying sentence_to_vector to each sentence
            X = np.array([sentence_to_vector(sentence, word2vec_model, embedding_size) for sentence in sentences])
            st.write(X)

            # Load the pickled NLP model
            with open("rf_model.pkl", "rb") as file:
                model = pickle.load(file)
	        
            # Test the model (example for a text classification model)
            predictions  = model.predict(X)
            predictions = predictions.flatten() if predictions.ndim > 1 else predictions  # Flatten if needed
	        
            # Convert predictions to a Pandas Series before adding to DataFrame
            data['Predicted'] = pd.Series(predictions, index=data.index)
	        
            # Reverse mapping dictionary
            mapping_dict = {
            0: 'Bodily Injury',
            1: 'Other',
            2: 'Property Damage',
            3: 'Uninsured or Underinsured'
}           
	        
          # Apply mapping to 'Predicted' column
            data['Predicted_Result'] = data['Predicted'].map(mapping_dict).fillna('Unknown')
	        
          #  Display updated DataFrame
            st.write("Prediction Result:")
            #st.write(data[['content','Predicted','Predicted_Result']])
            st.write(data[['content','Predicted_Result']])

        except Exception as e:
            pass

    if st.button("Submit(Predictions using BERT & Neural Network)"):

        try:
            bert_preprocess=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",name="bert_preprocess")
            bert_encoder   =hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",name="bert_encoder")

            # Load BERT Preprocessing & Encoder
            #bert_preprocess = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
            #bert_encoder = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

            #read xls or xlsx
            if uploaded_file is not None:
                data=pd.read_excel(uploaded_file,engine="openpyxl")
                st.write("Uploaded Data:")
                st.write(data[['content']])

            st.write(get_text_embeddings(data['content']))

        except Exception as e:
            st.write(e)

            
		
