import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

nltk.download('punkt')
nltk.download('wordnet')

# Define target products
target_products = [
    "Credit card",
    "Personal loan",
    "Buy Now, Pay Later (BNPL)",
    "Savings account",
    "Money transfers"
]

def clean_text(text):
    """
    Clean and normalize the text narrative.
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join back into string
    return ' '.join(tokens)

def preprocess_data(input_path, output_path, sample_rows=None):
    """
    Preprocess the complaint data and save to output path.
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Take a sample if specified
    if sample_rows is not None:
        df = df.sample(n=min(sample_rows, len(df)), random_state=42)
    
    # Filter for target products
    df = df[df['Product'].isin(target_products)]
    
    # Filter out rows without narratives
    df = df[df['Consumer complaint narrative'].notna()]
    
    # Clean the narratives
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)
    
    # Drop rows with empty cleaned narratives
    df = df[df['cleaned_narrative'].str.len() > 0]
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    input_path = 'data/raw/complaints.csv'
    output_path = 'data/processed/filtered_complaints.csv'
    # Use a smaller sample for development/testing
    sample_size = 20000 # Adjusted sample size to prevent memory crashes
    df = preprocess_data(input_path, output_path, sample_rows=sample_size)
    print(f"Processed {len(df)} complaints")
    print(f"Distribution by product:\n{df['Product'].value_counts()}")
