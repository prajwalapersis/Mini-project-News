import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from newspaper import Article
import openpyxl
import os

# Load AG News Dataset for categorization
ag_news = load_dataset('ag_news')

# Convert AG News dataset to DataFrame for preprocessing
ag_news_df = pd.DataFrame({
    'text': ag_news['train']['text'],
    'label': ag_news['train']['label']
})

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Vectorize the text data
X = vectorizer.fit_transform(ag_news_df['text'])
y = ag_news_df['label']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn.fit(X_train, y_train)

# Validate the classifier
y_pred = knn.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy}')

# Load BART model and tokenizer for summarization
summarizer_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
summarizer_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
summarizer_pipeline = pipeline('summarization', model=summarizer_model, tokenizer=summarizer_tokenizer)

def extract_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article

def categorize_text(text):
    text_vector = vectorizer.transform([text])
    category = knn.predict(text_vector)
    return ag_news['train'].features['label'].int2str(int(category[0]))

def summarize_text(text):
    # Split the text into manageable chunks
    max_chunk_length = 1024
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    # Summarize each chunk and concatenate summaries
    summaries = []
    for chunk in chunks:
        summary = summarizer_pipeline(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)

def save_to_excel(data, filename="news_summary.xlsx"):
    if os.path.exists(filename):
        # Load existing data
        existing_df = pd.read_excel(filename)
        # Convert new data to DataFrame
        new_df = pd.DataFrame(data)
        # Concatenate the new data with the existing data
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Convert new data to DataFrame
        df = pd.DataFrame(data)
    
    # Save the concatenated DataFrame back to the Excel file
    df.to_excel(filename, index=False)

def main(url):
    # Extract article text
    article = extract_article_text(url)
    article_text = article.text

    # Categorize the article
    category = categorize_text(article_text)

    # Summarize the article
    summary = summarize_text(article_text)

    # Prepare the data
    data = {
        'headline': article.title,
        'text': article_text,
        'summary': summary,
        'category': category,
        'image': article.top_image,
        'url': url,
        'source': article.source_url
    }

    return data

# Example usage
url = "https://timesofindia.indiatimes.com/home/science/nasa-overlooked-minor-helium-leak-in-boeing-starliner-pre-launch-leaving-sunita-williams-butch-wilmore-stranded-at-iss-report/articleshow/111216994.cms"
result = main(url)

# Save the result to Excel
save_to_excel([result])
print("Data saved to Excel:")
print("URL:", result['url'])
print("Category:", result['category'])
print("Summary:", result['summary'])
print("Headline:", result['headline'])
print("Image URL:", result['image'])
