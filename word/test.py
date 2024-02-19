from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize  # Make sure to install nltk: pip install nltk

# Sample data for training the Word2Vec model
corpus = [
    "I like to eat pizza",
    "Pizza is my favorite food",
    "I enjoy coding in Python",
    "Python is a powerful programming language",
]

# Tokenize the sentences
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train the Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Find similarity between two words
word1 = "pizza"
word2 = "coding"

similarity = model.wv.similarity(word1, word2)

print(f"Similarity between '{word1}' and '{word2}': {similarity}")