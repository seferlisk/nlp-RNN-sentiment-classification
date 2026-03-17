import pandas as pd
import numpy as np
from collections import Counter

class TweetDataPreprocessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.vocab = None
        self.word2idx = None
        self.vocab_size = 0
        self.max_length = 0
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    def process_data(self):
        # Load data and drop any rows with missing text
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=['processed_text'])

        texts = df['processed_text'].values
        labels = df['airline_sentiment'].map(self.label_map).values

        # Build Vocabulary
        words = ' '.join(texts).split()
        word_counts = Counter(words)
        # Sort words by frequency
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.word2idx = {word: i + 1 for i, word in enumerate(sorted_vocab)}  # 0 is reserved for padding
        self.vocab_size = len(self.word2idx) + 1

        # Tokenize texts
        encoded_texts = [[self.word2idx[word] for word in text.split()] for text in texts]

        # Dynamically calculate max_length based on 95th percentile
        self.max_length = int(np.percentile([len(seq) for seq in encoded_texts], 95))

        # Pad sequences
        padded_texts = np.zeros((len(encoded_texts), self.max_length), dtype=int)
        for i, seq in enumerate(encoded_texts):
            length = min(len(seq), self.max_length)
            padded_texts[i, :length] = seq[:length]

        return padded_texts, labels