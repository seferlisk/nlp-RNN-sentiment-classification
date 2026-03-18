import pandas as pd
import numpy as np
from collections import Counter

class TweetDataPreprocessor:
    """
        Handles the loading, vocabulary building, tokenization, and padding
        of the preprocessed tweet dataset.

        Attributes:
            csv_path (str): Path to the CSV dataset.
            vocab_size (int): Total number of unique words in the vocabulary + 1 (for padding).
            max_length (int): The 95th percentile length of the sequences used for padding.
            word2idx (dict): Mapping of words to their integer indices.
        """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.vocab_size = 0
        self.max_length = 0
        self.word2idx = None
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    def process_data(self):
        """
        Reads the data, builds a frequency-based vocabulary, maps words to integers,
        and pads the sequences dynamically based on the 95th percentile length.

        Returns:
            tuple: (padded_texts array, labels array)
        """
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=['processed_text'])

        texts = df['processed_text'].values
        labels = df['airline_sentiment'].map(self.label_map).values

        # Build Vocabulary
        words = ' '.join(texts).split()
        word_counts = Counter(words)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

        self.word2idx = {word: i + 1 for i, word in enumerate(sorted_vocab)}  # 0 is for padding
        self.vocab_size = len(self.word2idx) + 1

        # Tokenize
        encoded_texts = [[self.word2idx[word] for word in text.split()] for text in texts]

        # Calculate max_length and pad
        self.max_length = int(np.percentile([len(seq) for seq in encoded_texts], 95))
        padded_texts = np.zeros((len(encoded_texts), self.max_length), dtype=int)

        for i, seq in enumerate(encoded_texts):
            length = min(len(seq), self.max_length)
            padded_texts[i, :length] = seq[:length]

        return padded_texts, labels