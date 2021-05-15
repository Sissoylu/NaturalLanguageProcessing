from sklearn.feature_extraction.text import TfidfVectorizer
import nltk, string
import numpy as np
nltk.download('punkt') # if necessary...
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def stem_tokens(tokens): return [stemmer.stem(item) for item in tokens]
def normalize(text): return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
corpus = ["I'd like an apple", 'An apple a day keeps the doctor away', 'Never compare an apple to an orange', 'I prefer scikit-learn to Orange', 'The scikit-learn docs are Orange and Blue']
vect = TfidfVectorizer(tokenizer=normalize, stop_words='english')
tfidf = vect.fit_transform(corpus)
pairwise_similarity = tfidf * tfidf.T #view the pairwise similarities print(pairwise_similarity) #check how a string is normalized print(normalize('The scikit-learn docs are Orange and Blue'))

pairwise_similarity.toarray()

arr = pairwise_similarity.toarray()
np.fill_diagonal(arr, np.nan)
input_doc = 'The scikit-learn docs are Orange and Blue'
input_idx = corpus.index(input_doc)
result_idx = np.nanargmax(arr[input_idx])
res = corpus[result_idx]
print(res) #I prefer scikit-learn to Orange
