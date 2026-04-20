from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

texts = [
    "air pollution increasing urban areas",
    "students lack access quality education",
    "healthcare systems need better ai support"
]

vectorizer = TfidfVectorizer(max_features=10)
features = vectorizer.fit_transform(texts).toarray()

np.save("features.npy", features)
