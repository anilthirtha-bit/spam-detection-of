from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

messages = [
    "Win a free iPhone now",
    "Claim your prize immediately",
    "Congratulations you won money",
    "Meeting at 10 AM tomorrow",
    "Please review the report",
    "Let's have lunch today"
]

labels = [1, 1, 1, 0, 0, 0]  # 1=spam, 0=ham

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved")
