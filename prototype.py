import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd


# Download necessary NLTK resources (only if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Sample data (replace this with your actual dataset)
data = {
    'message': [
        "Congratulations! You have won a free ticket! Click here to claim your prize.",
        "Dear user, your account verification is needed. Please confirm your account.",
        "We are offering a risk-free investment opportunity. Act now!",
        "Hello, I just wanted to check in. How have you been?",
        "Important: Immediate action required regarding your account settings."
    ],
    'label': [1, 1, 1, 0, 1]  # 1 for manipulative, 0 for non-manipulative
}

# Create a DataFrame
df = pd.DataFrame(data)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = LogisticRegression()
model.fit(X_train, y_train)

# Particle class representing each message
class Particle:
    def __init__(self, message):
        self.message = message
        self.score = 0  # Score based on manipulation detection
        self.best_score = 0  # Best score observed
        self.best_message = message  # Best message observed

    def evaluate(self):
        """Evaluate the message using the machine learning model."""
        message_vector = vectorizer.transform([self.message])
        prediction = model.predict(message_vector)
        self.score = prediction[0]  # Update score based on prediction
        # Update best score and message
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_message = self.message

def main():
    # Sample messages for testing
    messages = [
        "Congratulations! You have won a free ticket! Click here to claim your prize.",
        "Dear user, your account verification is needed. Please confirm your account.",
        "We are offering a risk-free investment opportunity. Act now!",
        "Hello, I just wanted to check in. How have you been?",
        "Important: Immediate action required regarding your account settings."
    ]

    # Initialize particles
    swarm = [Particle(msg) for msg in messages]

    # Evaluate each particle
    for particle in swarm:
        particle.evaluate()

    # Sort particles by score
    most_manipulative = sorted(swarm, key=lambda p: p.score, reverse=True)

    print("Analysis Results:")
    for particle in most_manipulative:
        print(f"Message: {particle.message}\nManipulative Score: {particle.score}\n")

    # Implement a simple collaborative mechanism (not a full PSO)
    for particle in swarm:
        # Here we could allow particles to influence each other
        # For simplicity, we just print the best score found in the swarm
        print(f"Best score found in swarm: {particle.best_score} for message: '{particle.best_message}'")

if __name__ == "__main__":
    main()
