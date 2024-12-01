import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pyswarms as ps
import joblib
import os

# Step 1: Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Load the cleaned dataset
data = pd.read_csv('dataset.csv')

# Step 3: Preprocess the text data
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = text.split()
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['message'] = data['message'].apply(preprocess_text)

# Step 4: Split the data into training and testing sets
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert text data into numerical features using TF-IDF
tfidf = TfidfVectorizer(max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Save the TF-IDF vectorizer for later use
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Step 6: Check if a saved model exists
if os.path.exists('optimized_model.pkl') and os.path.exists('best_C.pkl'):
    # Load the saved model and the optimized hyperparameter C
    final_model = joblib.load('optimized_model.pkl')
    best_C = joblib.load('best_C.pkl')
    print(f"Loaded model with C = {best_C}")
else:
    # Step 7: Define the objective function for PSO to optimize Logistic Regression hyperparameters
    def objective_function(params):
        """Objective function for PSO, takes an array of parameters and returns negative accuracy."""
        C = params[0]  # Regularization parameter for Logistic Regression

        # Ensure C is positive
        if C <= 0:
            return float('inf')

        # Train the Logistic Regression model
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train_tfidf, y_train)

        # Predict on the validation set and calculate accuracy
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        # Return negative accuracy because PSO minimizes
        return -accuracy

    # Step 8: Wrapper function for PSO to evaluate multiple particles
    def fitness_function(hyperparams):
        """Fitness function to evaluate each particle's accuracy."""
        n_particles = hyperparams.shape[0]
        scores = []
        for i in range(n_particles):
            scores.append(objective_function(hyperparams[i]))
        return np.array(scores)

    # Step 9: Set bounds for the hyperparameter(s)
    # We are only optimizing `C`, the bounds will be between 0.01 and 10.0
    bounds = (np.array([0.01]), np.array([10.0]))

    # Step 10: Set up and run PSO
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # Hyperparameters for the swarm (inertia, cognitive, social)
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=1, options=options, bounds=bounds)
    best_C, _ = optimizer.optimize(fitness_function, iters=20)

    # Step 11: Ensure the optimized value of C is valid
    best_C = abs(best_C)  # Make sure that the value of C is positive

    # Step 12: Train the final Logistic Regression model with the optimized parameter
    final_model = LogisticRegression(C=best_C, max_iter=1000)
    final_model.fit(X_train_tfidf, y_train)

    # Step 13: Save the final model and the optimized hyperparameter C
    joblib.dump(final_model, 'optimized_model.pkl')
    joblib.dump(best_C, 'best_C.pkl')
    print(f"Model trained and saved with C = {best_C}")

# Step 14: Evaluate the final model
y_pred_final = final_model.predict(X_test_tfidf)
print("Optimized Accuracy:", accuracy_score(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final))
