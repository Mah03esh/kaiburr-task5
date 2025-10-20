"""
Task 5: Text Classification of Consumer Complaints
This script performs text classification on consumer complaints data
to categorize them into four product categories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("Warning: Could not download NLTK data. Continuing anyway...")

# ============================================================================
# STEP 0: SETUP AND DATA LOADING
# ============================================================================

def load_and_filter_data(filepath='complaints.csv'):
    """
    Load the complaints dataset and filter for the four target categories.
    """
    print("=" * 80)
    print("STEP 0: LOADING AND FILTERING DATA")
    print("=" * 80)
    
    # Load the dataset
    import os
    # Try multiple possible paths
    possible_paths = [
        filepath,  # Current directory
        os.path.join('task5', filepath),  # From parent directory
        os.path.join(os.path.dirname(__file__), filepath)  # Same dir as script
    ]
    
    actual_filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_filepath = path
            break
    
    if actual_filepath is None:
        raise FileNotFoundError(f"Could not find {filepath}")
    
    df = pd.read_csv(actual_filepath)
    print(f"Original dataset shape: {df.shape}")
    
    # Define the four target product categories
    target_products = [
        "Credit reporting, repair, or other",
        "Debt collection",
        "Consumer Loan",
        "Mortgage"
    ]
    
    # Filter data for only these four categories
    df_filtered = df[df['Product'].isin(target_products)].copy()
    print(f"Filtered dataset shape: {df_filtered.shape}")
    
    # Select relevant columns
    df_filtered = df_filtered[['Consumer complaint narrative', 'Product']]
    
    # Handle missing values - drop rows without complaint narrative
    df_filtered = df_filtered.dropna(subset=['Consumer complaint narrative'])
    print(f"After removing missing narratives: {df_filtered.shape}")
    
    print(f"\nCategories included:")
    for product in target_products:
        count = len(df_filtered[df_filtered['Product'] == product])
        print(f"  - {product}: {count} complaints")
    
    return df_filtered


# ============================================================================
# STEP 1: EXPLANATORY DATA ANALYSIS (EDA)
# ============================================================================

def perform_eda(df):
    """
    Perform exploratory data analysis and visualize class distribution.
    """
    print("\n" + "=" * 80)
    print("STEP 1: EXPLANATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    # Count complaints by category
    class_counts = df['Product'].value_counts()
    print("\nComplaint Distribution by Product Category:")
    print(class_counts)
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    class_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Distribution of Consumer Complaints by Product Category', fontsize=14, fontweight='bold')
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Number of Complaints', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the chart
    import os
    output_path = 'eda_class_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ EDA chart saved as '{output_path}'")
    plt.close()
    
    return class_counts


# ============================================================================
# STEP 2: TEXT PRE-PROCESSING
# ============================================================================

def clean_text(text):
    """
    Clean and preprocess text data.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = text.split()
        text = ' '.join([word for word in tokens if word not in stop_words])
    except:
        # If NLTK data not available, skip stopword removal
        pass
    
    return text


def preprocess_data(df):
    """
    Preprocess the text data and split into train/test sets.
    """
    print("\n" + "=" * 80)
    print("STEP 2: TEXT PRE-PROCESSING")
    print("=" * 80)
    
    # Clean the text data
    print("Cleaning text data...")
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)
    
    # Prepare features and target
    X = df['cleaned_narrative']
    y = df['Product']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Vectorize the text using TF-IDF
    print("\nVectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape: {X_train_vec.shape}")
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


# ============================================================================
# STEP 3 & 4: MODEL SELECTION AND COMPARISON
# ============================================================================

def train_and_compare_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classification models and compare their performance.
    """
    print("\n" + "=" * 80)
    print("STEP 3 & 4: MODEL SELECTION AND COMPARISON")
    print("=" * 80)
    
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Linear Support Vector Classifier': LinearSVC(random_state=42, max_iter=2000)
    }
    
    results = {}
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
        trained_models[model_name] = model
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Find the best model
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]
    best_accuracy = results[best_model_name]
    
    print("\n" + "-" * 80)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("-" * 80)
    
    return best_model, best_model_name, trained_models


# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================

def evaluate_model(model, model_name, X_test, y_test):
    """
    Evaluate the best model with detailed metrics and confusion matrix.
    """
    print("\n" + "=" * 80)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 80)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:")
    print("-" * 80)
    print(classification_report(y_test, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('Actual Category', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the heatmap
    output_path = 'evaluation_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved as '{output_path}'")
    plt.close()


# ============================================================================
# STEP 6: PREDICTION
# ============================================================================

def predict_complaint_category(text, vectorizer, model):
    """
    Predict the category of a new complaint text.
    """
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Vectorize the text
    text_vec = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vec)[0]
    
    return prediction


def run_example_predictions(vectorizer, model):
    """
    Run example predictions on sample complaint texts.
    """
    print("\n" + "=" * 80)
    print("STEP 6: PREDICTION - EXAMPLE PREDICTIONS")
    print("=" * 80)
    
    # Example complaint texts
    examples = [
        "I found errors on my credit report that are not accurate and need to be corrected immediately.",
        "A debt collector has been calling me constantly about a debt I already paid off years ago.",
        "The interest rate on my personal loan is much higher than what was promised when I applied.",
        "My mortgage company failed to process my payment correctly and now they are threatening foreclosure."
    ]
    
    print("\nPredicting categories for example complaints:\n")
    for i, example in enumerate(examples, 1):
        prediction = predict_complaint_category(example, vectorizer, model)
        print(f"Example {i}:")
        print(f"  Text: \"{example[:80]}...\"")
        print(f"  Predicted Category: {prediction}")
        print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to execute the entire classification workflow.
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  TASK 5: CONSUMER COMPLAINT TEXT CLASSIFICATION".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")
    
    try:
        # Step 0: Load and filter data
        df = load_and_filter_data('complaints.csv')
        
        # Step 1: Perform EDA
        perform_eda(df)
        
        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)
        
        # Step 3 & 4: Train and compare models
        best_model, best_model_name, all_models = train_and_compare_models(
            X_train, X_test, y_train, y_test
        )
        
        # Step 5: Evaluate the best model
        evaluate_model(best_model, best_model_name, X_test, y_test)
        
        # Step 6: Run example predictions
        run_example_predictions(vectorizer, best_model)
        
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  ✓ eda_class_distribution.png")
        print("  ✓ evaluation_confusion_matrix.png")
        print("\n")
        
    except FileNotFoundError:
        print("\n" + "!" * 80)
        print("ERROR: complaints.csv file not found!")
        print("!" * 80)
        print("\nPlease ensure that 'complaints.csv' is placed in the 'task5' folder.")
        print("You can download it from:")
        print("https://catalog.data.gov/dataset/consumer-complaint-database")
        print("\n")
        
    except Exception as e:
        print("\n" + "!" * 80)
        print(f"ERROR: An unexpected error occurred!")
        print("!" * 80)
        print(f"\nError details: {str(e)}")
        print("\n")


if __name__ == "__main__":
    main()
