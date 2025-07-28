import nltk
import json
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import re

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    print("NLTK data already downloaded or download failed")

class EnhancedMentalWellnessBot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!', '.', ',', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}']
        self.model = None
        self.vectorizer = None
        self.intents = None
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing with better cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatize and remove stop words
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and token not in self.ignore_words:
                # Lemmatize with POS tagging for better accuracy
                lemmatized = self.lemmatizer.lemmatize(token)
                if lemmatized:
                    processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def load_data(self, intents_file):
        """Load and preprocess intents data"""
        print("Loading intents data...")
        with open(intents_file, 'r', encoding='utf-8') as file:
            self.intents = json.load(file)
        
        # Process patterns and create documents
        for intent in self.intents['intents']:
            tag = intent['tag']
            if tag not in self.classes:
                self.classes.append(tag)
            
            for pattern in intent['patterns']:
                # Preprocess the pattern
                processed_pattern = self.preprocess_text(pattern)
                
                # Add to documents
                self.documents.append((processed_pattern, tag))
                
                # Add words to vocabulary
                words_in_pattern = processed_pattern.split()
                self.words.extend(words_in_pattern)
        
        # Remove duplicates and sort
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        print(f"Loaded {len(self.documents)} documents")
        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Vocabulary size: {len(self.words)} words")
        
    def create_tfidf_features(self):
        """Create TF-IDF features for better text representation"""
        print("Creating TF-IDF features...")
        
        # Prepare data for TF-IDF
        patterns = [doc[0] for doc in self.documents]
        labels = [doc[1] for doc in self.documents]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform
        X = self.vectorizer.fit_transform(patterns)
        
        # Create label encoder
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, label_encoder
    
    def create_enhanced_model(self, input_dim, num_classes):
        """Create an enhanced neural network model"""
        print("Creating enhanced model...")
        
        model = Sequential([
            # Input layer
            Dense(256, input_dim=input_dim, activation='relu'),
            Dropout(0.3),
            
            # Hidden layers
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile with better optimizer and learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, X_test, y_train, y_test, label_encoder):
        """Train the enhanced model with callbacks"""
        print("Training enhanced model...")
        
        # Create model
        self.model = self.create_enhanced_model(X_train.shape[1], len(self.classes))
        
        # Callbacks for better training
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Convert sparse matrices to dense for TensorFlow compatibility
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        
        # Train model
        history = self.model.fit(
            X_train_dense, y_train,
            validation_data=(X_test_dense, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test_dense)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=label_encoder.classes_))
        
        return history
    
    def save_model(self):
        """Save the trained model and preprocessing components"""
        print("Saving model and components...")
        
        # Save model
        self.model.save('enhanced_model.h5')
        
        # Save vectorizer
        with open('enhanced_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save classes
        with open('enhanced_classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)
        
        # Save intents
        with open('enhanced_intents.pkl', 'wb') as f:
            pickle.dump(self.intents, f)
        
        print("Model and components saved successfully!")
    
    def predict_intent(self, text, threshold=0.3):
        """Predict intent with confidence threshold"""
        # Preprocess input text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text])
        
        # Convert to dense for prediction
        X_dense = X.toarray()
        
        # Predict
        predictions = self.model.predict(X_dense)[0]
        
        # Get top predictions above threshold
        results = []
        for i, prob in enumerate(predictions):
            if prob > threshold:
                results.append({
                    'intent': self.classes[i],
                    'probability': float(prob)
                })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return results
    
    def get_response(self, intent_results):
        """Get response based on predicted intent"""
        if not intent_results:
            return "I'm not sure I understand. Could you rephrase that or tell me more about what's on your mind?"
        
        intent = intent_results[0]['intent']
        confidence = intent_results[0]['probability']
        
        # Find matching intent
        for intent_data in self.intents['intents']:
            if intent_data['tag'] == intent:
                responses = intent_data['responses']
                return random.choice(responses)
        
        return "I'm here to listen and support you. Could you tell me more about how you're feeling?"

def main():
    """Main training function"""
    print("🤖 Enhanced Mental Wellness Chatbot Training")
    print("=" * 50)
    
    # Initialize bot
    bot = EnhancedMentalWellnessBot()
    
    # Load data
    bot.load_data('enhanced_intents.json')
    
    # Create features
    X_train, X_test, y_train, y_test, label_encoder = bot.create_tfidf_features()
    
    # Train model
    history = bot.train_model(X_train, X_test, y_train, y_test, label_encoder)
    
    # Save model
    bot.save_model()
    
    # Test the model
    print("\n🧪 Testing the model...")
    test_phrases = [
        "I'm feeling really sad today",
        "I'm so anxious about my presentation",
        "Thank you for helping me",
        "I'm stressed about work",
        "I feel like I'm not good enough"
    ]
    
    for phrase in test_phrases:
        predictions = bot.predict_intent(phrase)
        response = bot.get_response(predictions)
        print(f"\nInput: {phrase}")
        print(f"Predicted Intent: {predictions[0]['intent'] if predictions else 'None'}")
        confidence = predictions[0]['probability'] if predictions else 0.0
        print(f"Confidence: {confidence:.3f}")
        print(f"Response: {response}")
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main() 