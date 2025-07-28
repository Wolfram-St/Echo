import pickle
import json
import random
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import os
from datetime import datetime
import logging
import tensorflow as tf
from collections import defaultdict

# Import our enhanced bot
from enhanced_training import EnhancedMentalWellnessBot

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the enhanced bot
bot = EnhancedMentalWellnessBot()

# Load trained model and components
try:
    bot.model = tf.keras.models.load_model('enhanced_model.h5')
    with open('enhanced_vectorizer.pkl', 'rb') as f:
        bot.vectorizer = pickle.load(f)
    with open('enhanced_classes.pkl', 'rb') as f:
        bot.classes = pickle.load(f)
    with open('enhanced_intents.pkl', 'rb') as f:
        bot.intents = pickle.load(f)
    print("✅ Enhanced model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please run enhanced_training.py first to train the model.")

# Voice processing components
class VoiceProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
        # Configure speech recognition
        self.recognizer.energy_threshold = 3000  # Lower threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Shorter pause threshold
        
        # Configure text-to-speech
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to set a female voice for better user experience
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        
        # Set speech rate and volume
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level
        
    def find_best_microphone(self):
        """Find the best available microphone"""
        try:
            # List all microphones
            mic_list = sr.Microphone.list_microphone_names()
            print(f"Available microphones: {len(mic_list)}")
            
            # Try to find a good microphone
            for i, name in enumerate(mic_list):
                if any(keyword in name.lower() for keyword in ['mic', 'microphone', 'input']):
                    print(f"Found microphone: {name} (index {i})")
                    return sr.Microphone(device_index=i)
            
            # Fallback to default
            print("Using default microphone")
            return sr.Microphone()
            
        except Exception as e:
            print(f"Error finding microphone: {e}")
            return sr.Microphone()
        
    def listen_and_transcribe(self):
        """Listen for voice input and return transcribed text"""
        try:
            # Find best microphone
            mic = self.find_best_microphone()
            
            with mic as source:
                print("🎤 Listening... Please speak now.")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio input with longer timeout
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)
                
                print("🎵 Processing audio...")
                
                # Try multiple recognition services
                try:
                    # First try Google Speech Recognition
                    text = self.recognizer.recognize_google(audio)
                    print(f"✅ Transcribed (Google): {text}")
                    return text.lower()
                except sr.RequestError:
                    print("🌐 Google Speech Recognition failed, trying alternative...")
                    # You could add other services here like Sphinx for offline recognition
                    return "ERROR"
                
        except sr.WaitTimeoutError:
            print("⏰ No speech detected within timeout")
            return "TIMEOUT"
        except sr.UnknownValueError:
            print("❓ Could not understand the audio")
            return "UNKNOWN"
        except sr.RequestError as e:
            print(f"🌐 Speech recognition service error: {e}")
            return "ERROR"
        except Exception as e:
            print(f"❌ Voice recognition error: {e}")
            return "ERROR"
            
    def speak(self, text):
        """Convert text to speech"""
        try:
            print(f"🔊 Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"❌ Text-to-speech error: {e}")
            return False

# Initialize voice processor
voice_processor = VoiceProcessor()

# Enhanced bot with emotional intelligence
class EmotionTracker:
    def __init__(self):
        self.user_mood_history = {}
        self.conversation_context = {}
        self.emotional_keywords = {
            'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
            'sad': ['sad', 'depressed', 'down', 'blue', 'hopeless', 'miserable', 'empty', 'heartbroken', 'worthless', 'lonely', 'isolated', 'lost', 'broken', 'defeated'],
            'anxious': ['anxious', 'worried', 'nervous', 'panicked', 'freaking out', 'overwhelmed', 'tense', 'on edge', 'restless', 'scared', 'afraid', 'terrified'],
            'frustrated': ['frustrated', 'angry', 'mad', 'annoyed', 'irritated', 'pissed off', 'fed up', 'furious', 'livid', 'rage', 'exhausted', 'tired'],
            'demotivated': ['demotivated', 'unmotivated', 'no drive', 'self doubt', 'stuck', 'lost', 'no purpose', 'no goals', 'giving up', 'failure', 'worthless'],
            'lonely': ['lonely', 'alone', 'isolated', 'no friends', 'disconnected', 'left out', 'don\'t belong', 'outsider', 'friendless'],
            'stressed': ['stressed', 'overwhelmed', 'pressure', 'swamped', 'busy', 'work stress', 'no time', 'behind', 'deadlines', 'overworked', 'exhausted', 'drowning'],
            'gratitude': ['grateful', 'thankful', 'appreciative', 'blessed', 'fortunate', 'lucky', 'content', 'happy', 'joyful', 'satisfied', 'fulfilled'],
        }
        
    def analyze_emotion(self, text):
        """Analyze the emotional content of text"""
        text_lower = text.lower()
        emotions = {}
        
        for emotion, keywords in self.emotional_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotions[emotion] = score
                
        return emotions
    
    def update_user_mood(self, user_id, text, intent):
        """Update user's mood based on conversation"""
        emotions = self.analyze_emotion(text)
        
        if user_id not in self.user_mood_history:
            self.user_mood_history[user_id] = []
            
        mood_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'intent': intent,
            'emotions': emotions,
            'overall_mood': self._calculate_overall_mood(emotions)
        }
        
        self.user_mood_history[user_id].append(mood_entry)
        
        # Keep only last 10 entries
        if len(self.user_mood_history[user_id]) > 10:
            self.user_mood_history[user_id] = self.user_mood_history[user_id][-10:]
    
    def _calculate_overall_mood(self, emotions):
        """Calculate overall mood score"""
        if not emotions:
            return 'neutral'
            
        # Simple scoring system
        positive_emotions = ['happy', 'excited', 'grateful']
        negative_emotions = ['sad', 'anxious', 'frustrated', 'demotivated', 'lonely', 'stressed']
        
        score = 0
        for emotion, intensity in emotions.items():
            if emotion in negative_emotions:
                score -= intensity
            elif emotion in positive_emotions:
                score += intensity
                
        if score > 0:
            return 'positive'
        elif score < 0:
            return 'negative'
        else:
            return 'neutral'
    
    def get_user_mood_trend(self, user_id):
        """Get user's mood trend over time"""
        if user_id not in self.user_mood_history:
            return 'neutral'
            
        recent_moods = [entry['overall_mood'] for entry in self.user_mood_history[user_id][-5:]]
        
        if not recent_moods:
            return 'neutral'
            
        # Simple trend analysis
        negative_count = recent_moods.count('negative')
        positive_count = recent_moods.count('positive')
        
        if negative_count > positive_count:
            return 'declining'
        elif positive_count > negative_count:
            return 'improving'
        else:
            return 'stable'
    
    def get_personalized_response(self, user_id, intent, base_response):
        mood_trend = self.get_user_mood_trend(user_id)
        last_support = user_support_history.get_last(user_id)
        support_types = ['joke', 'quote', 'story', 'coping']
        support_type = None
        support_text = ''

        # Only offer support if negative mood or intent
        negative_intents = ['feeling_sad', 'sad', 'feeling_frustrated', 'frustrated', 'feeling_demotivated', 'demotivated', 'feeling_lonely', 'lonely', 'feeling_stressed', 'stressed', 'self_doubt', 'work_stress']
        if intent in negative_intents or mood_trend == 'declining':
            # Pick a support type that is not the same as last time
            available_types = [t for t in support_types if t != last_support]
            if available_types:
                support_type = random.choice(available_types)
            else:
                support_type = random.choice(support_types)

            if support_type == 'joke':
                support_text = '\n\n😄 Here’s a little joke to lighten the mood:\n' + therapeutic_generator.get_joke()
            elif support_type == 'quote':
                support_text = '\n\n💡 Here’s a motivational quote for you:\n' + therapeutic_generator.get_quote()
            elif support_type == 'story':
                support_text = '\n\n📖 Here’s a short story that might inspire you:\n' + therapeutic_generator.get_story()
            elif support_type == 'coping':
                support_text = '\n\n🧘 Here’s a coping technique you can try:\n' + therapeutic_generator.get_coping_technique()

            user_support_history.update(user_id, support_type)

        # Add progress encouragement if mood is improving
        if mood_trend == 'improving':
            base_response += "\n\n✨ I can see you're making progress! Keep going - you're doing amazing work on your mental wellness journey."

        return base_response + support_text

class UserSupportHistory:
    def __init__(self):
        self.last_support_type = defaultdict(lambda: None)

    def update(self, user_id, support_type):
        self.last_support_type[user_id] = support_type

    def get_last(self, user_id):
        return self.last_support_type[user_id]

user_support_history = UserSupportHistory()

# Initialize emotion tracker
emotion_tracker = EmotionTracker()

# Conversation memory
conversation_history = {}

# Therapeutic response generator
class TherapeuticResponseGenerator:
    def __init__(self):
        self.jokes = [
            "Why don't scientists trust atoms? Because they make up everything! 😄",
            "What do you call a fake noodle? An impasta! 🍝",
            "Why did the scarecrow win an award? Because he was outstanding in his field! 🌾",
            "Why don't eggs tell jokes? They'd crack each other up! 🥚",
            "What do you call a bear with no teeth? A gummy bear! 🐻",
            "Why did the math book look so sad? Because it had too many problems! 📚",
            "Why don't skeletons fight each other? They don't have the guts! 💀",
            "What do you call a fish wearing a bowtie? So-fish-ticated! 🐠",
            "Why did the bicycle fall over? Because it was two-tired! 🚲",
            "What do you call a can opener that doesn't work? A can't opener! 😂"
        ]
        
        self.quotes = [
            "\"The only way to do great work is to love what you do.\" - Steve Jobs 💫",
            "\"Success is not final, failure is not fatal: it is the courage to continue that counts.\" - Winston Churchill 🦁",
            "\"The future belongs to those who believe in the beauty of their dreams.\" - Eleanor Roosevelt 🌙",
            "\"In the middle of difficulty lies opportunity.\" - Albert Einstein 🧠",
            "\"You are never too old to set another goal or to dream a new dream.\" - C.S. Lewis 🎯",
            "\"The only limit to our realization of tomorrow is our doubts of today.\" - Franklin D. Roosevelt 🌅",
            "\"Believe you can and you're halfway there.\" - Theodore Roosevelt 🚀",
            "\"It does not matter how slowly you go as long as you do not stop.\" - Confucius 🐢",
            "\"The journey of a thousand miles begins with one step.\" - Lao Tzu 👣",
            "\"What you get by achieving your goals is not as important as what you become by achieving your goals.\" - Zig Ziglar 🎯"
        ]
        
        self.stories = [
            "Here's a beautiful story: A young boy was throwing starfish back into the ocean. A man asked, 'Why bother? There are thousands of starfish.' The boy replied, 'It matters to this one.' 🌟 Every small act of kindness makes a difference!",
            "There's a story about a butterfly struggling to emerge from its cocoon. A person tried to help by cutting it open, but the butterfly couldn't fly. The struggle was necessary for its wings to develop strength. Your challenges are making you stronger! 🦋",
            "A wise man was asked, 'What's the secret to happiness?' He replied, 'I stopped trying to be happy and started trying to be helpful.' When we focus on others, we often find our own joy! 💫",
            "There's a tale about two wolves fighting inside us - one of anger and one of love. Which one wins? The one we feed. Choose to feed your positive thoughts! 🐺",
            "A student asked his teacher, 'How do I find my purpose?' The teacher took him to a river and said, 'Look at the water. It doesn't try to be a river, it just flows.' Sometimes we find our purpose by simply being ourselves! 🌊"
        ]
        
        self.coping_techniques = [
            "🧘 **4-7-8 Breathing**: Inhale for 4, hold for 7, exhale for 8. Repeat 3 times. This helps calm your nervous system.",
            "🎯 **5-4-3-2-1 Grounding**: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
            "💪 **Progressive Muscle Relaxation**: Tense and release each muscle group from toes to head, holding for 5 seconds each.",
            "🌱 **Mindful Walking**: Take a slow walk, focusing on each step and your surroundings.",
            "📝 **Thought Journaling**: Write down your thoughts and challenge negative ones with evidence.",
            "🎵 **Music Therapy**: Listen to calming music or create a playlist of songs that make you feel good.",
            "🌿 **Nature Connection**: Spend time outdoors, even just sitting in a park or garden.",
            "🤗 **Self-Compassion**: Talk to yourself like you would talk to a dear friend who's struggling."
        ]
    
    def get_joke(self):
        """Get a random joke"""
        return random.choice(self.jokes)
    
    def get_quote(self):
        """Get a random motivational quote"""
        return random.choice(self.quotes)
    
    def get_story(self):
        """Get a random inspiring story"""
        return random.choice(self.stories)
    
    def get_coping_technique(self):
        """Get a random coping technique"""
        return random.choice(self.coping_techniques)
    
    def get_emotional_support(self, emotion):
        """Get specific support based on emotion"""
        support_responses = {
            'sad': [
                "💙 I can see you're going through a difficult time. Remember, it's okay to not be okay. Every emotion is temporary, and you're stronger than you know.",
                "🌧️ Sadness can feel like a heavy cloud, but clouds don't last forever. The sun will shine again. Would you like to try a gentle breathing exercise together?",
                "🫂 Your feelings are valid, and you don't have to go through this alone. Sometimes just talking about it can help lighten the load."
            ],
            'anxious': [
                "🧘 Anxiety can feel like a storm inside your mind. Let's take a moment to ground ourselves. You're safe right now, even if it doesn't feel that way.",
                "🌸 Remember the 3-3-3 rule: Name 3 things you can see, 3 sounds you can hear, and move 3 parts of your body. This helps bring you back to the present.",
                "💫 Anxiety often makes us feel like we're alone, but you're not. I'm here to help you find some calm. What would help you feel more grounded?"
            ],
            'frustrated': [
                "🔥 Frustration can feel like hitting a wall repeatedly. Your feelings are valid, and it's okay to be upset. Sometimes we need to vent before we can find solutions.",
                "💪 I can hear how fed up you are, and that's completely understandable. What's been the biggest source of frustration? Let's work through this together.",
                "🌊 Frustration often comes from feeling stuck or unheard. I'm here to listen and help you work through this. What would help you feel heard?"
            ],
            'demotivated': [
                "🌱 It's completely normal to feel unmotivated sometimes - even the most successful people go through this. Your feelings are valid, and it's okay to need time to recharge.",
                "💫 Motivation can be like a flame that sometimes flickers. Sometimes when we're feeling demotivated, it's our mind's way of telling us we need a break or a change.",
                "🌟 You're not alone in feeling this way. Many people go through periods of low motivation, and it doesn't mean anything is wrong with you."
            ]
        }
        
        return random.choice(support_responses.get(emotion, ["I'm here to support you through this. What would help you feel better?"]))

# Initialize therapeutic response generator
therapeutic_generator = TherapeuticResponseGenerator()

@app.route('/')
def home():
    return render_template('enhanced_index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    """Get chatbot response via HTTP POST"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', 'default')
        
        if not user_message:
            return jsonify({'response': 'Please say something so I can help you.'})
        
        # Get bot response
        predictions = bot.predict_intent(user_message)
        response = bot.get_response(predictions)
        
        # Update emotion tracking
        intent = predictions[0]['intent'] if predictions else 'unknown'
        emotion_tracker.update_user_mood(user_id, user_message, intent)
        
        # Get personalized response
        response = emotion_tracker.get_personalized_response(user_id, intent, response)
        
        # Update emotion tracking
        intent = predictions[0]['intent'] if predictions else 'unknown'
        emotion_tracker.update_user_mood(user_id, user_message, intent)
        
        # Get personalized response
        response = emotion_tracker.get_personalized_response(user_id, intent, response)
        
        # Store in conversation history
        if user_id not in conversation_history:
            conversation_history[user_id] = []
        
        conversation_history[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'bot': response,
            'intent': predictions[0]['intent'] if predictions else 'unknown',
            'confidence': predictions[0]['probability'] if predictions else 0
        })
        
        # Keep only last 20 messages
        if len(conversation_history[user_id]) > 20:
            conversation_history[user_id] = conversation_history[user_id][-20:]
        
        return jsonify({
            'response': response,
            'intent': predictions[0]['intent'] if predictions else 'unknown',
            'confidence': predictions[0]['probability'] if predictions else 0
        })
        
    except Exception as e:
        logging.error(f"Error in get_bot_response: {e}")
        return jsonify({'response': 'I\'m having trouble processing that right now. Could you try again?'})

@app.route('/voice_input', methods=['POST'])
def voice_input():
    """Handle voice input"""
    try:
        # Start listening
        text = voice_processor.listen_and_transcribe()
        
        if text and text not in ["TIMEOUT", "UNKNOWN", "ERROR"]:
            # Get bot response
            predictions = bot.predict_intent(text)
            response = bot.get_response(predictions)
            
            # Speak response
            voice_processor.speak(response)
            
            return jsonify({
                'success': True,
                'transcribed_text': text,
                'response': response,
                'intent': predictions[0]['intent'] if predictions else 'unknown'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not understand audio. Please try again.'
            })
            
    except Exception as e:
        logging.error(f"Error in voice_input: {e}")
        return jsonify({
            'success': False,
            'error': 'Voice processing error. Please try again.'
        })

@app.route('/speak', methods=['POST'])
def speak_text():
    """Convert text to speech"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if text:
            voice_processor.speak(text)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'No text provided'})
            
    except Exception as e:
        logging.error(f"Error in speak_text: {e}")
        return jsonify({'success': False, 'error': 'Speech synthesis error'})

@app.route('/conversation_history/<user_id>')
def get_conversation_history(user_id):
    """Get conversation history for a user"""
    history = conversation_history.get(user_id, [])
    return jsonify({'history': history})

@app.route('/mood_tracker', methods=['POST'])
def track_mood():
    """Track user mood over time"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        mood_score = data.get('mood_score', 5)  # 1-10 scale
        mood_note = data.get('mood_note', '')
        
        # Store mood data (in a real app, this would go to a database)
        if 'mood_data' not in conversation_history.get(user_id, {}):
            conversation_history[user_id] = {'mood_data': []}
        
        conversation_history[user_id]['mood_data'].append({
            'timestamp': datetime.now().isoformat(),
            'mood_score': mood_score,
            'mood_note': mood_note
        })
        
        return jsonify({'success': True, 'message': 'Mood tracked successfully'})
        
    except Exception as e:
        logging.error(f"Error in track_mood: {e}")
        return jsonify({'success': False, 'error': 'Failed to track mood'})

@app.route('/therapeutic/joke', methods=['GET'])
def get_joke():
    """Get a random joke to cheer up the user"""
    try:
        joke = therapeutic_generator.get_joke()
        return jsonify({'success': True, 'joke': joke})
    except Exception as e:
        logging.error(f"Error getting joke: {e}")
        return jsonify({'success': False, 'error': 'Failed to get joke'})

@app.route('/therapeutic/quote', methods=['GET'])
def get_quote():
    """Get a motivational quote"""
    try:
        quote = therapeutic_generator.get_quote()
        return jsonify({'success': True, 'quote': quote})
    except Exception as e:
        logging.error(f"Error getting quote: {e}")
        return jsonify({'success': False, 'error': 'Failed to get quote'})

@app.route('/therapeutic/story', methods=['GET'])
def get_story():
    """Get an inspiring story"""
    try:
        story = therapeutic_generator.get_story()
        return jsonify({'success': True, 'story': story})
    except Exception as e:
        logging.error(f"Error getting story: {e}")
        return jsonify({'success': False, 'error': 'Failed to get story'})

@app.route('/therapeutic/coping', methods=['GET'])
def get_coping_technique():
    """Get a coping technique"""
    try:
        technique = therapeutic_generator.get_coping_technique()
        return jsonify({'success': True, 'technique': technique})
    except Exception as e:
        logging.error(f"Error getting coping technique: {e}")
        return jsonify({'success': False, 'error': 'Failed to get coping technique'})

@app.route('/therapeutic/support/<emotion>', methods=['GET'])
def get_emotional_support(emotion):
    """Get emotional support for specific emotion"""
    try:
        support = therapeutic_generator.get_emotional_support(emotion)
        return jsonify({'success': True, 'support': support})
    except Exception as e:
        logging.error(f"Error getting emotional support: {e}")
        return jsonify({'success': False, 'error': 'Failed to get emotional support'})

@app.route('/mood_analysis/<user_id>', methods=['GET'])
def get_mood_analysis(user_id):
    """Get mood analysis for a user"""
    try:
        mood_trend = emotion_tracker.get_user_mood_trend(user_id)
        mood_history = emotion_tracker.user_mood_history.get(user_id, [])
        
        return jsonify({
            'success': True,
            'mood_trend': mood_trend,
            'mood_history': mood_history[-5:],  # Last 5 entries
            'total_entries': len(mood_history)
        })
    except Exception as e:
        logging.error(f"Error getting mood analysis: {e}")
        return jsonify({'success': False, 'error': 'Failed to get mood analysis'})

# WebSocket events for real-time communication
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to Luna - Your AI Wellness Companion'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(data):
    """Handle real-time messages"""
    try:
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', 'default')
        
        if not user_message:
            emit('response', {'response': 'Please say something so I can help you.'})
            return
        
        # Get bot response
        predictions = bot.predict_intent(user_message)
        response = bot.get_response(predictions)
        
        # Store in conversation history
        if user_id not in conversation_history:
            conversation_history[user_id] = []
        
        conversation_history[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'bot': response,
            'intent': predictions[0]['intent'] if predictions else 'unknown',
            'confidence': predictions[0]['probability'] if predictions else 0
        })
        
        # Emit response
        emit('response', {
            'response': response,
            'intent': predictions[0]['intent'] if predictions else 'unknown',
            'confidence': predictions[0]['probability'] if predictions else 0
        })
        
    except Exception as e:
        logging.error(f"Error in handle_message: {e}")
        emit('response', {'response': 'I\'m having trouble processing that right now. Could you try again?'})

@socketio.on('voice_start')
def handle_voice_start():
    """Start voice recognition"""
    try:
        # voice_processor.start_listening() # This line is removed as per the new VoiceProcessor class
        emit('voice_status', {'status': 'listening'})
    except Exception as e:
        emit('voice_status', {'status': 'error', 'message': str(e)})

@socketio.on('voice_stop')
def handle_voice_stop():
    """Stop voice recognition and process audio"""
    try:
        # voice_processor.stop_listening() # This line is removed as per the new VoiceProcessor class
        
        # Process audio
        text = voice_processor.listen_and_transcribe() # Changed to use the new method
        
        if text and text not in ["TIMEOUT", "UNKNOWN", "ERROR"]:
            # Get bot response
            predictions = bot.predict_intent(text)
            response = bot.get_response(predictions)
            
            emit('voice_result', {
                'success': True,
                'transcribed_text': text,
                'response': response,
                'intent': predictions[0]['intent'] if predictions else 'unknown'
            })
            
            # Speak response
            voice_processor.speak(response)
        else:
            emit('voice_result', {
                'success': False,
                'error': 'Could not understand audio. Please try again.'
            })
            
    except Exception as e:
        emit('voice_result', {
            'success': False,
            'error': f'Voice processing error: {str(e)}'
        })

if __name__ == '__main__':
    print("🤖 Enhanced Mental Wellness Chatbot Starting...")
    print("🌐 Web Interface: http://localhost:5000")
    print("🎤 Voice features enabled")
    print("💬 Real-time chat enabled")
    
    # Run the app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 