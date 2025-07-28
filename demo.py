#!/usr/bin/env python3
"""
Demo script for Enhanced Mental Wellness Chatbot
Showcases the chatbot's capabilities and tests various features
"""

import pickle
import json
import random
import time
from enhanced_training import EnhancedMentalWellnessBot

def print_demo_banner():
    """Print demo banner"""
    print("=" * 70)
    print("🌙 Luna - Enhanced AI Mental Wellness Chatbot Demo")
    print("=" * 70)
    print("This demo showcases Luna's enhanced capabilities:")
    print("• Advanced NLP with TF-IDF vectorization")
    print("• Empathetic and contextual responses")
    print("• Better intent recognition")
    print("• Confidence scoring")
    print("• English language optimization")
    print("=" * 70)
    print()

def load_bot():
    """Load the trained bot"""
    print("🤖 Loading Luna...")
    try:
        bot = EnhancedMentalWellnessBot()
        bot.model = tf.keras.models.load_model('enhanced_model.h5')
        with open('enhanced_vectorizer.pkl', 'rb') as f:
            bot.vectorizer = pickle.load(f)
        with open('enhanced_classes.pkl', 'rb') as f:
            bot.classes = pickle.load(f)
        with open('enhanced_intents.pkl', 'rb') as f:
            bot.intents = pickle.load(f)
        print("✅ Luna loaded successfully!")
        return bot
    except Exception as e:
        print(f"❌ Error loading Luna: {e}")
        print("💡 Make sure you've run 'python enhanced_training.py' first")
        return None

def demo_conversation(bot):
    """Demo conversation flow"""
    print("\n💬 Starting conversation demo...")
    print("-" * 50)
    
    # Sample conversation flows
    conversations = [
        [
            "Hi there!",
            "I'm feeling really sad today",
            "I just feel like nothing is going right",
            "I don't know what to do anymore",
            "Thank you for listening to me"
        ],
        [
            "Hello Luna",
            "I'm so anxious about my presentation tomorrow",
            "I can't stop worrying about it",
            "What should I do?",
            "That's really helpful, thank you"
        ],
        [
            "Hey",
            "I'm feeling really stressed at work",
            "I feel like I'm drowning in responsibilities",
            "I'm not sure how to handle it all",
            "You're right, I need to take a break"
        ]
    ]
    
    for i, conversation in enumerate(conversations, 1):
        print(f"\n🎭 Conversation {i}:")
        print("-" * 30)
        
        for user_input in conversation:
            print(f"\n👤 You: {user_input}")
            
            # Get bot response
            predictions = bot.predict_intent(user_input)
            response = bot.get_response(predictions)
            
            print(f"🤖 Luna: {response}")
            
            if predictions:
                intent = predictions[0]['intent']
                confidence = predictions[0]['probability']
                print(f"   📊 Intent: {intent} (Confidence: {confidence:.3f})")
            
            time.sleep(1)  # Pause for readability

def demo_intent_recognition(bot):
    """Demo intent recognition capabilities"""
    print("\n🧠 Testing Intent Recognition...")
    print("-" * 50)
    
    test_phrases = [
        ("I'm feeling down", "feeling_sad"),
        ("I'm so happy today!", "greeting"),
        ("I'm really anxious about everything", "feeling_anxious"),
        ("Work is stressing me out", "feeling_stressed"),
        ("I feel like I'm not good enough", "self_doubt"),
        ("I'm so lonely", "feeling_lonely"),
        ("Thank you for helping me", "gratitude"),
        ("I need some advice", "coping_help"),
        ("I'm having relationship problems", "relationship_issues"),
        ("I hate my job", "work_stress")
    ]
    
    correct = 0
    total = len(test_phrases)
    
    for phrase, expected_intent in test_phrases:
        predictions = bot.predict_intent(phrase)
        
        if predictions:
            predicted_intent = predictions[0]['intent']
            confidence = predictions[0]['probability']
            
            status = "✅" if predicted_intent == expected_intent else "❌"
            print(f"{status} '{phrase}' → {predicted_intent} ({confidence:.3f})")
            
            if predicted_intent == expected_intent:
                correct += 1
        else:
            print(f"❌ '{phrase}' → No intent detected")
    
    accuracy = (correct / total) * 100
    print(f"\n📊 Intent Recognition Accuracy: {accuracy:.1f}% ({correct}/{total})")

def demo_response_variety(bot):
    """Demo response variety for the same intent"""
    print("\n🎭 Testing Response Variety...")
    print("-" * 50)
    
    test_phrase = "I'm feeling really sad"
    print(f"👤 Input: '{test_phrase}'")
    print("\n🤖 Multiple responses:")
    
    responses = set()
    for i in range(10):
        predictions = bot.predict_intent(test_phrase)
        response = bot.get_response(predictions)
        responses.add(response)
        print(f"   {i+1}. {response}")
    
    print(f"\n📊 Unique responses: {len(responses)}/10")
    print(f"🎯 Variety score: {(len(responses)/10)*100:.1f}%")

def demo_confidence_thresholds(bot):
    """Demo confidence threshold effects"""
    print("\n🎯 Testing Confidence Thresholds...")
    print("-" * 50)
    
    test_phrases = [
        "I'm feeling sad",
        "The weather is nice today",
        "I need help with my mental health",
        "What's 2+2?",
        "I'm having a great day!"
    ]
    
    thresholds = [0.1, 0.3, 0.5, 0.7]
    
    for phrase in test_phrases:
        print(f"\n👤 '{phrase}':")
        
        for threshold in thresholds:
            predictions = bot.predict_intent(phrase, threshold=threshold)
            
            if predictions:
                intent = predictions[0]['intent']
                confidence = predictions[0]['probability']
                print(f"   Threshold {threshold}: {intent} ({confidence:.3f})")
            else:
                print(f"   Threshold {threshold}: No intent detected")

def demo_voice_features():
    """Demo voice feature descriptions"""
    print("\n🎤 Voice Features Demo...")
    print("-" * 50)
    
    voice_features = [
        "🎙️ Speech-to-Text: Real-time voice input processing",
        "🔊 Text-to-Speech: Natural voice responses",
        "🎭 Voice Cloning: Customizable voice settings",
        "⚡ Background Processing: Non-blocking operations",
        "🎵 Multiple Voices: Choose from available system voices",
        "🔧 Adjustable Settings: Speed, volume, and pitch control"
    ]
    
    for feature in voice_features:
        print(f"   {feature}")
    
    print("\n💡 To test voice features, run the web application:")
    print("   python enhanced_app.py")
    print("   Then click the microphone button in the interface")

def demo_mood_tracking():
    """Demo mood tracking features"""
    print("\n💭 Mood Tracking Features...")
    print("-" * 50)
    
    mood_features = [
        "📊 Daily Mood Logging: 1-10 scale tracking",
        "📝 Mood Notes: Optional text descriptions",
        "📈 Progress Visualization: Track patterns over time",
        "🧠 AI Insights: Personalized mood analysis",
        "📅 Historical Data: View past mood entries",
        "🎯 Trend Analysis: Identify emotional patterns"
    ]
    
    for feature in mood_features:
        print(f"   {feature}")
    
    print("\n💡 To use mood tracking, run the web application and")
    print("   click the heart icon in the interface")

def interactive_demo(bot):
    """Interactive demo where user can chat with the bot"""
    print("\n🎮 Interactive Demo Mode")
    print("-" * 50)
    print("Chat with Luna! Type 'quit' to exit, 'help' for commands")
    print()
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("🤖 Luna: Goodbye! Take care of yourself! 💙")
                break
            elif user_input.lower() == 'help':
                print("🤖 Available commands:")
                print("   - quit/exit/bye: Exit the demo")
                print("   - help: Show this help message")
                print("   - info: Show intent and confidence info")
                print("   - Any other text: Chat with Luna")
                continue
            elif user_input.lower() == 'info':
                print("🤖 Info mode enabled - will show intent details")
                continue
            elif not user_input:
                continue
            
            # Get bot response
            predictions = bot.predict_intent(user_input)
            response = bot.get_response(predictions)
            
            print(f"🤖 Luna: {response}")
            
            if predictions:
                intent = predictions[0]['intent']
                confidence = predictions[0]['probability']
                print(f"   📊 Intent: {intent} (Confidence: {confidence:.3f})")
            else:
                print("   📊 No intent detected")
                
        except KeyboardInterrupt:
            print("\n🤖 Luna: Goodbye! Take care! 💙")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demo function"""
    print_demo_banner()
    
    # Load the bot
    bot = load_bot()
    if not bot:
        return
    
    # Run demos
    demo_conversation(bot)
    demo_intent_recognition(bot)
    demo_response_variety(bot)
    demo_confidence_thresholds(bot)
    demo_voice_features()
    demo_mood_tracking()
    
    # Interactive demo
    print("\n" + "=" * 70)
    print("🎮 Ready for interactive demo!")
    print("=" * 70)
    
    try:
        interactive_demo(bot)
    except KeyboardInterrupt:
        print("\n🤖 Demo ended. Thank you for trying Luna! 💙")
    
    print("\n" + "=" * 70)
    print("🌙 Luna Demo Complete!")
    print("=" * 70)
    print("To run the full web application:")
    print("   python enhanced_app.py")
    print("Then open http://localhost:5000 in your browser")
    print("=" * 70)

if __name__ == "__main__":
    main() 