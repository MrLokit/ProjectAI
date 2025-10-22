import speech_recognition as sr
import pyttsx3
import datetime
import numpy as np
import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle


class NeuralJarvis:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()

        self.intents = {
            "greeting": {
                "patterns": ["привет", "здравствуй", "добрый день", "хай", "hello", "джарвис"],
                "responses": ["Здравствуйте!", "Привет! Как дела?", "Рад вас слышать!"]
            },
            "time": {
                "patterns": ["который час", "сколько время", "время", "time", "current time"],
                "responses": ["Сейчас {}"]
            },
            "date": {
                "patterns": ["какое число", "какая дата", "дата", "date", "today"],
                "responses": ["Сегодня {}"]
            },
            "calculation": {
                "patterns": ["посчитай", "вычисли", "сколько будет", "calculate", "calc"],
                "responses": ["Результат: {}"]
            },
            "thanks": {
                "patterns": ["спасибо", "благодарю", "молодец", "thanks", "thank you"],
                "responses": ["Всегда рад помочь!", "Пожалуйста!", "Обращайтесь!"]
            },
            "goodbye": {
                "patterns": ["пока", "до свидания", "выход", "bye", "goodbye"],
                "responses": ["До свидания!", "Удачи!", "Был рад помочь!"]
            }
        }

        self.vectorizer = None
        self.label_encoder = None
        self.model = None
        self.train_neural_network()

        print("Калибровка микрофона...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        print("Калибровка завершена!")

    def prepare_training_data(self):
        """Подготовка данных для обучения нейросети"""
        texts = []
        labels = []

        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data["patterns"]:
                texts.append(pattern)
                labels.append(intent_name)

        return texts, labels

    def create_neural_network(self, input_dim, output_dim):
        """Создание архитектуры нейросети"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(output_dim, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_neural_network(self):
        """Обучение нейросети для классификации намерений"""
        print("Обучение нейросети...")

        texts, labels = self.prepare_training_data()

        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3), max_features=100)
        X = self.vectorizer.fit_transform(texts).toarray()

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        self.model = self.create_neural_network(X.shape[1], len(self.label_encoder.classes_))

        self.model.fit(
            X, y,
            epochs=200,
            batch_size=8,
            verbose=0,
            validation_split=0.1
        )

        print("Нейросеть обучена!")

    def predict_intent(self, text):
        """Предсказание намерения с помощью нейросети"""
        if not text:
            return None

        text_vector = self.vectorizer.transform([text]).toarray()

        prediction = self.model.predict(text_vector, verbose=0)
        intent_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > 0.6:
            return self.label_encoder.inverse_transform([intent_index])[0]
        else:
            return "unknown"

    def speak(self, text):
        """Произносит текст"""
        print(f"Джарвис: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen(self):
        """Слушает и распознает речь"""
        try:
            with self.microphone as source:
                print("Слушаю...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)

            command = self.recognizer.recognize_google(audio, language="ru-RU")
            print(f"Вы сказали: {command}")
            return command.lower()

        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("Не удалось распознать речь")
            return ""
        except sr.RequestError as e:
            print(f"Ошибка сервиса распознавания; {e}")
            return ""

    def get_time(self):
        """Возвращает текущее время"""
        now = datetime.datetime.now()
        return f"Сейчас {now.hour} часов {now.minute} минут"

    def get_date(self):
        """Возвращает текущую дату"""
        now = datetime.datetime.now()
        months = ["января", "февраля", "марта", "апреля", "мая", "июня",
                  "июля", "августа", "сентября", "октября", "ноября", "декабря"]
        return f"Сегодня {now.day} {months[now.month - 1]} {now.year} года"

    def calculate_expression(self, text):
        """Вычисляет математическое выражение из текста"""
        try:
            text = text.replace('плюс', '+').replace('минус', '-').replace('умножить на', '*').replace('делить на', '/')

            if '+' in text:
                parts = text.split('+')
                if len(parts) == 2:
                    a = float(''.join(filter(str.isdigit, parts[0])))
                    b = float(''.join(filter(str.isdigit, parts[1])))
                    return a + b

            if 'сколько будет' in text:
                expr = text.replace('сколько будет', '').strip()
                if '+' in expr:
                    nums = [float(x) for x in expr.split('+')]
                    return sum(nums)

            return "Не могу вычислить это выражение"

        except:
            return "Ошибка в вычислениях"

    def process_command(self, command):
        """Обработка команды с использованием нейросети"""
        if not command:
            return True

        intent = self.predict_intent(command)
        print(f"Нейросеть определила намерение: {intent}")

        if intent == "greeting":
            response = random.choice(self.intents["greeting"]["responses"])
            self.speak(response)

        elif intent == "time":
            time_str = self.get_time()
            response = random.choice(self.intents["time"]["responses"]).format(time_str)
            self.speak(response)

        elif intent == "date":
            date_str = self.get_date()
            response = random.choice(self.intents["date"]["responses"]).format(date_str)
            self.speak(response)

        elif intent == "calculation":
            result = self.calculate_expression(command)
            response = random.choice(self.intents["calculation"]["responses"]).format(result)
            self.speak(response)

        elif intent == "thanks":
            response = random.choice(self.intents["thanks"]["responses"])
            self.speak(response)

        elif intent == "goodbye":
            response = random.choice(self.intents["goodbye"]["responses"])
            self.speak(response)
            return False

        else:
            self.speak("Извините, я не понял команду. Попробуйте сказать 'время', 'дата' или 'посчитай'.")

        return True

    def run(self):
        """Основной цикл работы ассистента"""
        self.speak("Нейросетевой Джарвис активирован! Готов к работе.")

        running = True
        while running:
            command = self.listen()
            running = self.process_command(command)

def test_neural_network():
    """Функция для тестирования работы нейросети"""
    jarvis = NeuralJarvis()

    test_phrases = [
        "привет",
        "который час",
        "какая дата",
        "посчитай 5 плюс 3",
        "спасибо",
        "пока",
        "что ты умеешь"
    ]

    print("\nТестирование нейросети:")
    for phrase in test_phrases:
        intent = jarvis.predict_intent(phrase)
        print(f"'{phrase}' -> {intent}")


if __name__ == "__main__":
    jarvis = NeuralJarvis()
    jarvis.run()