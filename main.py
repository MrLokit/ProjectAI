import speech_recognition as sr
import pyttsx3
import datetime
import numpy as np
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pickle
import re
import os
import time


class Matt:
    def __init__(self, load_existing_model=True):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()

        self.recognizer.pause_threshold = 1.5

        self.activation_words = ["мэтт", "мет", "мэт", "мат", "метт"]
        self.is_activated = False

        self.intents = self.create_dataset()

        self.vectorizer = None
        self.label_encoder = None
        self.model = None

        if load_existing_model and os.path.exists('best_matt_model.h5') and os.path.exists('matt_model.pkl'):
            print("Загрузка обученной модели...")
            self.load_model()
        else:
            print("Обучение модели...")
            self.train_model()

        print("Калибровка микрофона...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=3)
        print("Калибровка завершена!\n")

    def create_dataset(self):
        """Создание датасета"""
        return {
            "greeting": {
                "patterns": [
                    "привет", "здравствуй", "добрый день", "хай", "hello",
                    "доброе утро", "добрый вечер", "приветствую", "салют",
                    "привет друг", "привет ассистент", "здравствуйте", "приветик",
                    "доброй ночи", "привет всем", "здорово", "мое почтение",
                    "доброго времени суток", "приветствую вас", "рад тебя видеть",
                    "привет как дела", "привет старший", "здравствуй мэтт",
                    "привет умник", "добрый", "приветствие", "здарова", "хай там"
                ],
                "responses": [
                    "Здравствуйте!", "Привет! Как дела?", "Рад вас слышать!", "Приветствую!",
                    "Здорово!", "Привет! Чем могу помочь?", "Добрый день!", "Привет, друг!"
                ]
            },
            "time": {
                "patterns": [
                    "который час", "сколько время", "время", "time", "current time",
                    "скажи время", "подскажи время", "сколько времени", "который сейчас час",
                    "время сейчас", "текущее время", "сколько сейчас времени"
                ],
                "responses": ["Сейчас {}", "Текущее время: {}", "На часах {}", "Время: {}"]
            },
            "date": {
                "patterns": [
                    "какое число", "какая дата", "дата", "date", "today",
                    "какой сегодня день", "подскажи дату", "какое сегодня число",
                    "число сегодня", "текущая дата", "какой сегодня день месяца"
                ],
                "responses": ["Сегодня {}", "Текущая дата: {}", "Сегодняшнее число: {}", "Дата: {}"]
            },
            "calculation": {
                "patterns": [
                    "посчитай", "вычисли", "сколько будет", "calculate", "calc",
                    "сложи", "прибавь", "отними", "умножь", "раздели",
                    "сложи числа", "вычисли пример", "реши пример", "посчитай пример",
                    "математика", "арифметика", "реши задачу", "вычисли результат",
                    "посчитай сумму", "сложи цифры", "прибавь числа", "отними числа",
                    "умножь числа", "раздели числа", "математическая операция",
                    "сколько получится", "какой результат", "вычисли значение",
                    "реши математический пример", "вычисли арифметическое выражение",
                    "посчитай выражение", "вычисли сумму", "найди разность", "умножь цифры",
                    "раздели цифры", "вычисли произведение", "найди частное",
                    "обучи математике", "научи решать примеры", "объясни вычисления",
                    "покажи как считать", "обучение математике", "научи математике"
                ],
                "responses": ["Результат: {}", "Ответ: {}", "Получается: {}", "Вычисляю: {}"]
            },
            "math_learning": {
                "patterns": [
                    "обучи математике", "научи решать примеры", "объясни вычисления",
                    "покажи как считать", "обучение математике", "научи математике",
                    "объясни математику", "научи считать", "как решать примеры",
                    "обучение вычислениям", "математическое обучение", "научи арифметике",
                    "объясни сложение", "объясни вычитание", "объясни умножение",
                    "объясни деление", "как складывать", "как вычитать", "как умножать",
                    "как делить", "урок математики", "математический урок"
                ],
                "responses": [
                    "С удовольствием научу вас математике! Например: '5 плюс 3' будет 8.",
                    "Давайте решать примеры вместе! Скажите 'посчитай 10 минус 4'.",
                    "Математика - это просто! Попробуйте: '6 умножить на 7' равно 42.",
                    "Я помогу с вычислениями. Просто скажите 'сколько будет 15 разделить на 3'."
                ]
            },
            "thanks": {
                "patterns": [
                    "спасибо", "благодарю", "молодец", "thanks", "thank you",
                    "ты лучший", "отлично", "хорошая работа", "отлично работает"
                ],
                "responses": [
                    "Всегда рад помочь!", "Пожалуйста!", "Обращайтесь!", "Рад был помочь!"
                ]
            },
            "mood": {
                "patterns": [
                    "как дела", "как настроение", "как себя чувствуешь", "как жизнь",
                    "как твои дела", "что нового", "как ты", "how are you",
                    "расскажи о настроении", "какое у тебя настроение"
                ],
                "responses": [
                    "Всё отлично, спасибо!",
                    "Прекрасно! Готов помогать.",
                    "Как у настоящего ИИ - без эмоций, но эффективно!",
                    "Замечательно! А у вас?"
                ]
            },
            "goodbye": {
                "patterns": [
                    "пока", "до свидания", "выход", "bye", "goodbye",
                    "закончим", "до встречи", "прощай", "всего хорошего"
                ],
                "responses": [
                    "До свидания!", "Удачи!", "Был рад помочь!", "До новых встреч!"
                ]
            }
        }

    def prepare_training_data(self):
        """Подготовка данных для обучения"""
        texts = []
        labels = []

        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data["patterns"]:
                texts.append(self.preprocess_text(pattern))
                labels.append(intent_name)

        return texts, labels

    def preprocess_text(self, text):
        """Предобработка текста"""
        text = text.lower()
        text = re.sub(r'[^\w\s!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_model(self, input_dim, output_dim):
        """Создание архитектуры нейросети"""
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,),
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),

            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),

            Dense(64, activation='relu'),
            Dropout(0.2),

            Dense(output_dim, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, epochs=1000):
        """Обучение нейросети"""
        print("Запуск обучения нейросети...")

        texts, labels = self.prepare_training_data()

        print(f"Всего примеров для обучения: {len(texts)}")
        print(f"Количество классов: {len(set(labels))}")

        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=1000,
            min_df=1,
            max_df=0.9
        )

        X = self.vectorizer.fit_transform(texts).toarray()
        print(f"Размерность данных после векторизации: {X.shape}")

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.15,
            random_state=42,
            stratify=y
        )

        print(f"Обучающая выборка: {X_train.shape[0]} примеров")
        print(f"Валидационная выборка: {X_val.shape[0]} примеров\n")

        self.model = self.create_model(X_train.shape[1], len(self.label_encoder.classes_))

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=0.0001,
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            'best_matt_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        print("Начинаем обучение...\n")

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            shuffle=True
        )

        if os.path.exists('best_matt_model.h5'):
            self.model.load_weights('best_matt_model.h5')
            print("Загружена лучшая модель!")

        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        print(f"\nОбучение завершено!")
        print(f"Точность на обучении: {final_train_acc:.4f}")
        print(f"Точность на валидации: {final_val_acc:.4f}\n")

        self.save_model()

    def save_model(self):
        """Сохранение обученной модели"""
        model_data = {
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'intents': self.intents
        }

        with open('matt_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        print("Модель сохранена\n")

    def load_model(self):
        """Загрузка обученной модели"""
        try:
            with open('matt_model.pkl', 'rb') as f:
                model_data = pickle.load(f)

            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.intents = model_data['intents']

            texts, _ = self.prepare_training_data()
            X_sample = self.vectorizer.transform(texts[:1]).toarray()

            self.model = self.create_model(X_sample.shape[1], len(self.label_encoder.classes_))
            self.model.load_weights('best_matt_model.h5')

            print("Модель загружена успешно!\n")
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def predict_intent(self, text):
        """Предсказание намерения"""
        if not text:
            return None, 0.0

        processed_text = self.preprocess_text(text)

        try:
            text_vector = self.vectorizer.transform([processed_text]).toarray()
            prediction = self.model.predict(text_vector, verbose=0)
            intent_index = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.6:
                return self.label_encoder.inverse_transform([intent_index])[0], confidence
            else:
                return "unknown", confidence

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return "unknown", 0.0

    def speak(self, text):
        """Произносит текст (теперь без звука)"""
        print(f"Мэтт: {text}\n")

    def check_activation_word(self, text):
        """Проверяет наличие ключевого слова в тексте"""
        text_lower = text.lower()
        for word in self.activation_words:
            if word in text_lower:
                return True, word
        return False, None

    def listen_for_activation(self):
        """Слушает ключевое слово для активации"""
        try:
            with self.microphone as source:
                print("Ожидание ключевого слова 'Мэтт'...")
                time.sleep(0.5)
                audio = self.recognizer.listen(source, timeout=20, phrase_time_limit=10)

            text = self.recognizer.recognize_google(audio, language="ru-RU")

            has_activation, activation_word = self.check_activation_word(text)

            if has_activation:
                print(f"Ключевое слово '{activation_word}' распознано")
                for word in self.activation_words:
                    if word in text.lower():
                        command = text.lower().split(word, 1)[1].strip()
                        if command:
                            print(f"Команда: {command}")
                            return command
                        else:
                            return True
            return None

        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Ошибка сервиса распознавания: {e}")
            return None

    def listen_for_command(self):
        """Слушает команду после активации"""
        try:
            self.speak("Слушаю")
            with self.microphone as source:
                time.sleep(0.5)
                audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=10)

            command = self.recognizer.recognize_google(audio, language="ru-RU")
            print(f"Команда: {command}")
            return command.lower()

        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("Не удалось распознать речь")
            return ""
        except sr.RequestError as e:
            print(f"Ошибка сервиса распознавания: {e}")
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

    def safe_calculate(self, expression):
        """Безопасное вычисление математического выражения"""
        try:
            safe_chars = set('0123456789+-*/.() ')
            cleaned_expression = ''.join(char for char in expression if char in safe_chars)

            if not cleaned_expression:
                return None, "Пустое выражение"

            result = eval(cleaned_expression)
            return result, None

        except ZeroDivisionError:
            return None, "Деление на ноль невозможно"
        except Exception as e:
            return None, f"Ошибка вычисления: {str(e)}"

    def calculate_expression(self, text):
        """Вычисляет математическое выражение - улучшенная версия"""
        try:
            numbers = re.findall(r'\d+', text)
            numbers = [int(num) for num in numbers]

            if numbers:
                if any(word in text for word in ['плюс', 'прибавь', 'сложи', '+']):
                    result = sum(numbers)
                    explanation = f"{' + '.join(map(str, numbers))} = {result}"
                    return f"{result} ({explanation})"

                elif any(word in text for word in ['минус', 'отними', 'вычти', '-']):
                    if len(numbers) >= 2:
                        result = numbers[0] - sum(numbers[1:])
                        explanation = f"{numbers[0]} - {sum(numbers[1:])} = {result}"
                        return f"{result} ({explanation})"
                    else:
                        return f"{numbers[0]}"

                elif any(word in text for word in ['умножь', 'умножить', '*', '×']):
                    result = 1
                    for num in numbers:
                        result *= num
                    explanation = f"{' × '.join(map(str, numbers))} = {result}"
                    return f"{result} ({explanation})"

                elif any(word in text for word in ['раздели', 'дели', '/', '÷']):
                    if len(numbers) >= 2:
                        result = numbers[0]
                        for num in numbers[1:]:
                            if num != 0:
                                result /= num
                        explanation = f"{numbers[0]} ÷ {numbers[1]} = {result:.2f}"
                        return f"{result:.2f} ({explanation})"
                    else:
                        return "Недостаточно чисел для деления"

            math_text = text.lower()
            replacements = {
                'плюс': '+', 'минус': '-', 'прибавь': '+', 'отними': '-', 'вычти': '-',
                'умножь на': '*', 'умножить на': '*', 'умножь': '*', 'умножить': '*',
                'раздели на': '/', 'дели на': '/', 'раздели': '/', 'дели': '/',
                'скобка открывается': '(', 'скобка закрывается': ')',
                'открывающая скобка': '(', 'закрывающая скобка': ')'
            }

            for word, symbol in replacements.items():
                math_text = math_text.replace(word, symbol)

            math_text = re.sub(r'[^\d\+\-\*\/\(\)\.\s]', '', math_text)
            math_text = math_text.strip()

            if math_text:
                result, error = self.safe_calculate(math_text)
                if error is None:
                    explanation = f"{math_text} = {result}"
                    return f"{result} ({explanation})"
                else:
                    return f"Ошибка: {error}"
            else:
                return "Не могу распознать математическое выражение"

        except Exception as e:
            return f"Ошибка в вычислениях: {str(e)}"

    def process_command(self, command):
        """Обработка команды"""
        if not command:
            return True

        time.sleep(0.3)

        intent, confidence = self.predict_intent(command)
        print(f"Определено намерение: {intent} (уверенность: {confidence:.2f})\n")

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

        elif intent == "mood":
            response = random.choice(self.intents["mood"]["responses"])
            self.speak(response)

        elif intent == "goodbye":
            response = random.choice(self.intents["goodbye"]["responses"])
            self.speak(response)
            return False

        else:
            self.speak("Извините, я не понял команду.")

        return True

    def run(self):
        """Основной цикл работы ассистента"""
        print("Ассистент Мэтт активирован! Готов к работе.\n")
        print("Скажите 'Мэтт' для активации...\n")

        running = True
        while running:
            result = self.listen_for_activation()

            if result is True:
                command = self.listen_for_command()
                if command:
                    running = self.process_command(command)
            elif result:
                running = self.process_command(result)

            time.sleep(0.2)


def train_extended_model():
    """Функция для расширенного обучения модели"""
    print("=" * 50)
    print("ЗАПУСК РАСШИРЕННОГО ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 50)

    matt = Matt(load_existing_model=False)

    print("\nНачинаем расширенное обучение...")
    matt.train_model(epochs=2000)
    print("Расширенное обучение завершено!\n")


def test_model():
    """Функция для тестирования модели"""
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("=" * 50)

    matt = Matt()

    test_phrases = [
        "привет",
        "который час",
        "какая дата",
        "посчитай 5 плюс 3",
        "сколько будет 10 умножить на 2",
        "спасибо",
        "как дела",
        "пока"
    ]

    print("\nТестирование классификации команд:")
    for phrase in test_phrases:
        intent, confidence = matt.predict_intent(phrase)
        print(f"'{phrase}' -> {intent} (уверенность: {confidence:.2f})")

    print("\nТестирование завершено!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("АССИСТЕНТ МЭТТ")
    print("=" * 50)
    print()

    # ВАРИАНТЫ ЗАПУСКА:

    # 1. Только тестирование модели (без запуска ассистента)
    # test_model()

    # 2. Расширенное обучение модели (2000 эпох)
    # train_extended_model()

    # 3. Обычный запуск ассистента
    matt = Matt()
    matt.run()