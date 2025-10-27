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


class UltraMegaJarvis:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()

        # УЛЬТРА МЕГА ДАННЫЕ - профессиональный уровень
        self.intents = self.create_mega_dataset()

        self.vectorizer = None
        self.label_encoder = None
        self.model = None
        self.train_ultra_mega_network()

        print("Профессиональная калибровка микрофона...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=3)
        print("Калибровка завершена!")

    def create_mega_dataset(self):
        """Создание ультра-мега датасета"""
        return {
            "greeting": {
                "patterns": self.generate_variations([
                    "привет", "здравствуй", "добрый день", "хай", "hello",
                    "доброе утро", "добрый вечер", "приветствую", "салют",
                    "привет друг", "привет ассистент", "здравствуйте", "приветик",
                    "доброй ночи", "привет всем", "здорово", "мое почтение",
                    "доброго времени суток", "приветствую вас", "рад тебя видеть",
                    "привет как дела", "привет старший", "здравствуй джарвис",
                    "привет умник", "добрый", "приветствие", "здарова", "хай там",
                    "приветствую тебя", "доброго здоровья", "приветики", "здоровенько",
                    "доброго дня", "приветик всем", "здаров", "привет народ",
                    "добрый вечерок", "доброе утречко", "приветики пистолетики"
                ], 50),
                "responses": [
                    "Здравствуйте!", "Привет! Как дела?", "Рад вас слышать!", "Приветствую!",
                    "Здорово!", "Привет! Чем могу помочь?", "Добрый день!", "Привет, друг!",
                    "Приветствую вас!", "Здравствуй! Как настроение?"
                ]
            },
            "time": {
                "patterns": self.generate_variations([
                    "который час", "сколько время", "время", "time", "current time",
                    "скажи время", "подскажи время", "сколько времени", "который сейчас час",
                    "время сейчас", "текущее время", "сколько сейчас времени", "который час времени",
                    "подскажи который час", "скажи который час", "время узнать", "узнать время",
                    "сколько на часах", "который час на часах", "время суток", "точное время",
                    "подскажи текущее время", "скажи текущее время", "время сейчас сколько",
                    "который час подскажи", "сколько времени сейчас", "время узнать сейчас",
                    "текущее время узнать", "время на данный момент", "который час времени сейчас"
                ], 40),
                "responses": ["Сейчас {}", "Текущее время: {}", "На часах {}", "Время: {}"]
            },
            "date": {
                "patterns": self.generate_variations([
                    "какое число", "какая дата", "дата", "date", "today",
                    "какой сегодня день", "подскажи дату", "какое сегодня число",
                    "число сегодня", "текущая дата", "какой сегодня день месяца",
                    "подскажи какое число", "скажи дату", "какое число сегодня",
                    "какой сегодня число", "дата сегодня", "сегодняшняя дата",
                    "какой день сегодня", "число месяца", "текущее число",
                    "подскажи сегодняшнюю дату", "скажи сегодняшнее число", "какое сегодня число месяца",
                    "текущая дата сегодня", "сегодня какое число", "число сегодняшнее",
                    "дата текущего дня", "какой сегодня день недели", "текущий день месяца"
                ], 40),
                "responses": ["Сегодня {}", "Текущая дата: {}", "Сегодняшнее число: {}", "Дата: {}"]
            },
            "calculation": {
                "patterns": self.generate_variations([
                    "посчитай", "вычисли", "сколько будет", "calculate", "calc",
                    "сложи", "прибавь", "отними", "умножь", "раздели",
                    "сложи числа", "вычисли пример", "реши пример", "посчитай пример",
                    "математика", "арифметика", "реши задачу", "вычисли результат",
                    "посчитай сумму", "сложи цифры", "прибавь числа", "отними числа",
                    "умножь числа", "раздели числа", "математическая операция",
                    "сколько получится", "какой результат", "вычисли значение",
                    "реши математический пример", "вычисли арифметическое выражение",
                    "посчитай выражение", "вычисли сумму", "найди разность", "умножь цифры",
                    "раздели цифры", "вычисли произведение", "найди частное"
                ], 50),
                "responses": ["Результат: {}", "Ответ: {}", "Получается: {}", "Вычисляю: {}",
                              "Результат вычислений: {}"]
            },
            "thanks": {
                "patterns": self.generate_variations([
                    "спасибо", "благодарю", "молодец", "thanks", "thank you",
                    "ты лучший", "отлично", "хорошая работа", "отлично работает",
                    "спасибо помощник", "благодарю тебя", "спасибо большое",
                    "огромное спасибо", "благодарствую", "признателен", "ценю",
                    "ты крут", "отлично справился", "хорошо работаешь", "отличная работа",
                    "благодарю за помощь", "спасибо за помощь", "ты супер", "отлично помог",
                    "выручил", "помог хорошо", "работаешь отлично", "спасибо дружище",
                    "благодарность", "признательность", "ценю помощь", "спасибо за поддержку"
                ], 40),
                "responses": [
                    "Всегда рад помочь!", "Пожалуйста!", "Обращайтесь!", "Рад был помочь!",
                    "Не за что!", "Для вас всегда!", "Это моя работа!", "Всегда к вашим услугам!"
                ]
            },
            "mood": {
                "patterns": self.generate_variations([
                    "как дела", "как настроение", "как себя чувствуешь", "как жизнь",
                    "как твои дела", "что нового", "как ты", "how are you",
                    "расскажи о настроении", "какое у тебя настроение", "как твое самочувствие",
                    "как поживаешь", "что у тебя нового", "как твои успехи", "как ты сегодня",
                    "как твое здоровье", "что расскажешь", "как жизнь молодая", "как сам",
                    "как у тебя дела", "настроение какое", "как твое настроение сегодня",
                    "что нового у тебя", "как твои дела сегодня", "как твое самочувствие сегодня",
                    "как твое состояние", "как твои дела вообще", "что нового в жизни",
                    "как твое положение", "как твои успехи сегодня", "как твое здоровье сегодня",
                    "как твое настроение в целом", "что нового расскажешь", "как твои дела вообще"
                ], 50),
                "responses": [
                    "Всё отлично, спасибо!",
                    "Прекрасно! Готов помогать.",
                    "Как у настоящего ИИ - без эмоций, но эффективно!",
                    "Замечательно! А у вас?",
                    "Отлично! Программируюсь и радуюсь жизни!",
                    "Супер! Готов к новым задачам!",
                    "Все хорошо, работаю в штатном режиме!",
                    "Настроение отличное! А у вас?",
                    "Лучше всех! Чем могу помочь?",
                    "Великолепно! Готов к работе!"
                ]
            },
            "goodbye": {
                "patterns": self.generate_variations([
                    "пока", "до свидания", "выход", "bye", "goodbye",
                    "закончим", "до встречи", "прощай", "всего хорошего",
                    "пока пока", "до завтра", "спокойной ночи", "всего доброго",
                    "до скорого", "увидимся", "бывай", "прощевайте", "до свиданья",
                    "пока дружище", "до следующего раза", "завершить работу",
                    "отключись", "выключись", "закончить", "стоп", "хватит",
                    "до скорой встречи", "всего наилучшего", "пока всего хорошего",
                    "до новых встреч", "заканчиваем", "прощай друг", "до завтрашнего дня",
                    "спокойной ночи друг", "всего доброго друг", "бывай здоров"
                ], 40),
                "responses": [
                    "До свидания!", "Удачи!", "Был рад помочь!", "До новых встреч!",
                    "Пока! Обращайтесь еще!", "Всего доброго!", "До скорой встречи!",
                    "Спокойной ночи!", "Всего наилучшего!", "Бывай!"
                ]
            }
        }

    def generate_variations(self, base_phrases, target_count):
        """Генерация вариаций фраз"""
        variations = base_phrases.copy()

        if len(variations) >= target_count:
            return variations[:target_count]

        # Добавляем синонимы и вариации
        synonyms = {
            "привет": ["приветик", "приветствую", "здравствуй"],
            "спасибо": ["благодарю", "мерси", "thanks"],
            "пока": ["до свидания", "прощай", "бывай"],
            "как дела": ["как жизнь", "как сам", "как поживаешь"],
            "время": ["который час", "сколько времени", "время суток"],
            "дата": ["какое число", "какой день", "текущая дата"]
        }

        while len(variations) < target_count:
            for phrase in base_phrases:
                if len(variations) >= target_count:
                    break

                # Добавляем синонимы
                for word, syns in synonyms.items():
                    if word in phrase and len(variations) < target_count:
                        for syn in syns:
                            new_phrase = phrase.replace(word, syn)
                            if new_phrase not in variations:
                                variations.append(new_phrase)
                                if len(variations) >= target_count:
                                    break

                # Добавляем с восклицательными знаками
                if len(variations) < target_count and random.random() > 0.7:
                    variations.append(phrase + "!")

                # Добавляем с вопросами
                if len(variations) < target_count and random.random() > 0.7:
                    variations.append(phrase + "?")

        return variations[:target_count]

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
        """Профессиональная предобработка текста"""
        text = text.lower()
        # Удаляем лишние символы, но сохраняем важные для интонации
        text = re.sub(r'[^\w\s!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_ultra_mega_network(self, input_dim, output_dim):
        """Создание УЛЬТРА-МЕГА архитектуры нейросети"""
        model = Sequential([
            # Первый слой - мощный и широкий
            Dense(1024, activation='relu', input_shape=(input_dim,),
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.6),

            # Второй слой
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),

            # Третий слой
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),

            # Четвертый слой
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),

            # Пятый слой
            Dense(64, activation='relu'),
            Dropout(0.2),

            # Выходной слой
            Dense(output_dim, activation='softmax')
        ])

        # Профессиональный компилятор
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.0005,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy']
        )

        return model

    def train_ultra_mega_network(self):
        """УЛЬТРА-МЕГА обучение нейросети"""
        print("🚀 Запуск УЛЬТРА-МЕГА обучения нейросети...")

        # Подготовка данных
        texts, labels = self.prepare_training_data()

        print(f"📊 Всего примеров для обучения: {len(texts)}")
        print(f"🎯 Количество классов: {len(set(labels))}")

        # Профессиональный векторизатор
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            max_features=2000,
            min_df=1,
            max_df=0.85,
            stop_words=None,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )

        X = self.vectorizer.fit_transform(texts).toarray()
        print(f"🔢 Размерность данных после векторизации: {X.shape}")

        # Кодирование меток
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        # Разделение на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.15,
            random_state=42,
            stratify=y
        )

        print(f"📚 Обучающая выборка: {X_train.shape[0]} примеров")
        print(f"🔍 Валидационная выборка: {X_val.shape[0]} примеров")

        # Создание модели
        self.model = self.create_ultra_mega_network(
            X_train.shape[1],
            len(self.label_encoder.classes_)
        )

        print("🧠 Архитектура нейросети:")
        self.model.summary()

        # ПРОФЕССИОНАЛЬНЫЕ callback'ы
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=0.00001,
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            'best_jarvis_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        print("🔥 Начинаем МЕГА обучение...")

        # МЕГА обучение
        history = self.model.fit(
            X_train, y_train,
            epochs=500,
            batch_size=64,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            shuffle=True
        )

        # Загрузка лучшей модели
        if os.path.exists('best_jarvis_model.h5'):
            self.model.load_weights('best_jarvis_model.h5')
            print("✅ Загружена лучшая модель из checkpoint!")

        # Анализ результатов
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        print(f"\n🎉 УЛЬТРА-МЕГА обучение завершено!")
        print(f"📈 Финальная точность на обучении: {final_train_acc:.4f}")
        print(f"📊 Финальная точность на валидации: {final_val_acc:.4f}")
        print(f"📉 Финальные потери на обучении: {final_train_loss:.4f}")
        print(f"📋 Финальные потери на валидации: {final_val_loss:.4f}")

        # Сохранение модели
        self.save_model()

    def save_model(self):
        """Сохранение обученной модели"""
        model_data = {
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'intents': self.intents
        }

        with open('jarvis_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        print("💾 Модель сохранена в jarvis_model.pkl")

    def load_model(self):
        """Загрузка обученной модели"""
        try:
            with open('jarvis_model.pkl', 'rb') as f:
                model_data = pickle.load(f)

            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.intents = model_data['intents']

            # Нужно пересоздать архитектуру модели
            texts, _ = self.prepare_training_data()
            X_sample = self.vectorizer.transform(texts[:1]).toarray()

            self.model = self.create_ultra_mega_network(
                X_sample.shape[1],
                len(self.label_encoder.classes_)
            )

            # Загружаем веса
            self.model.load_weights('best_jarvis_model.h5')

            print("✅ Модель загружена успешно!")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def predict_intent(self, text):
        """УЛЬТРА-точное предсказание намерения"""
        if not text:
            return None, 0.0

        processed_text = self.preprocess_text(text)

        try:
            text_vector = self.vectorizer.transform([processed_text]).toarray()
            prediction = self.model.predict(text_vector, verbose=0)
            intent_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # Адаптивный порог уверенности
            adaptive_threshold = 0.6  # Высокий порог для надежности

            if confidence > adaptive_threshold:
                return self.label_encoder.inverse_transform([intent_index])[0], confidence
            else:
                return "unknown", confidence

        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            return "unknown", 0.0

    def speak(self, text):
        """Произносит текст"""
        print(f"🤖 Джарвис: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen(self):
        """Слушает и распознает речь"""
        try:
            with self.microphone as source:
                print("🎤 Слушаю...")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=8)

            command = self.recognizer.recognize_google(audio, language="ru-RU")
            print(f"👤 Вы сказали: {command}")
            return command.lower()

        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("❌ Не удалось распознать речь")
            return ""
        except sr.RequestError as e:
            print(f"❌ Ошибка сервиса распознавания: {e}")
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
        """Профессиональное вычисление математического выражения"""
        try:
            text = text.lower()

            # Расширенная замена слов на операторы
            replacements = {
                'плюс': '+', 'минус': '-', 'прибавить': '+', 'отнять': '-',
                'умножить на': '*', 'умножить': '*', 'умножит': '*',
                'делить на': '/', 'разделить на': '/', 'делить': '/',
                'и': '', 'на': '', 'сколько будет': '', 'посчитай': '',
                'вычисли': '', 'сумма': '+', 'разность': '-', 'произведение': '*'
            }

            for word, replacement in replacements.items():
                text = text.replace(word, replacement)

            # Извлекаем все числа (включая десятичные)
            numbers = re.findall(r'\d+\.?\d*', text)
            numbers = [float(num) for num in numbers]

            if not numbers:
                return "Не найдены числа для вычисления"

            # Определяем операцию
            if '+' in text:
                result = sum(numbers)
                return f"{result}"
            elif '-' in text:
                result = numbers[0] - sum(numbers[1:]) if len(numbers) > 1 else numbers[0]
                return f"{result}"
            elif '*' in text:
                result = 1
                for num in numbers:
                    result *= num
                return f"{result}"
            elif '/' in text:
                if len(numbers) >= 2:
                    result = numbers[0]
                    for num in numbers[1:]:
                        if num != 0:
                            result /= num
                    return f"{result:.2f}"
                else:
                    return "Недостаточно чисел для деления"
            else:
                # Если операция не указана, возвращаем первое число
                return f"{numbers[0]}"

        except Exception as e:
            return f"Ошибка в вычислениях: {str(e)}"

    def process_command(self, command):
        """Обработка команды"""
        if not command:
            return True

        intent, confidence = self.predict_intent(command)
        print(f"🧠 УЛЬТРА-нейросеть определила: {intent} (уверенность: {confidence:.2f})")

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
            self.speak("Извините, я не понял команду. Попробуйте сказать 'привет', 'время', 'дата' или 'как дела'.")

        return True

    def run(self):
        """Основной цикл работы ассистента"""
        self.speak("УЛЬТРА-МЕГА нейросетевой Джарвис активирован! Готов к работе.")

        running = True
        while running:
            command = self.listen()
            running = self.process_command(command)


def test_ultra_mega_network():
    """Тестирование УЛЬТРА-МЕГА нейросети"""
    print("🧪 Запуск тестирования УЛЬТРА-МЕГА нейросети...")

    jarvis = UltraMegaJarvis()

    test_phrases = [
        "привет",
        "привет друг",
        "здравствуйте",
        "который час",
        "какая дата",
        "посчитай 5 плюс 3",
        "сколько будет 10 умножить на 2",
        "спасибо",
        "как дела",
        "как настроение",
        "что нового",
        "как твои дела",
        "пока",
        "до свидания",
        "добрый день",
        "скажи время",
        "какое сегодня число",
        "вычисли 15 минус 7",
        "прибавь 20 к 30"
    ]

    print("\n" + "=" * 80)
    print("🧪 ТЕСТИРОВАНИЕ УЛЬТРА-МЕГА НЕЙРОСЕТИ:")
    print("=" * 80)

    results = []
    for phrase in test_phrases:
        intent, confidence = jarvis.predict_intent(phrase)
        status = "✅" if confidence > 0.7 else "⚠️" if confidence > 0.5 else "❌"
        results.append((phrase, intent, confidence, status))
        print(f"{status} '{phrase}' -> {intent} (уверенность: {confidence:.2f})")

    # Статистика
    successful = sum(1 for _, _, _, status in results if status == "✅")
    total = len(results)

    print(f"\n📊 Результаты: {successful}/{total} успешных распознаваний ({successful / total * 100:.1f}%)")


if __name__ == "__main__":
    # Тестирование
    test_ultra_mega_network()

    # Запуск ассистента
    print("\n" + "=" * 80)
    print("🚀 ЗАПУСК УЛЬТРА-МЕГА АССИСТЕНТА:")
    print("=" * 80)

    jarvis = UltraMegaJarvis()
    jarvis.run()