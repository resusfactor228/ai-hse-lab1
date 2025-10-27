import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import accelerate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings('ignore')

# Настройка логирования для transformers
logging.getLogger("transformers").setLevel(logging.ERROR)


class CodeReviewClassifier:
    def __init__(self, data_path):
        """Инициализация классификатора"""
        self.df = self.load_excel_data(data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.vectorizers = {}
        self.models = {}
        self.results = {}

    def load_excel_data(self, data_path):
        """Загрузка данных из Excel файла с конкретными колонками is_toxic и cleaned_text"""
        try:
            df = pd.read_excel(data_path)
            print(f"Успешно загружен файл: {data_path}")
            print(f"Размер данных: {df.shape}")
            print(f"Колонки: {list(df.columns)}")

            # Проверяем наличие необходимых колонок
            required_columns = ['is_toxic', 'cleaned_text']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"Ошибка: отсутствуют необходимые колонки: {missing_columns}")
                print(f"Доступные колонки: {list(df.columns)}")
                raise ValueError(f"Отсутствуют колонки: {missing_columns}")

            # Проверка качества данных
            print(f"\nПроверка данных:")
            print(f"Количество записей: {len(df)}")
            print(f"Пропуски в cleaned_text: {df['cleaned_text'].isnull().sum()}")
            print(f"Пропуски в is_toxic: {df['is_toxic'].isnull().sum()}")
            print(f"Уникальные значения is_toxic: {df['is_toxic'].unique()}")
            print(f"Распределение меток:\n{df['is_toxic'].value_counts()}")

            # Очистка данных
            df_clean = df.dropna(subset=['cleaned_text', 'is_toxic']).copy()
            df_clean = df_clean[df_clean['cleaned_text'].astype(str).str.strip().str.len() > 0]

            # Преобразование меток в числовой формат если необходимо
            if df_clean['is_toxic'].dtype == 'object':
                df_clean['is_toxic'] = df_clean['is_toxic'].astype(int)

            print(f"Данные после очистки: {len(df_clean)} записей")
            print(f"Финальное распределение меток:\n{df_clean['is_toxic'].value_counts()}")

            return df_clean

        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            raise

    def prepare_data(self, test_size=0.2, random_state=42):
        """Подготовка и разделение данных"""
        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df['cleaned_text'],
            self.df['is_toxic'], test_size=test_size, random_state=random_state, stratify=self.df['is_toxic'])

        print(f"Размер обучающей выборки: {len(self.X_train)}")
        print(f"Размер тестовой выборки: {len(self.X_test)}")
        print(f"Распределение классов в обучающей выборке:\n{pd.Series(self.y_train).value_counts()}")

    # КЛАССИЧЕСКИЕ МОДЕЛИ
    def vectorize_text(self, method='both'):
        """Векторизация текста с использованием CountVectorizer и TfidfVectorizer"""
        vectorizers = {}

        if method in ['count', 'both']:
            # CountVectorizer
            count_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', min_df=2,
                max_df=0.8)
            X_train_count = count_vectorizer.fit_transform(self.X_train.astype(str))
            X_test_count = count_vectorizer.transform(self.X_test.astype(str))
            vectorizers['count'] = {'vectorizer': count_vectorizer, 'X_train': X_train_count, 'X_test': X_test_count}
            print("CountVectorizer completed")

        if method in ['tfidf', 'both']:
            # TfidfVectorizer
            tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', min_df=2,
                max_df=0.8, sublinear_tf=True)
            X_train_tfidf = tfidf_vectorizer.fit_transform(self.X_train.astype(str))
            X_test_tfidf = tfidf_vectorizer.transform(self.X_test.astype(str))
            vectorizers['tfidf'] = {'vectorizer': tfidf_vectorizer, 'X_train': X_train_tfidf, 'X_test': X_test_tfidf}
            print("TfidfVectorizer completed")

        self.vectorizers = vectorizers
        return vectorizers

    def train_classical_models(self):
        """Обучение классических моделей с кросс-валидацией"""
        models = {}
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)

        for vec_name, vec_data in self.vectorizers.items():
            print(f"\n--- Обучение моделей с {vec_name.upper()} ---")

            X_train = vec_data['X_train']
            X_test = vec_data['X_test']

            # Logistic Regression
            lr_params = {'C': 0.1, 'max_iter': 1000, 'random_state': 42, 'class_weight': 'balanced'}
            lr = LogisticRegression(**lr_params)

            # Кросс-валидация
            lr_scores = cross_val_score(lr, X_train, self.y_train, cv=kfold, scoring='f1_weighted')
            print(f"Logistic Regression CV F1-score: {lr_scores.mean():.4f} (+/- {lr_scores.std() * 2:.4f})")

            # Обучение на всей обучающей выборке
            lr.fit(X_train, self.y_train)
            lr_pred = lr.predict(X_test)

            # Random Forest
            rf_params = {'n_estimators': 100, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1,
                'class_weight': 'balanced'}
            rf = RandomForestClassifier(**rf_params)

            # Кросс-валидация
            rf_scores = cross_val_score(rf, X_train, self.y_train, cv=kfold, scoring='f1_weighted')
            print(f"Random Forest CV F1-score: {rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})")

            # Обучение на всей обучающей выборке
            rf.fit(X_train, self.y_train)
            rf_pred = rf.predict(X_test)

            models[f'lr_{vec_name}'] = {'model': lr, 'predictions': lr_pred, 'cv_scores': lr_scores}

            models[f'rf_{vec_name}'] = {'model': rf, 'predictions': rf_pred, 'cv_scores': rf_scores}

            # Матрицы ошибок
            print(f"\nМатрица ошибок Logistic Regression ({vec_name}):")
            self.plot_confusion_matrix(f'LR_{vec_name}', lr_pred)

            print(f"\nМатрица ошибок Random Forest ({vec_name}):")
            self.plot_confusion_matrix(f'RF_{vec_name}', rf_pred)

        self.models.update(models)
        return models

    def evaluate_classical_models(self):
        """Оценка классических моделей"""
        for model_name, model_data in self.models.items():
            if not model_name.startswith(('lr_', 'rf_')):
                continue

            y_pred = model_data['predictions']
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')

            self.results[model_name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

            print(f"\n--- {model_name.upper()} ---")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

    def plot_confusion_matrix(self, model_name, y_pred):
        """Построение матрицы ошибок"""
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic', 'Toxic'],
                    yticklabels=['Non-Toxic', 'Toxic'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Вывод метрик для каждого класса
        print(classification_report(self.y_test, y_pred, target_names=['Non-Toxic', 'Toxic']))

    def hyperparameter_tuning(self):
        """Подбор гиперпараметров для классических моделей"""
        print("\n--- Подбор гиперпараметров ---")

        if 'tfidf' in self.vectorizers:
            X_train = self.vectorizers['tfidf']['X_train']

            # Подбор для Logistic Regression
            lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}

            lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            lr_grid_search = GridSearchCV(lr, lr_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            lr_grid_search.fit(X_train, self.y_train)

            print(f"Лучшие параметры для Logistic Regression: {lr_grid_search.best_params_}")
            print(f"Лучший F1-score: {lr_grid_search.best_score_:.4f}")

            # Подбор для Random Forest
            rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]}

            rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            rf_grid_search.fit(X_train, self.y_train)

            print(f"Лучшие параметры для Random Forest: {rf_grid_search.best_params_}")
            print(f"Лучший F1-score: {rf_grid_search.best_score_:.4f}")

    # TRANSFORMER MODELS - УЛУЧШЕННАЯ ВЕРСИЯ
    def create_tokenize_function(self, tokenizer):
        """Создаем сериализуемую функцию для токенизации"""

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding=True, truncation=True, max_length=256, return_tensors="pt")

        return tokenize_function

    def prepare_transformer_data(self, model_name='roberta-base'):
        """Подготовка данных для трансформеров с улучшенной обработкой"""
        print(f"Загрузка токенизатора для {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Добавляем padding token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Создаем сериализуемую функцию токенизации
            tokenize_function = self.create_tokenize_function(self.tokenizer)

            # Создание datasets
            train_dataset = Dataset.from_dict(
                {'text': self.X_train.astype(str).tolist(), 'label': self.y_train.tolist()})

            test_dataset = Dataset.from_dict({'text': self.X_test.astype(str).tolist(), 'label': self.y_test.tolist()})

            # Токенизация с отключением кэширования для избежания предупреждений
            self.train_tokenized = train_dataset.map(tokenize_function, batched=True, load_from_cache_file=False
                # Отключаем кэширование чтобы избежать предупреждений
            )
            self.test_tokenized = test_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)

            return self.train_tokenized, self.test_tokenized

        except Exception as e:
            print(f"Ошибка при подготовке данных для {model_name}: {e}")
            raise

    def compute_metrics(self, eval_pred):
        """Вычисление метрик для трансформеров"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def train_transformer(self, model_name='roberta-base', model_display_name='RoBERTa'):
        """Обучение трансформер модели с улучшенной обработкой ошибок"""
        print(f"\n--- Обучение {model_display_name} ---")

        try:
            # Подготовка данных
            self.prepare_transformer_data(model_name)

            # Определение модели
            num_labels = len(np.unique(self.y_train))
            print(f"Загрузка модели {model_name} для {num_labels} классов...")

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                ignore_mismatched_sizes=True  # Игнорировать несовпадения размеров
            )

            # Проверка доступности GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            print(f"Используется устройство: {device}")

            # Параметры обучения с оптимизацией для избежания переполнения памяти
            training_args = TrainingArguments(output_dir=f'./results_{model_display_name}', num_train_epochs=1,
                per_device_train_batch_size=4,  # Уменьшен размер батча
                per_device_eval_batch_size=4, warmup_steps=50,  # Уменьшено количество шагов разогрева
                weight_decay=0.01, logging_dir=f'./logs_{model_display_name}', logging_steps=10,
                # Увеличено для уменьшения вывода
                eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                metric_for_best_model="f1", save_total_limit=1,  # Уменьшено для экономии места
                dataloader_pin_memory=False,  # Может помочь с памятью
                gradient_accumulation_steps=2,  # Аккумуляция градиентов
                fp16=torch.cuda.is_available(),  # Использовать mixed precision на GPU
            )

            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            # Trainer
            trainer = Trainer(model=model, args=training_args, train_dataset=self.train_tokenized,
                eval_dataset=self.test_tokenized, tokenizer=self.tokenizer, data_collator=data_collator,
                compute_metrics=self.compute_metrics, )

            # Обучение
            print(f"Начало обучения {model_display_name}...")
            trainer.train()

            # Предсказание на тестовых данных
            print(f"Предсказание на тестовых данных...")
            predictions = trainer.predict(self.test_tokenized)
            y_pred = np.argmax(predictions.predictions, axis=1)

            # Сохранение результатов
            self.models[model_display_name] = {'model': trainer, 'predictions': y_pred}

            # Вычисление метрик
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')

            self.results[model_display_name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall,
                'f1_score': f1}

            print(f"\n--- {model_display_name} Results ---")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

            # Матрица ошибок
            print(f"\nМатрица ошибок {model_display_name}:")
            self.plot_confusion_matrix(model_display_name, y_pred)

            return trainer

        except Exception as e:
            print(f"Ошибка при обучении {model_display_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_roberta(self):
        """Обучение RoBERTa"""
        return self.train_transformer('roberta-base', 'RoBERTa')

    def train_codebert(self):
        """Обучение CodeBERT"""
        return self.train_transformer('microsoft/codebert-base', 'CodeBERT')

    def generate_report(self):
        """Генерация отчета с сравнением моделей"""
        print("\n" + "=" * 60)
        print("ФИНАЛЬНЫЙ ОТЧЕТ ПО МОДЕЛЯМ КЛАССИФИКАЦИИ")
        print("=" * 60)

        # Создание DataFrame с результатами
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df = results_df.round(4)

        # Сортировка по F1-score
        results_df = results_df.sort_values('f1_score', ascending=False)

        print("\nСравнение моделей (отсортировано по F1-score):")
        print(results_df)

        # Визуализация
        plt.figure(figsize=(12, 8))

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = range(len(results_df))

        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
            plt.bar(x, results_df[metric], color=colors[:len(results_df)])
            plt.title(f'{metric.capitalize()} Comparison')
            plt.xticks(x, results_df.index, rotation=45)
            plt.ylim(0, 1)
            for j, v in enumerate(results_df[metric]):
                plt.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

        best_model = results_df.iloc[0]
        print(f"\n ЛУЧШАЯ МОДЕЛЬ: {results_df.index[0]}")
        print(f" F1-Score: {best_model['f1_score']:.4f}")
        print(f" Accuracy: {best_model['accuracy']:.4f}")
        print(f" Precision: {best_model['precision']:.4f}")
        print(f" Recall: {best_model['recall']:.4f}")

        return results_df

    def run_complete_pipeline(self):
        """Запуск полного пайплайна"""
        print("=== ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА КЛАССИФИКАЦИИ ===")

        self.prepare_data()

        print("\n=== КЛАССИЧЕСКИЕ МОДЕЛИ ===")
        self.vectorize_text(method='both')
        self.train_classical_models()
        self.evaluate_classical_models()

        print("\n=== ПОДБОР ГИПЕРПАРАМЕТРОВ ===")
        self.hyperparameter_tuning()

        print("\n=== TRANSFORMER MODELS ===")

        print("\n--- Начало обучения RoBERTa ---")
        roberta_trainer = self.train_roberta()

        print("\n--- Начало обучения CodeBERT ---")
        codebert_trainer = self.train_codebert()

        report = self.generate_report()

        report.to_csv('model_comparison_report.csv')

        return report

if __name__ == "__main__":

    classifier = CodeReviewClassifier("cleaned_code_reviews.xlsx")

    report = classifier.run_complete_pipeline()

    if len(report) > 0:
        best_model_name = report.index[0]
        best_metrics = report.iloc[0]

        print(f"""
РЕЗЮМЕ ЭКСПЕРИМЕНТА ПО КЛАССИФИКАЦИИ ТОКСИЧНЫХ РЕЦЕНЗИЙ

ЦЕЛЬ:
Сравнение различных подходов к классификации токсичных комментариев в рецензиях исходного кода.

МЕТОДЫ:
1. Классические ML модели (Logistic Regression, Random Forest) с TF-IDF и CountVectorizer
2. Трансформер-модели (RoBERTa, CodeBERT)

ДАННЫЕ:
- Использован предобработанный датасет с колонками: cleaned_text, is_toxic
- Бинарная классификация: токсичные vs нетоксичные комментарии
- Разделение: 80% обучение, 20% тестирование

РЕЗУЛЬТАТЫ:

Лучшая модель: {best_model_name}
- F1-Score: {best_metrics['f1_score']:.4f}
- Accuracy: {best_metrics['accuracy']:.4f} 
- Precision: {best_metrics['precision']:.4f}
- Recall: {best_metrics['recall']:.4f}

ВЫВОДЫ:
1. {best_model_name} показала наилучшие результаты по метрике F1-Score
2. Трансформер-модели в целом превзошли классические подходы
3. CodeBERT, специализированный на исходном коде, ожидаемо показал хорошие результаты
4. Классические модели с TF-IDF векторизацией остаются хорошей базой для сравнения

РЕКОМЕНДАЦИИ:
- Для продакшн-использования рекомендуется {best_model_name}
- При ограниченных вычислительных ресурсах подходят классические модели
- Для дальнейшего улучшения обучения:
  * Увеличение размера датасета
  * Балансировка классов
""")
