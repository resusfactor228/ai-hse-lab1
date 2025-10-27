import pandas as pd
import re
import numpy as np
from collections import defaultdict


class CodeReviewDataPreprocessor:
    def __init__(self):
        self.contraction_map = {"don't": "do not", "doesn't": "does not", "isn't": "is not", "aren't": "are not",
            "can't": "cannot", "couldn't": "could not", "won't": "will not", "we're": "we are", "i'm": "i am"}

        # ИСПРАВЛЕННЫЙ СЛОВАРЬ obscene words
        self.obscene_words_patterns = self._load_obscene_words()

    def _load_obscene_words(self):
        """
        Загрузка и нормализация словаря obscene words
        """
        # Исходные данные (пример из вашего описания)
        raw_obscene_dict = {
            ' fuck ': ['(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
            '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)', 'f u u c',
            '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\\*', 'feck ', ' fux ', 'f\\*\\*', 'f\\-ing', 'f\\.u\\.', 'f###', ' fu ',
            'f@ck', 'f u c k', 'f uck', 'f ck'

            ],

            ' crap ': [' (c)(r|[^a-z0-9 ])(a|[^a-z0-9 ])(p|[^a-z0-9 ])([^ ])*',
                ' (c)([^a-z]*)(r)([^a-z]*)(a)([^a-z]*)(p)', ' c[!@#\\$%\\^\\&\\*]*r[!@#\\$%\\^&\\*]*p', 'cr@p', ' c r a p',

            ],

            ' ass ': ['[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\\$\\$'
                                                                     '[^a-z]anus', ' a\\*s\\*s', '[^a-z]ass[^a-z ]',
                'a[@#\\$%\\^&\\*][@#\\$%\\^&\\*]', '[^a-z]anal ', 'a s s'
            ],

            ' ass hole ': [' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\\$\\$hole'],

            ' bitch ': ['bitches', ' b[w]*i[t]*ch', ' b!tch', ' bi\\+ch', ' b!\\+ch',
                ' (b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)', ' biatch', ' bi\\*\\*h', ' bytch', 'b i t c h'],

            ' bastard ': ['ba[s|z]+t[e|a]+rd'],

            ' transgender': ['transgender'],

            ' gay ': ['gay', 'homo'],

            ' cock ': ['[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',
                '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'],

            ' dick ': [' dick[^aeiou]', 'd i c k'],

            ' suck ': ['sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'],

            ' cunt ': ['cunt', 'c u n t'],

            ' bull shit ': ['bullsh\\*t', 'bull\\$hit', 'bull sh.t'],

            ' jerk ': ['jerk'],

            ' idiot ': ['i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots' 'i d i o t'],

            ' dumb ': ['(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'],

            ' shit ': ['shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\\$hit', 's h i t', 'sh\\*tty',
                'sh\\*ty', 'sh\\*t'],

            ' shit hole ': ['shythole', 'sh.thole'],

            ' retard ': ['returd', 'retad', 'retard', 'wiktard', 'wikitud'],

            ' rape ': ['raped'],

            ' dumb ass': ['dumbass', 'dubass'],

            ' ass head': ['butthead'],

            ' sex ': ['sexy', 's3x', 'sexuality'],

            ' nigger ': ['nigger', 'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'],

            ' shut the fuck up': [' stfu' '^stfu'],

            ' for your fucking information': [' fyfi', '^fyfi'], ' get the fuck off': ['gtfo', '^gtfo'],

            ' oh my fucking god ': [' omfg', '^omfg'],

            ' what the hell ': [' wth', '^wth'],

            ' what the fuck ': [' wtf', '^wtf'], ' son of bitch ': [' sob ', '^sob '],

            ' pussy ': ['pussy[^c]', 'pusy', 'pussi[^l]', 'pusses',
                '(p)(u|[^a-z0-9 ])(s|[^a-z0-9 ])(s|[^a-z0-9 ])(y)', ],

            ' faggot ': ['faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',
                '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot', ],

            ' mother fucker': [' motha f', ' mother f', 'motherucker', ' mofo', ' mf ', ],

            ' whore ': ['wh\\*\\*\\*', 'w h o r e'],

            ' haha ': ['ha\\*\\*\\*ha', ],
        }

        # Нормализация словаря
        normalized_dict = {}
        for word, patterns in raw_obscene_dict.items():
            # Убираем пробелы в ключах
            clean_word = word.strip()

            normalized_patterns = []
            for pattern in patterns:
                # Если паттерн - просто слово, добавляем границы слов
                if self._is_simple_word(pattern):
                    normalized_pattern = r'\b' + re.escape(pattern) + r'\b'
                else:
                    # Для regex паттернов добавляем границы слов если их нет
                    if not pattern.startswith('\\b') and not pattern.startswith('(?<!'):
                        normalized_pattern = r'\b' + pattern + r'\b'
                    else:
                        normalized_pattern = pattern

                normalized_patterns.append(normalized_pattern)

            normalized_dict[clean_word] = normalized_patterns

        return normalized_dict

    def _is_simple_word(self, text):
        """Проверяет, является ли текст простым словом (без спец. символов regex)"""
        regex_chars = r'[]().*+?^${}|\\'
        return not any(char in text for char in regex_chars)

    def _safe_regex_replace(self, text, pattern, replacement):
        """
        Безопасная замена с обработкой ошибок regex
        """
        try:
            return re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        except re.error as e:
            print(f"Ошибка в regex паттерне: {pattern} - {e}")
            return text

    def correct_obscene_words(self, text):
        """Коррекция obscene слов с улучшенной логикой"""
        if not isinstance(text, str):
            return ""

        for replacement_word, patterns in self.obscene_words_patterns.items():
            for pattern in patterns:
                text = self._safe_regex_replace(text, pattern, replacement_word)

        return text

    def load_excel_data(self, filepath, sheet_name=0, text_column='text', label_column=None):
        """Загрузка данных из Excel файла"""
        try:
            self.df = pd.read_excel(filepath, sheet_name=sheet_name)
            print(f"Успешно загружен Excel файл: {filepath}")
            print(f"Размер данных: {self.df.shape}")

            # Проверка колонок
            if text_column not in self.df.columns:
                available_columns = list(self.df.columns)
                print(f"Колонка '{text_column}' не найдена. Доступные колонки: {available_columns}")
                if available_columns:
                    text_column = available_columns[0]
                    print(f"Используется первая доступная колонка: '{text_column}'")

            self.text_column = text_column
            self.label_column = label_column

            print(f"\nПервые 3 строки данных:")
            print(self.df.head(3))

        except Exception as e:
            print(f"Ошибка при загрузке Excel файла: {e}")
            raise

    def validate_obscene_patterns(self, test_texts=None):
        """
        Валидация obscene паттернов на тестовых текстах
        """
        if test_texts is None:
            test_texts = ["This is dumb code", "This is d-u-m-b code", "What a shitty implementation",
                "This is s h i t", "This is sh*t", "This is a good implementation"  # негативный пример
            ]

        print("\n" + "=" * 50)
        print("ВАЛИДАЦИЯ OBSCENE PATTERNS")
        print("=" * 50)

        for test_text in test_texts:
            cleaned = self.correct_obscene_words(test_text)
            print(f"ДО: {test_text}")
            print(f"ПОСЛЕ: {cleaned}")
            print("-" * 40)

    def explore_obscene_words_distribution(self):
        """
        Анализ распределения obscene слов в датасете
        """
        if not hasattr(self, 'df') or self.text_column not in self.df.columns:
            print("Данные не загружены")
            return

        print("\nАНАЛИЗ OBSCENE СЛОВ В ДАТАСЕТЕ:")

        # Временная колонка для анализа
        temp_df = self.df.copy()
        temp_df['contains_obscene'] = False

        for word, patterns in self.obscene_words_patterns.items():
            for pattern in patterns:
                mask = temp_df[self.text_column].str.contains(pattern, case=False, na=False, regex=True)
                temp_df.loc[mask, 'contains_obscene'] = True

        obscene_count = temp_df['contains_obscene'].sum()
        total_count = len(temp_df)

        print(f"Текстов с obscene словами: {obscene_count}/{total_count} ({obscene_count / total_count * 100:.2f}%)")

        if obscene_count > 0:
            print("\nПримеры текстов с obscene словами:")
            obscene_samples = temp_df[temp_df['contains_obscene']].head(5)
            for idx, row in obscene_samples.iterrows():
                print(f"- {row[self.text_column][:100]}...")

    def preprocess_text(self, text):
        """Полный пайплайн предобработки текста"""
        if not isinstance(text, str):
            return ""

        processing_pipeline = [self.remove_urls, self.remove_code_snippets, self.expand_contractions,
            self.remove_repeated_chars, self.correct_obscene_words,  # obscene words correction
            self.remove_special_chars, self.additional_cleaning]

        for func in processing_pipeline:
            text = func(text)

        return text

    # Остальные методы остаются без изменений
    def remove_urls(self, text):
        if not isinstance(text, str):
            return ""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    def remove_code_snippets(self, text):
        if not isinstance(text, str):
            return ""
        code_pattern = r'```.*?```|`.*?`'
        return re.sub(code_pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    def expand_contractions(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        for contraction, expansion in self.contraction_map.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        return text

    def remove_repeated_chars(self, text):
        if not isinstance(text, str):
            return ""
        pattern = r'(.)\1{2,}'
        return re.sub(pattern, r'\1\1', text)

    def remove_special_chars(self, text):
        if not isinstance(text, str):
            return ""
        cleaned_text = re.sub(r'[&|#|^|*|@|~|`]', '', text)
        return cleaned_text

    def additional_cleaning(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = ' '.join(text.split())
        return text

    def clean_data(self, remove_duplicates=True, remove_empty_texts=True):
        initial_size = len(self.df)
        print(f"Начальный размер данных: {initial_size}")

        self.df = self.df.dropna(subset=[self.text_column])
        after_missing = len(self.df)
        print(f"Удалено пропущенных значений: {initial_size - after_missing}")

        if remove_empty_texts:
            self.df = self.df[self.df[self.text_column].astype(str).str.strip().str.len() > 0]
            after_empty = len(self.df)
            print(f"Удалено пустых текстов: {after_missing - after_empty}")

        if remove_duplicates:
            self.df = self.df.drop_duplicates(subset=[self.text_column])
            after_duplicates = len(self.df)
            print(f"Удалено дубликатов: {after_empty - after_duplicates}")

        print(f"Финальный размер данных: {len(self.df)}")

    def preprocess_dataset(self, show_examples=True):
        print("\nНачало предобработки текстов...")

        self.df['original_text'] = self.df[self.text_column].copy()
        self.df['cleaned_text'] = self.df[self.text_column].apply(self.preprocess_text)

        initial_size = len(self.df)
        self.df = self.df[self.df['cleaned_text'].str.len() > 0]
        print(f"Удалено пустых текстов после очистки: {initial_size - len(self.df)}")

        if show_examples:
            self.show_cleaning_examples()

    def show_cleaning_examples(self, num_examples=3):
        print(f"\nПримеры предобработки (первые {num_examples}):")
        print("=" * 80)

        for i in range(min(num_examples, len(self.df))):
            original = str(self.df.iloc[i]['original_text'])
            cleaned = str(self.df.iloc[i]['cleaned_text'])

            print(f"\nПример {i + 1}:")
            print(f"ДО: {original[:100]}{'...' if len(original) > 100 else ''}")
            print(f"ПОСЛЕ: {cleaned[:100]}{'...' if len(cleaned) > 100 else ''}")
            print("-" * 40)


# Пример использования с валидацией
def main():
    preprocessor = CodeReviewDataPreprocessor()

    # Валидация паттернов перед обработкой
    preprocessor.validate_obscene_patterns()

    # Загрузка данных из Excel
    preprocessor.load_excel_data(filepath="ToxiCR/models/code-review-dataset-full.xlsx", text_column='review_text', label_column='label')

    # Анализ распределения obscene слов
    preprocessor.explore_obscene_words_distribution()

    # Очистка и предобработка
    preprocessor.clean_data()
    preprocessor.preprocess_dataset(show_examples=True)

    # Сохранение результатов
    preprocessor.df.to_excel("cleaned_code_reviews.xlsx", index=False)
    print("Очищенные данные сохранены в 'cleaned_code_reviews.xlsx'")


if __name__ == "__main__":
    main()