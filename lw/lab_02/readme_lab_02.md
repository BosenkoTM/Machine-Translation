# Лабораторная работа 2

## «Нейросетевой перевод на базе Hugging Face Transformers (MarianMT, NLLB)»

**Курс:** «Основы машинного перевода» · **Разделы:** 2 и 4
**Максимум: 20 баллов** · **Срок сдачи:** 01.11.2026
**Среда выполнения:** Google Colab (Python 3.10+, GPU T4)
**Сквозной кейс:** перевод того же обучающего модуля по информатике, что и в ЛР № 1, двумя
открытыми NMT-моделями.

---

## 1. Цель и формируемые компетенции

**Цель** — развернуть NMT-пайплайн в Colab, исследовать субсловную токенизацию, батчинг,
условную кросс-энтропию и механизм внимания, а также сравнить специализированную EN→RU модель
**MarianMT** с многоязычной **NLLB**.

| Компетенция | Что проверяется |
|---|---|
| **ОПК-2** | применение NMT при создании локализованных образовательных материалов |
| **ОПК-9** | программное использование NLP-моделей и библиотек |
| **УК-1** | анализ поведения двух моделей и причин переводческих ошибок |
| **УК-6** | организация воспроизводимого эксперимента с фиксированными данными и версиями |

---

## 2. Входные данные: результаты ЛР № 1

Работа продолжает **тот же индивидуальный вариант**, что и ЛР № 1. Нужны три файла:

| Файл | Обязательные колонки | Минимальный объём |
|---|---|---|
| `segments_en.csv` | `segment_id`, `source` | **120 сегментов** |
| `glossary.csv` | `en`, `ru` | **100 терминов** |
| `aligned_reference.csv` | `segment_id`, `source`, `reference` | те же 120 сегментов |

`reference` — подтверждённый человеком перевод; он нужен для расчёта кросс-энтропии и для
сравнения моделей.

> **Если корпус ЛР № 1 меньше требуемого объёма**, расширьте исходный учебный модуль и
> глоссарий **до начала работы**. Все выводы лабораторной опираются на статистику: на полутора
> десятках сегментов доля нарушений глоссария и различия моделей недостоверны. Расширять
> корпус допускается за счёт материалов той же темы (учебники, документация, курсы) с
> обязательной ручной проверкой эталонного перевода.

### Как подключить файлы в Colab

**Способ 1 — загрузка с компьютера** (блок 2 ноутбука):

```python
from google.colab import files
uploaded = files.upload()          # выбрать три CSV
for name, content in uploaded.items():
    (DATA / name).write_bytes(content)
```

**Способ 2 — Google Drive** (удобнее, если работа идёт в несколько сессий):

```python
from google.colab import drive
drive.mount("/content/drive")
import shutil
for name in ("segments_en.csv", "glossary.csv", "aligned_reference.csv"):
    shutil.copy(f"/content/drive/MyDrive/lab01/{name}", DATA / name)
```

> Файлы Colab **не сохраняются** между сессиями: после отключения среды каталог `/content`
> очищается. Скачивайте архив с результатами до закрытия вкладки (блок 16) или храните рабочие
> файлы на Google Drive.

---

## 3. Требования к среде

* **Runtime → Change runtime type → T4 GPU.** На 120 сегментах и двух моделях работа без GPU
  занимает часы: NLLB-600M на CPU выдаёт примерно 3–6 с на сегмент.
* Первый запуск скачивает около **2.7 ГБ** весов (MarianMT ≈ 300 МБ, NLLB-600M ≈ 2.4 ГБ).
* Ориентировочное время на T4: перевод 120 сегментов двумя моделями с батчингом — 3–6 минут,
  весь ноутбук целиком — 15–25 минут вместе с загрузкой моделей.
* Установка (в Colab `torch` уже есть):

```python
!pip install -q "transformers>=4.40" sentencepiece sacremoses
```

`sentencepiece` требуется токенизатору NLLB, `sacremoses` — предобработке MarianMT.

**Модели:**

* `Helsinki-NLP/opus-mt-en-ru` — специализированная EN→RU, ≈ 76 млн параметров,
  токенизатор `MarianTokenizer`, словарь 62 518 единиц;
* `facebook/nllb-200-distilled-600M` — многоязычная, 200 языков, ≈ 615 млн параметров,
  токенизатор `NllbTokenizer`, словарь 256 204 единицы.

Документация NLLB: <https://huggingface.co/docs/transformers/model_doc/nllb>

**Языковые коды NLLB:** `src_lang="eng_Latn"`, целевой язык задаётся при генерации через
`forced_bos_token_id=tokenizer.convert_tokens_to_ids("rus_Cyrl")`. Ошибка в коде приводит к
переводу не на тот язык: в эталонном прогоне без `forced_bos_token_id` модель выдала перевод
на испанский.

### Две особенности актуальных версий библиотек

1. **`attn_implementation="eager"` обязателен для блока с вниманием.** Начиная с
   `transformers` 5.x модели по умолчанию используют ускоренную реализацию внимания
   (SDPA/Flash), которая не сохраняет веса. При `output_attentions=True` поле
   `outputs.cross_attentions` оказывается **пустым кортежем**, и код падает с
   `IndexError: tuple index out of range`. Модель нужно загружать так:

```python
model = AutoModelForSeq2SeqLM.from_pretrained(name, attn_implementation="eager")
```

   Старые версии библиотеки такого аргумента не знают — оборачивайте вызов в
   `try / except (TypeError, ValueError)`.

2. **Предупреждение о `max_new_tokens` и `max_length`.** В `generation_config` моделей задан
   `max_length=512`, поэтому при каждом вызове `generate(max_new_tokens=...)` печатается
   предупреждение. Приоритет у `max_new_tokens`, результат корректен; чтобы не зашумлять
   вывод, в начале ноутбука вызывается `transformers.logging.set_verbosity_error()`.

---

## 4. Ход работы

| Шаг | Что делается | Блок ноутбука |
|---:|---|---|
| 1 | Определение устройства, фиксация зерна, структура каталогов | 1 |
| 2 | Подключение данных ЛР № 1 и проверка объёма корпуса | 2 |
| 3 | Загрузка MarianMT (`eval()`, `eager`) | 3 |
| 4 | Субсловная токенизация 100+ терминов глоссария | 5, 12 |
| 5 | Перевод одного сегмента, параметры декодирования | 4 |
| 6 | Батчинг: последовательный и пакетный инференс, замер времени | 10 |
| 7 | Условная кросс-энтропия для 5+ пар `source` / `reference` | 6, 13 |
| 8 | Cross-attention для 2+ пар, форма тензора, усреднение голов | 7, 13 |
| 9 | Загрузка NLLB, языковые коды, контрольный запуск без `forced_bos_token_id` | 3, 9 |
| 10 | Сравнение токенизаторов Marian и NLLB | 12 |
| 11 | Полный перевод модуля обеими моделями в JSONL | 11 |
| 12 | Проверка терминологии по глоссарию (glossary QA) | 14 |
| 13 | Сводная таблица `comparison.csv` и экспертный выбор | 14 |

### Ключевые формулы и понятия

Условная кросс-энтропия целевой строки при данном источнике:

```math
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log P\left(y_t \mid y_{1:t-1}, x\right)
```

Меньший loss означает, что строка более вероятна для данной модели. **Loss не является
метрикой качества перевода** и несопоставим между моделями с разными словарями.

Форма тензора cross-attention: `[batch, heads, target_len, source_len]`. Внимание —
**диагностический сигнал**, а не доказательство того, что конкретная голова «выучила правило
перевода».

О токенизации: не делайте механический вывод «Marian = BPE, NLLB = SentencePiece». Проверяйте
класс токенизатора, model card и config; в актуальной документации Hugging Face токенизатор
NLLB описан как основанный на **Unigram**.

---

## 5. Обязательный минимум

1. перевести **120+ одинаковых сегментов** с помощью MarianMT;
2. перевести те же **120+ сегментов** с помощью NLLB;
3. исследовать токенизацию **100+ терминов** из `glossary.csv` обоими токенизаторами;
4. вычислить cross-entropy минимум для **5 пар** `source` / `reference`;
5. исследовать cross-attention минимум для **2 пар**;
6. измерить последовательный и пакетный инференс (не менее двух размеров пачки);
7. выполнить glossary QA с ручным подтверждением нарушений;
8. выделить минимум **5 содержательных различий или ошибок** между моделями;
9. для каждого из пяти случаев аргументировать предпочтительный перевод;
10. построить не менее **4 графиков** и сохранить их в `figures/`;
11. сохранить результаты в воспроизводимом виде (`JSONL` / `CSV` + ноутбук + `requirements.txt`).

---

## 6. Таблица вариантов

Вариант совпадает с вариантом ЛР № 1; используются те же лингвистические ресурсы.
**Студенческих вариантов — 25** (номера 1–10 и 12–26). Вариант 11 — демонстрационное решение
преподавателя, студентам не назначается.

| Вариант | Тема модуля | Особое внимание | Контекстно неоднозначные единицы |
|---:|---|---|---|
| 1 | Variables and Data Types | variable, value, data type, assignment, expression, operator, type conversion | value, type, expression |
| 2 | Conditional Statements | condition, Boolean expression, branch, if statement, else clause, comparison operator, control flow | branch, condition, statement |
| 3 | Loops and Iteration | loop, iteration, for loop, while loop, range, break, continue, termination condition | `break` и `continue` в коде не переводятся; их значение в пояснительном тексте анализируется отдельно |
| 4 | Functions and Parameters | function, parameter, argument, return value, scope, default argument, function call | argument, scope, return |
| 5 | Lists, Tuples and Sequence Operations | list, tuple, sequence, index, slice, element, mutable, immutable, membership | slice, index, element |
| 6 | Dictionaries and Sets | dictionary, key, value, key-value pair, mapping, set, hash, lookup, intersection, union | key, mapping, set |
| 7 | File Input and Output | file, input, output, stream, file path, encoding, read mode, write mode, file handle, buffer | stream, mode, handle |
| 8 | Classes and Objects | class, object, instance, attribute, method, constructor, inheritance, encapsulation, composition | class, object, method |
| 9 | Exceptions and Error Handling | exception, error, raise, try block, except block, finally block, exception handler, traceback, runtime error | raise, handler, failure |
| 10 | Modules, Packages and Imports | module, package, import, namespace, library, dependency, standard library, virtual environment, entry point | package, module, library |
| **11** | **Algorithms and Computational Complexity** | демонстрационное решение преподавателя | **студентам не назначается** |
| 12 | Recursion and Backtracking | recursion, base case, recursive call, call stack, stack depth, backtracking, memoization, tail recursion | case, stack, depth |
| 13 | Sorting and Searching Algorithms | sorting, comparison, swap, stability, pivot, partition, merge, search key, linear search | key, order, merge |
| 14 | Strings and Text Processing | string, substring, character, encoding, concatenation, split, strip, formatting, whitespace | string, format, escape |
| 15 | Regular Expressions | pattern, match, group, quantifier, anchor, character class, greedy matching, backreference | match, group, class |
| 16 | Object-Oriented Design Principles | abstraction, encapsulation, interface, polymorphism, coupling, cohesion, refactoring, design pattern | interface, abstraction, contract |
| 17 | Testing and Debugging | unit test, test case, assertion, coverage, breakpoint, stack trace, regression, mock object | case, mock, coverage |
| 18 | Version Control with Git | repository, commit, branch, merge, conflict, pull request, staging area, rebase, remote | branch, merge, head |
| 19 | Databases and SQL Basics | table, row, column, primary key, foreign key, query, join, index, transaction | key, index, table |
| 20 | Web Fundamentals: HTTP and APIs | request, response, endpoint, status code, header, payload, REST, authentication, session | request, header, payload |
| 21 | Data Analysis with pandas | dataframe, series, index, column, aggregation, grouping, missing value, merge, pivot | index, series, merge |
| 22 | Numerical Computing and Arrays | array, shape, axis, broadcasting, vectorization, dtype, slicing, matrix product | axis, shape, array |
| 23 | Machine Learning Basics | feature, label, training set, model, overfitting, validation, hyperparameter, accuracy | feature, model, label |
| 24 | Neural Networks and Deep Learning | neuron, layer, weight, activation function, gradient, backpropagation, epoch, batch | layer, weight, batch |
| 25 | Computer Networks Fundamentals | packet, protocol, router, IP address, port, latency, bandwidth, handshake, gateway | port, packet, host |
| 26 | Operating Systems and Processes | process, thread, scheduler, memory allocation, deadlock, system call, kernel, virtual memory | process, thread, kernel |

Вариант 11 продолжает тестовый вариант 11 ЛР № 1 и служит образцом ожидаемой структуры решения
(`lab_02_teacher.ipynb`). Объём его данных (14 сегментов, 19 терминов) намеренно меньше
студенческого минимума: эталон демонстрирует методику, а не объём.

---

## 7. Результаты работы

После выполнения ноутбука в Colab создаётся каталог `/content/lab02`:

```text
/content/lab02/
├── data/
│   ├── segments_en.csv
│   ├── glossary.csv
│   └── aligned_reference.csv
├── results/
│   ├── results_marian.jsonl
│   ├── results_nllb.jsonl
│   ├── comparison.csv
│   ├── tokenization_10_terms.csv     ← таблица по всем терминам глоссария
│   ├── loss_5_pairs.csv
│   ├── attention_marian_<segment_id>.csv
│   ├── attention_summary.csv
│   ├── timing.csv
│   └── glossary_qa.csv
├── figures/
│   ├── timing.png
│   ├── tokenization_comparison.png
│   ├── loss.png
│   ├── cross_attention.png
│   └── glossary_qa.png
├── report.md
├── requirements.txt
└── lab02_results.zip                 ← архив для скачивания
```

Поля `preferred` и `comment` в `comparison.csv`, а также `confirmed` и `comment` в
`glossary_qa.csv` заполняются **вручную** после экспертного анализа. Это принципиальная часть
работы, и автоматический выбор модели её не заменяет.

Для сдачи в GitHub-репозиторий содержимое архива раскладывается в каталог `lab02/`, туда же
добавляется сам ноутбук `LAB_02.ipynb`.

### Содержание `report.md`

1. среда и устройство (включая тип GPU в Colab и версии `torch` / `transformers`);
2. model id обеих моделей, классы токенизаторов, размеры словарей;
3. таблица токенизации и график сравнения дробления терминов;
4. 120+ пар переводов (файлы JSONL) и примеры расхождений;
5. loss для 5 примеров и график;
6. форма тензора и анализ attention с тепловой картой;
7. время инференса в двух режимах и график;
8. 5+ ошибок или содержательных различий с аргументацией;
9. glossary compliance с указанием подтверждённых нарушений и false positive;
10. вывод.

---

## 8. Критериальный лист (20 баллов)

| Критерий | Состав критерия | Баллы |
|---|---|---:|
| **Работоспособность пайплайна/кода** | MarianMT — 2 · NLLB — 2 · batching/loss/attention — 2 | **6** |
| **Корректность лингвистической обработки и валидации** | языковые коды и токенизация — 2 · объём корпуса (120+ сегментов, 100+ терминов) — 1 · glossary QA — 2 | **5** |
| **Анализ ошибок и аргументация решений** | 5+ различий моделей — 2 · корректная интерпретация loss/attention — 2 · вывод — 1 | **5** |
| **Оформление, воспроизводимость** | результаты и 4+ графика — 1 · README — 1 · зависимости — 1 · воспроизводимая структура — 1 | **4** |
| **Итого** |  | **20** |

**Снижения:**

* объём корпуса меньше 120 сегментов или 100 терминов — −2 балла;
* колонки `preferred` / `comment` не заполнены или заполнены автоматически — −2 балла;
* использованы демонстрационные данные вместо собственного варианта — −3 балла;
* ноутбук не выполняется целиком (`Runtime → Run all`) — −3 балла;
* графики отсутствуют или их меньше четырёх — −1 балл;
* вывод ячеек не сохранён — −1 балл.

---

## 9. Типичные ошибки

| Ошибка | Следствие | Как избежать |
|---|---|---|
| Модель загружена без `attn_implementation="eager"` | `outputs.cross_attentions` пуст, `IndexError: tuple index out of range` | передавать аргумент при загрузке и проверять `cross_attentions` перед индексированием |
| Не задан `forced_bos_token_id` для NLLB | перевод не на тот язык (в эталонном прогоне — на испанский) | получить код через `convert_tokens_to_ids("rus_Cyrl")` |
| Модель не переведена в `eval()` | нестабильный результат между запусками | вызвать `model.eval()` сразу после загрузки |
| Замер времени без прогрева | завышенное время первого режима | сделать один холостой вызов до замеров |
| `add_special_tokens=True` при анализе токенизации | в разборе появляются `</s>` и код языка | явно передать `add_special_tokens=False` |
| Loss интерпретируется как качество | неверные выводы | loss — вероятность строки; качество измеряется в ЛР № 4 |
| Сравнение loss между Marian и NLLB | некорректный вывод | словари различаются вчетверо, значения несопоставимы |
| Знаменатель glossary QA — все пары «сегмент × термин» | завышенная доля соблюдения | считать только фактические вхождения терминов в исходный текст |
| Строковый glossary QA без ручной проверки | ложные срабатывания из-за русской морфологии | подтверждать каждое нарушение в колонке `confirmed` |
| Все 100+ терминов на одном графике | нечитаемая диаграмма | отбирать 20–30 показательных терминов |
| Работа закрыта до скачивания архива | результаты потеряны вместе со средой | скачать `lab02_results.zip` или хранить файлы на Drive |

---

## 10. Состав методических материалов

| Файл | Назначение |
|---|---|
| `readme_lab_02.md` | настоящее описание работы |
| `lab_02.ipynb` | студенческая заготовка для Colab: шаблоны функций, автотесты, блоки для самостоятельного выполнения и построения графиков |
| `lab_02_teacher.ipynb` | эталонное решение варианта 11: полный пайплайн, пояснения к каждой ячейке, семь графиков и генерация всех результирующих файлов |
