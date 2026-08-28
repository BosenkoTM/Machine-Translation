# Лабораторный практикум по дисциплине «Основы машинного перевода»

## Лабораторная работа 2. «Нейросетевой перевод на базе Hugging Face Transformers (MarianMT, NLLB)»

**Разделы курса:** 2 и 4  
**Максимум:** 20 баллов  
**Сквозной кейс:** перевод того же обучающего модуля по информатике с помощью двух открытых NMT-моделей.

### Цель и формируемые компетенции (ОПК-2, ОПК-9, УК-1, УК-6)

**Цель** — развернуть локальный NMT-пайплайн, исследовать субсловную токенизацию, батчинг, условную кросс-энтропию и механизм внимания, а также сравнить специализированную EN→RU модель MarianMT с многоязычной NLLB.

- **ОПК-2:** применение NMT при создании локализованных образовательных материалов;
- **ОПК-9:** программное использование NLP-моделей и библиотек;
- **УК-1:** анализ поведения двух моделей и причин переводческих ошибок;
- **УК-6:** организация воспроизводимого эксперимента с фиксированными данными и версиями.

### Входные данные и подготовительные требования

Используйте результаты ЛР-1:

```text
data/source_en.md
data/segments_en.csv
data/glossary.csv
data/reference_ru.md
```

`reference_ru.md` — подтвержденный человеком перевод. Он нужен для вычисления loss и последующего оценивания.

В ЛР № 2 студент продолжает **тот же индивидуальный вариант, который выполнял в ЛР № 1**.

| Вариант | Тема исходного учебного модуля |
|---:|---|
| 1 | Variables and Data Types |
| 2 | Conditional Statements |
| 3 | Loops and Iteration |
| 4 | Functions and Parameters |
| 5 | Lists, Tuples and Sequence Operations |
| 6 | Dictionaries and Sets |
| 7 | File Input and Output |
| 8 | Classes and Objects |
| 9 | Exceptions and Error Handling |
| 10 | Modules, Packages and Imports |

**Вариант 11 — `Algorithms and Computational Complexity`** используется только как демонстрационное тестовое решение преподавателя и студентам как индивидуальное задание не назначается.

Требования:

- Python 3.10+;
- 8 ГБ RAM минимум;
- интернет для первичной загрузки моделей;
- GPU желателен, но MarianMT работает и на CPU;
- NLLB-600M на CPU может работать заметно медленнее.

Установка:

```bash
pip install torch transformers sentencepiece pandas numpy
```

Модели:

- `Helsinki-NLP/opus-mt-en-ru`;
- `facebook/nllb-200-distilled-600M`.

Документация NLLB:

<https://huggingface.co/docs/transformers/model_doc/nllb>

### Пошаговое руководство к выполнению (с примерами кода на Python 3.10+ и инструкциями для Smartcat/API)

#### Шаг 1. Вычислительное устройство

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
```

#### Шаг 2. Загрузка MarianMT

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MARIAN_NAME = "Helsinki-NLP/opus-mt-en-ru"

marian_tokenizer = AutoTokenizer.from_pretrained(MARIAN_NAME)
marian_model = AutoModelForSeq2SeqLM.from_pretrained(
    MARIAN_NAME
).to(device)

marian_model.eval()
```

#### Шаг 3. Субсловная токенизация

```python
samples = [
    "machine translation",
    "tokenization",
    "conditional statement",
    "debugging",
    "reusability",
]

for text in samples:
    ids = marian_tokenizer(
        text,
        add_special_tokens=False
    )["input_ids"]
    tokens = marian_tokenizer.convert_ids_to_tokens(ids)
    print(text, "=>", tokens)
```

В отчете объясните:

1. где границы токенов совпадают со словами;
2. где используются субсловные части;
3. почему open vocabulary полезен для технических терминов;
4. почему субсловный токен не следует считать морфемой.

#### Шаг 4. Перевод одного сегмента MarianMT

```python
text = "A loop repeats a block of instructions."

inputs = marian_tokenizer(
    text,
    return_tensors="pt",
    truncation=True
).to(device)

with torch.no_grad():
    generated = marian_model.generate(
        **inputs,
        num_beams=4,
        max_new_tokens=80
    )

translation = marian_tokenizer.decode(
    generated[0],
    skip_special_tokens=True
)

print(translation)
```

#### Шаг 5. Батчинг

```python
segments = [
    "A variable stores a value.",
    "A condition controls which branch is executed.",
    "A loop repeats a block of instructions.",
    "A function can return a value.",
]

batch = marian_tokenizer(
    segments,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(device)

with torch.no_grad():
    generated = marian_model.generate(
        **batch,
        num_beams=4,
        max_new_tokens=96
    )

translations = marian_tokenizer.batch_decode(
    generated,
    skip_special_tokens=True
)

for src, tgt in zip(segments, translations):
    print("EN:", src)
    print("RU:", tgt)
    print()
```

Измерьте время:

```python
from time import perf_counter

start = perf_counter()

with torch.no_grad():
    _ = marian_model.generate(
        **batch,
        num_beams=4,
        max_new_tokens=96
    )

elapsed = perf_counter() - start
print(f"{elapsed:.3f} s")
```

Сравните последовательный и пакетный инференс.

#### Шаг 6. Кросс-энтропия

Для пары source/reference:

```python
source = "A variable stores a value."
target = "Переменная хранит значение."

batch = marian_tokenizer(
    source,
    text_target=target,
    return_tensors="pt",
    truncation=True
).to(device)

with torch.no_grad():
    outputs = marian_model(**batch)

print("cross_entropy =", float(outputs.loss))
```

```math
\mathcal{L}
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log P\left(y_t \mid y_{1:t-1}, x\right).
```

Меньший loss означает, что reference более вероятен для данной модели, но loss **не является полноценной метрикой переводческого качества**.

#### Шаг 7. Cross-attention

```python
source = "A function returns a value."
target = "Функция возвращает значение."

batch = marian_tokenizer(
    source,
    text_target=target,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = marian_model(
        **batch,
        output_attentions=True,
        return_dict=True
    )

last_layer = outputs.cross_attentions[-1]
print(last_layer.shape)
# [batch, heads, target_length, source_length]
```

Усреднение голов:

```python
attention = last_layer[0].mean(dim=0).cpu()
print(attention.shape)
```

Токены:

```python
src_tokens = marian_tokenizer.convert_ids_to_tokens(
    batch["input_ids"][0].cpu().tolist()
)

labels = batch["labels"][0].cpu().tolist()
labels = [x for x in labels if x != -100]
tgt_tokens = marian_tokenizer.convert_ids_to_tokens(labels)

print(src_tokens)
print(tgt_tokens)
```

> Attention — диагностический сигнал, но не доказательство того, что конкретная голова «выучила правило перевода».

#### Шаг 8. Загрузка NLLB

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NLLB_NAME = "facebook/nllb-200-distilled-600M"

nllb_tokenizer = AutoTokenizer.from_pretrained(
    NLLB_NAME,
    src_lang="eng_Latn",
    tgt_lang="rus_Cyrl"
)

nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
    NLLB_NAME
).to(device)

nllb_model.eval()
```

#### Шаг 9. Перевод NLLB

```python
text = "A function can accept several parameters."

inputs = nllb_tokenizer(
    text,
    return_tensors="pt",
    truncation=True
).to(device)

rus_id = nllb_tokenizer.convert_tokens_to_ids("rus_Cyrl")

with torch.no_grad():
    generated = nllb_model.generate(
        **inputs,
        forced_bos_token_id=rus_id,
        num_beams=4,
        max_new_tokens=96
    )

translation = nllb_tokenizer.decode(
    generated[0],
    skip_special_tokens=True
)

print(translation)
```

NLLB использует языковые коды. Неверные `eng_Latn` / `rus_Cyrl` могут привести к некорректной маршрутизации языка.

#### Шаг 10. Сравнение токенизаторов

```python
term = "conditional statement"

for name, tokenizer in [
    ("Marian", marian_tokenizer),
    ("NLLB", nllb_tokenizer),
]:
    ids = tokenizer(term, add_special_tokens=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(name, tokens)
```

Не делайте механический вывод «Marian = BPE, NLLB = SentencePiece». Проверяйте конкретный tokenizer, model card и config. В актуальной документации Hugging Face NLLB tokenizer описывается как основанный на **Unigram**.

#### Шаг 11. Полный перевод файла в JSONL

```python
from pathlib import Path
import json

source_lines = [
    line.strip()
    for line in Path("data/source_en.md").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

records = []

for idx, source in enumerate(source_lines, 1):
    inputs = marian_tokenizer(
        source,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        output = marian_model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=128
        )

    target = marian_tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    records.append({
        "id": f"s{idx:03d}",
        "source": source,
        "translation": target,
        "model": MARIAN_NAME,
    })

with open(
    "lab02/results_marian.jsonl",
    "w",
    encoding="utf-8"
) as f:
    for row in records:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

Повторите для NLLB.

#### Шаг 12. Проверка терминологии из ЛР-1

```python
import pandas as pd

glossary = pd.read_csv("data/glossary.csv")

def term_compliance(source, translation, glossary):
    violations = []

    source_l = source.lower()
    target_l = translation.lower()

    for row in glossary.itertuples(index=False):
        en = str(row.en).lower()
        ru = str(row.ru).lower()

        if en in source_l and ru not in target_l:
            violations.append({
                "source_term": row.en,
                "preferred_ru": row.ru,
            })

    return violations
```

Из-за русской морфологии строковый QA может давать false positive; каждое нарушение проверяется вручную.

#### Шаг 13. Сводная таблица

Подготовьте:

| segment_id | source | MarianMT | NLLB | preferred |
|---|---|---|---|---|
| s001 | ... | ... | ... | Marian/NLLB/оба/ни один |

Для каждого выбора добавьте краткое обоснование.

### Задания для самостоятельного выполнения (10 индивидуальных вариантов)

Студент использует **тот же вариант и те же лингвистические ресурсы, что и в ЛР № 1**.

Для любого варианта обязательно:

1. перевести **12+ одинаковых сегментов** с помощью MarianMT;
2. перевести те же **12+ сегментов** с помощью NLLB;
3. исследовать токенизацию минимум **10 терминов** из `glossary.csv`;
4. вычислить cross-entropy минимум для **5 source/reference пар**;
5. исследовать cross-attention минимум для **2 пар**;
6. измерить последовательный и пакетный инференс;
7. выполнить glossary QA;
8. выделить минимум **5 содержательных различий или ошибок** между моделями;
9. для каждого из пяти случаев аргументировать предпочтительный перевод;
10. сохранить результаты в воспроизводимом виде (`JSONL`/`CSV`, notebook или скрипты).

#### Вариант 1. Variables and Data Types

Особое внимание: `variable`, `value`, `data type`, `assignment`, `expression`, `operator`, `type conversion`.

Контекстно неоднозначные единицы: `value`, `type`, `expression`.

#### Вариант 2. Conditional Statements

Особое внимание: `condition`, `Boolean expression`, `branch`, `if statement`, `else clause`, `comparison operator`, `control flow`.

Контекстно неоднозначные единицы: `branch`, `condition`, `statement`.

#### Вариант 3. Loops and Iteration

Особое внимание: `loop`, `iteration`, `for loop`, `while loop`, `range`, `break`, `continue`, `termination condition`.

Python keywords `break` и `continue` в коде не переводятся; их значение в пояснительном тексте анализируется отдельно.

#### Вариант 4. Functions and Parameters

Особое внимание: `function`, `parameter`, `argument`, `return value`, `scope`, `default argument`, `function call`.

Контекстно неоднозначные единицы: `argument`, `scope`, `return`.

#### Вариант 5. Lists, Tuples and Sequence Operations

Особое внимание: `list`, `tuple`, `sequence`, `index`, `slice`, `element`, `mutable`, `immutable`, `membership`.

Контекстно неоднозначные единицы: `slice`, `index`, `element`.

#### Вариант 6. Dictionaries and Sets

Особое внимание: `dictionary`, `key`, `value`, `key-value pair`, `mapping`, `set`, `hash`, `lookup`, `intersection`, `union`.

Контекстно неоднозначные единицы: `key`, `mapping`, `set`.

#### Вариант 7. File Input and Output

Особое внимание: `file`, `input`, `output`, `stream`, `file path`, `encoding`, `read mode`, `write mode`, `file handle`, `buffer`.

Контекстно неоднозначные единицы: `stream`, `mode`, `handle`.

#### Вариант 8. Classes and Objects

Особое внимание: `class`, `object`, `instance`, `attribute`, `method`, `constructor`, `inheritance`, `encapsulation`, `composition`.

Контекстно неоднозначные единицы: `class`, `object`, `method`.

#### Вариант 9. Exceptions and Error Handling

Особое внимание: `exception`, `error`, `raise`, `try block`, `except block`, `finally block`, `exception handler`, `traceback`, `runtime error`.

Контекстно неоднозначные единицы: `raise`, `handler`, `failure`.

#### Вариант 10. Modules, Packages and Imports

Особое внимание: `module`, `package`, `import`, `namespace`, `library`, `dependency`, `standard library`, `virtual environment`, `entry point`.

Контекстно неоднозначные единицы: `package`, `module`, `library`.

### Вариант 11. Тестовое решение преподавателя

**Тема:** `Algorithms and Computational Complexity`.

Вариант 11 продолжает тестовый вариант 11 ЛР № 1 и использует те же исходные данные. Он предназначен для демонстрации ожидаемой структуры решения и **не назначается студентам**.

Комплект:

- [`variant_11_solution/README.md`](variant_11_solution/README.md) — инструкция по запуску;
- [`variant_11_solution/LAB_02_VARIANT_11_SOLUTION.ipynb`](variant_11_solution/LAB_02_VARIANT_11_SOLUTION.ipynb) — готовый notebook;
- [`variant_11_solution/report.md`](variant_11_solution/report.md) — шаблон-эталон отчета;
- [`variant_11_solution/translate_nmt.py`](variant_11_solution/translate_nmt.py) — перевод двумя моделями;
- [`variant_11_solution/inspect_tokenizer.py`](variant_11_solution/inspect_tokenizer.py) — анализ токенизации;
- [`variant_11_solution/inspect_attention.py`](variant_11_solution/inspect_attention.py) — извлечение cross-attention;
- [`variant_11_solution/requirements.txt`](variant_11_solution/requirements.txt) — зависимости;
- [`variant_11_solution/data/source_en.md`](variant_11_solution/data/source_en.md) — исходный англоязычный модуль;
- [`variant_11_solution/data/segments_en.csv`](variant_11_solution/data/segments_en.csv) — сегменты из ЛР № 1;
- [`variant_11_solution/data/glossary.csv`](variant_11_solution/data/glossary.csv) — утвержденная терминология;
- [`variant_11_solution/data/reference_ru.md`](variant_11_solution/data/reference_ru.md) — эталонный перевод;
- [`variant_11_solution/data/aligned_reference.csv`](variant_11_solution/data/aligned_reference.csv) — выровненные EN→RU сегменты.

После выполнения notebook автоматически создаются:

```text
results/
├── results_marian.jsonl
├── results_nllb.jsonl
├── comparison.csv
├── tokenization_10_terms.csv
├── loss_5_pairs.csv
├── attention_marian_s002.csv
├── attention_marian_s005.csv
├── attention_summary.csv
├── timing.csv
└── glossary_qa.csv
```

Поля `preferred` и `comment` в `comparison.csv` заполняются вручную после экспертного анализа. Это принципиальная часть лабораторной работы и не должна заменяться автоматическим выбором модели.

### Требования к оформлению отчета в GitHub-репозитории

```text
lab02/
├── LAB_02.ipynb
├── translate_nmt.py
├── inspect_tokenizer.py
├── inspect_attention.py
├── results/
│   ├── results_marian.jsonl
│   ├── results_nllb.jsonl
│   ├── comparison.csv
│   ├── tokenization_10_terms.csv
│   ├── loss_5_pairs.csv
│   ├── attention_summary.csv
│   ├── timing.csv
│   └── glossary_qa.csv
├── report.md
└── requirements.txt
```

`report.md` должен включать:

1. среду и устройство;
2. model id обеих моделей;
3. таблицу токенизации;
4. 12+ пар переводов;
5. loss для 5 примеров;
6. shape/анализ attention;
7. время инференса;
8. 5+ ошибок или содержательных различий;
9. glossary compliance;
10. вывод.

### Детализированный критериальный лист оценки (Рубрикатор на 20 баллов):

| Критерий | Состав критерия | Баллы |
|---|---|---:|
| **Работоспособность пайплайна/кода** | MarianMT — 2; NLLB — 2; batching/loss/attention — 2 | **6** |
| **Корректность лингвистической обработки и валидации** | Языковые коды и токенизация — 2; 12+ сегментов — 1; glossary QA — 2 | **5** |
| **Анализ ошибок и аргументация решений** | 5+ различий моделей — 2; корректная интерпретация loss/attention — 2; вывод — 1 | **5** |
| **Оформление репозитория, reproducibility** | Скрипты/результаты — 1; README — 1; зависимости — 1; воспроизводимая структура — 1 | **4** |
| **Итого** |  | **20** |
