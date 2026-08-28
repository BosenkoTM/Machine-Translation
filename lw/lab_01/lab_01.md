# Лабораторный практикум по дисциплине «Основы машинного перевода»

## Лабораторная работа 1. «Компьютерная лексикография, терминологические базы (TBX) и Translation Memory (TMX) в Smartcat»

**Разделы курса:** 1 и 3  
**Максимум:** 20 баллов  
**Сквозной кейс:** подготовка лингвистических ресурсов для локализации англоязычного обучающего модуля по информатике.

### Цель и формируемые компетенции (ОПК-2, ОПК-9, УК-1, УК-6)

**Цель работы** — научиться представлять переводческие знания в машиночитаемом виде: выделять термины, нормализовать терминологию, формировать терминологическую базу TBX, создавать память переводов TMX и подключать лингвистические ресурсы к CAT-конвейеру.

- **ОПК-2** — применение современных образовательных и цифровых технологий при подготовке учебных материалов;
- **ОПК-9** — использование ИКТ и цифровых инструментов профессиональной деятельности;
- **УК-1** — критический анализ источников, терминологических решений и неоднозначных единиц;
- **УК-6** — организация собственной деятельности, версионирование ресурсов и воспроизводимость.

После выполнения работы студент должен уметь различать termbase и TM, выделять термины, формировать TBX/TMX и проверять терминологическую согласованность CAT-проекта.

### Входные данные и подготовительные требования

Используйте один из **10 индивидуальных вариантов** обучающего модуля.  
**Вариант 11** является тестовым демонстрационным решением преподавателя и студентам как индивидуальное задание не назначается.

| Вариант | Тема локализуемого модуля |
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

Для проверки методики и воспроизводимости в комплект лабораторной работы включен **вариант 11 — Algorithms and Computational Complexity** с полным набором входных данных и готовым notebook-решением.


Минимальный объем исходного текста — **12 сегментов и не менее 180 слов**. В нем должны быть 12–20 предметных терминов, минимум 3 многозначных единицы, 2 фрагмента кода/inline-code, заголовок, список и методическая инструкция.

Пример `data/source_en.md`:

```markdown
# Variables and Data Types

A variable stores a value that can be reused by a program.

In Python, the expression `x = 10` binds the name `x` to an integer object.

A data type determines which operations can be applied to a value.

Students should predict the output before running the code.
```

Подготовка среды:

```bash
python -m venv .venv
pip install pandas lxml scikit-learn openpyxl
```

> **Важно.** TBX — стандарт обмена терминологическими данными ISO 30042. В Smartcat на август 2026 г. прямой импорт TBX не заявлен: glossary импортируется из XLSX/MultiTerm XML, а TM — из TMX/SDLTM/XLSX. Поэтому в работе студент создает и валидирует TBX, затем преобразует его в XLSX для Smartcat; TMX импортируется напрямую.

Официальные материалы:

- <https://www.tbxinfo.net/>
- <https://www.tbxinfo.net/developer-resources/tbx-elements-reference/>
- <https://help.smartcat.com/creating-and-managing-a-glossary-in-smartcat/>
- <https://help.smartcat.com/translation-memory-create-import-export-and-manage/>

### Пошаговое руководство к выполнению (с примерами кода на Python 3.10+ и инструкциями для Smartcat/API)

#### Шаг 1. Сегментация

Для Markdown не применяйте `split(".")`; для лабораторной используйте сегментацию по непустым строкам:

```python
from pathlib import Path

text = Path("data/source_en.md").read_text(encoding="utf-8")
segments = [
    line.strip()
    for line in text.splitlines()
    if line.strip() and not line.strip().startswith("```")
]

for i, segment in enumerate(segments, 1):
    print(f"{i:02d}: {segment}")
```

```python
import pandas as pd

df = pd.DataFrame({
    "segment_id": [f"s{i:03d}" for i in range(1, len(segments) + 1)],
    "source_en": segments,
})
df.to_csv("data/segments_en.csv", index=False, encoding="utf-8")
```

#### Шаг 2. Извлечение кандидатов в термины

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 3),
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b",
)

matrix = vectorizer.fit_transform(df["source_en"])
scores = matrix.mean(axis=0).A1
terms = vectorizer.get_feature_names_out()

candidates = (
    pd.DataFrame({"term": terms, "score": scores})
      .sort_values("score", ascending=False)
      .head(30)
)
print(candidates)
```

TF-IDF дает **кандидатов**, а не готовую терминологию. Студент вручную отбирает 15–20 единиц.

#### Шаг 3. Рабочая таблица терминов

```csv
concept_id,en,ru,part_of_speech,status,definition,context
c001,variable,переменная,noun,preferred,"Named storage/reference used by a program","A variable stores a value."
c002,data type,тип данных,noun,preferred,"Category that determines operations","A data type determines..."
c003,assignment,присваивание,noun,preferred,"Assigning a value","The assignment statement..."
```

#### Шаг 4. Формирование TBX

TBX 2019 использует `<tbx>` и namespace `urn:iso:std:iso:30042:ed-2`.

```python
import pandas as pd
from lxml import etree

NS = "urn:iso:std:iso:30042:ed-2"
XML = "http://www.w3.org/XML/1998/namespace"
TBX = f"{{{NS}}}"

glossary = pd.read_csv("data/glossary.csv")

root = etree.Element(
    TBX + "tbx",
    nsmap={None: NS},
    style="dca",
    type="TBX-Basic",
)
root.set(f"{{{XML}}}lang", "en")

header = etree.SubElement(root, TBX + "tbxHeader")
file_desc = etree.SubElement(header, TBX + "fileDesc")
source_desc = etree.SubElement(file_desc, TBX + "sourceDesc")
etree.SubElement(source_desc, TBX + "p").text = "MGPU MT laboratory"

text_el = etree.SubElement(root, TBX + "text")
body = etree.SubElement(text_el, TBX + "body")

for row in glossary.itertuples(index=False):
    concept = etree.SubElement(body, TBX + "conceptEntry", id=str(row.concept_id))
    etree.SubElement(concept, TBX + "descrip", type="definition").text = str(row.definition)

    for lang, value in [("en", row.en), ("ru", row.ru)]:
        lang_sec = etree.SubElement(concept, TBX + "langSec")
        lang_sec.set(f"{{{XML}}}lang", lang)
        term_sec = etree.SubElement(lang_sec, TBX + "termSec")
        etree.SubElement(term_sec, TBX + "term").text = str(value)
        etree.SubElement(term_sec, TBX + "termNote", type="partOfSpeech").text = str(row.part_of_speech)

etree.ElementTree(root).write(
    "data/termbase.tbx",
    pretty_print=True,
    xml_declaration=True,
    encoding="UTF-8",
)
```

#### Шаг 5. Проверка TBX

```python
from lxml import etree

doc = etree.parse("data/termbase.tbx")
assert doc.getroot().tag.endswith("tbx")
assert len(doc.xpath("//*[local-name()='conceptEntry']")) >= 15
assert len(doc.xpath("//*[local-name()='term']")) >= 30
print("Basic TBX checks passed")
```

Для полной проверки структуры: <https://www.tbxinfo.net/validating-a-tbx-file/>.

#### Шаг 6. TBX → XLSX для Smartcat

```python
from lxml import etree
import pandas as pd

doc = etree.parse("data/termbase.tbx")
rows = []

for entry in doc.xpath("//*[local-name()='conceptEntry']"):
    item = {"concept_id": entry.get("id"), "English": None, "Russian": None}
    for lang_sec in entry.xpath("./*[local-name()='langSec']"):
        lang = lang_sec.get("{http://www.w3.org/XML/1998/namespace}lang")
        terms = lang_sec.xpath(".//*[local-name()='term']/text()")
        if terms and lang.startswith("en"):
            item["English"] = terms[0]
        elif terms and lang.startswith("ru"):
            item["Russian"] = terms[0]
    rows.append(item)

pd.DataFrame(rows).to_excel("data/glossary_smartcat.xlsx", index=False)
```

#### Шаг 7. Создание TMX

```python
from lxml import etree

translation_pairs = [
    (
        "A variable stores a value that can be reused by a program.",
        "Переменная хранит значение, которое программа может использовать повторно."
    ),
    (
        "A data type determines which operations can be applied to a value.",
        "Тип данных определяет, какие операции можно применять к значению."
    ),
]

root = etree.Element("tmx", version="1.4")
etree.SubElement(
    root, "header",
    creationtool="MGPU-lab",
    creationtoolversion="1.0",
    segtype="sentence",
    adminlang="en",
    srclang="en",
    datatype="PlainText",
)
body = etree.SubElement(root, "body")

for source, target in translation_pairs:
    tu = etree.SubElement(body, "tu")
    en = etree.SubElement(tu, "tuv", {"{http://www.w3.org/XML/1998/namespace}lang": "en"})
    etree.SubElement(en, "seg").text = source
    ru = etree.SubElement(tu, "tuv", {"{http://www.w3.org/XML/1998/namespace}lang": "ru"})
    etree.SubElement(ru, "seg").text = target

etree.ElementTree(root).write(
    "data/memory.tmx",
    pretty_print=True,
    xml_declaration=True,
    encoding="UTF-8",
)
```

В итоговом `memory.tmx` должно быть **не менее 8** пар сегментов.

#### Шаг 8. Smartcat

**Glossary:**

1. Workspace → **Intelligence Fabric → Terminology**.
2. Создайте glossary.
3. Загрузите `glossary_smartcat.xlsx`.
4. Сопоставьте English/Russian.
5. Подключите glossary через **Linguistic Assets**.

**TM:**

1. **Intelligence Fabric → Reviewed Translations**.
2. Откройте Translation memories.
3. Создайте EN→RU TM.
4. Импортируйте `memory.tmx`.
5. Подключите TM к проекту.

В CAT Editor зафиксируйте минимум 5 случаев: glossary match, exact/fuzzy TM match, альтернативный AI/MT-вариант и финальное решение студента.

#### Шаг 9. Терминологический QA

```python
import pandas as pd

glossary = pd.read_csv("data/glossary.csv")
translated = open("data/reference_ru.md", encoding="utf-8").read().lower()

for row in glossary.itertuples(index=False):
    if str(row.ru).lower() not in translated:
        print("Check:", row.en, "=>", row.ru)
```

Строковая проверка может ошибаться из-за русской морфологии; результаты нужно верифицировать вручную.

### Задания для самостоятельного выполнения (10 индивидуальных вариантов)

Все варианты выполняются по единому регламенту:

- исходный текст — **12+ сегментов и 180+ английских слов**;
- 15–20 утвержденных терминов;
- не менее 3 многозначных или контекстно зависимых единиц;
- TBX;
- XLSX glossary для Smartcat;
- TMX с **8+** парами сегментов;
- 5 CAT-наблюдений;
- терминологическая QA;
- отчет в GitHub.

#### Вариант 1. Variables and Data Types

Базовые кандидаты: `variable`, `value`, `data type`, `integer`, `floating-point number`, `string`, `Boolean`, `assignment`, `expression`, `operator`, `literal`, `type conversion`, `dynamic typing`, `identifier`, `constant`.

Контекстно неоднозначные единицы для анализа: `value`, `type`, `expression`.

#### Вариант 2. Conditional Statements

Базовые кандидаты: `condition`, `Boolean expression`, `branch`, `if statement`, `else clause`, `comparison operator`, `nested condition`, `control flow`, `truth value`, `indentation`, `logical operator`, `predicate`, `decision`, `code block`, `execution path`.

Контекстно неоднозначные единицы: `branch`, `condition`, `statement`.

#### Вариант 3. Loops and Iteration

Базовые кандидаты: `loop`, `iteration`, `for loop`, `while loop`, `counter`, `range`, `break`, `continue`, `loop body`, `termination condition`, `iterator`, `sequence`, `nested loop`, `infinite loop`, `iteration variable`.

Контекстно неоднозначные единицы: `range`, `break`, `continue`.

#### Вариант 4. Functions and Parameters

Базовые кандидаты: `function`, `parameter`, `argument`, `return value`, `function call`, `scope`, `local variable`, `default argument`, `docstring`, `recursion`, `signature`, `keyword argument`, `positional argument`, `return statement`, `call stack`.

Контекстно неоднозначные единицы: `argument`, `scope`, `return`.

#### Вариант 5. Lists, Tuples and Sequence Operations

Базовые кандидаты: `list`, `tuple`, `sequence`, `index`, `slice`, `element`, `mutable`, `immutable`, `append`, `length`, `membership`, `concatenation`, `iteration`, `negative index`, `nested sequence`.

Контекстно неоднозначные единицы: `slice`, `index`, `element`.

#### Вариант 6. Dictionaries and Sets

Базовые кандидаты: `dictionary`, `key`, `value`, `key-value pair`, `mapping`, `set`, `unique element`, `membership test`, `hash`, `lookup`, `update`, `intersection`, `union`, `difference`, `iteration`.

Контекстно неоднозначные единицы: `key`, `mapping`, `set`.

#### Вариант 7. File Input and Output

Базовые кандидаты: `file`, `input`, `output`, `stream`, `file path`, `encoding`, `text file`, `binary file`, `read mode`, `write mode`, `append mode`, `file handle`, `context manager`, `buffer`, `end of file`.

Контекстно неоднозначные единицы: `stream`, `mode`, `handle`.

#### Вариант 8. Classes and Objects

Базовые кандидаты: `class`, `object`, `instance`, `attribute`, `method`, `constructor`, `inheritance`, `encapsulation`, `instance variable`, `class variable`, `self`, `interface`, `composition`, `base class`, `derived class`.

Контекстно неоднозначные единицы: `class`, `object`, `method`.

#### Вариант 9. Exceptions and Error Handling

Базовые кандидаты: `exception`, `error`, `raise`, `try block`, `except block`, `finally block`, `exception handler`, `traceback`, `runtime error`, `validation`, `recovery`, `exception type`, `assertion`, `resource cleanup`, `failure`.

Контекстно неоднозначные единицы: `raise`, `handler`, `failure`.

#### Вариант 10. Modules, Packages and Imports

Базовые кандидаты: `module`, `package`, `import`, `namespace`, `library`, `dependency`, `standard library`, `third-party package`, `module path`, `alias`, `package manager`, `virtual environment`, `version`, `installation`, `entry point`.

Контекстно неоднозначные единицы: `package`, `module`, `library`.

### Вариант 11. Тестовое решение преподавателя

**Тема:** `Algorithms and Computational Complexity`.

Вариант 11 предназначен для демонстрации ожидаемой структуры решения и **не используется как индивидуальный вариант**.

Комплект:

- [`variant_11_solution/README.md`](variant_11_solution/README.md) — инструкция по запуску;
- [`variant_11_solution/LAB_01_VARIANT_11_SOLUTION.ipynb`](variant_11_solution/LAB_01_VARIANT_11_SOLUTION.ipynb) — готовое notebook-решение;
- [`variant_11_solution/report.md`](variant_11_solution/report.md) — пример отчета;
- [`variant_11_solution/data/source_en.md`](variant_11_solution/data/source_en.md) — входной англоязычный модуль;
- [`variant_11_solution/data/reference_ru.md`](variant_11_solution/data/reference_ru.md) — эталонная локализация;
- [`variant_11_solution/data/glossary.csv`](variant_11_solution/data/glossary.csv) — 18 утвержденных терминов;
- [`variant_11_solution/data/tm_pairs.csv`](variant_11_solution/data/tm_pairs.csv) — 10 EN→RU пар для TMX;
- [`variant_11_solution/data/segments_en.csv`](variant_11_solution/data/segments_en.csv) — результат сегментации;
- [`variant_11_solution/data/term_candidates.csv`](variant_11_solution/data/term_candidates.csv) — TF-IDF кандидаты;
- [`variant_11_solution/data/termbase.tbx`](variant_11_solution/data/termbase.tbx) — готовая termbase;
- [`variant_11_solution/data/glossary_smartcat.xlsx`](variant_11_solution/data/glossary_smartcat.xlsx) — glossary для импорта в Smartcat;
- [`variant_11_solution/data/memory.tmx`](variant_11_solution/data/memory.tmx) — готовая Translation Memory;
- [`variant_11_solution/data/terminology_qa.csv`](variant_11_solution/data/terminology_qa.csv) — результат первичной QA;
- [`variant_11_solution/requirements.txt`](variant_11_solution/requirements.txt) — зависимости Python.

#### Что показывает тестовое решение

Notebook автоматически выполняет:

```text
source_en.md
    ↓
Markdown segmentation
    ↓
segments_en.csv
    ↓
TF-IDF term candidates
    ↓
manual approved glossary.csv
    ↓
termbase.tbx
    ↓
glossary_smartcat.xlsx
    ↓
memory.tmx
    ↓
terminology_qa.csv
```

Интерактивная часть Smartcat остается ручной, поскольку требует учетной записи и текущего web-интерфейса. Эталонный пакет показывает все локально воспроизводимые артефакты и ожидаемую аналитическую часть.

### Требования к оформлению отчета в GitHub-репозитории

`lab01/report.md` должен содержать цель, вариант, терминологическую таблицу, 3–5 отклоненных кандидатов, фрагменты TBX/TMX, результат XML-проверки, скриншоты Smartcat, анализ 5 сегментов и вывод.

Обязательные файлы:

```text
data/source_en.md
data/glossary.csv
data/termbase.tbx
data/glossary_smartcat.xlsx
data/memory.tmx
data/reference_ru.md
lab01/report.md
requirements.txt
README.md
```

### Детализированный критериальный лист оценки (Рубрикатор на 20 баллов):

| Критерий | Состав критерия | Баллы |
|---|---|---:|
| **Работоспособность пайплайна/кода** | Сегментация/извлечение — 2; TBX/TMX — 2; Smartcat — 2 | **6** |
| **Корректность лингвистической обработки и валидации** | 15–20 терминов — 2; корректное различение TB/TM — 1; XML/terminology QA — 2 | **5** |
| **Анализ ошибок и аргументация решений** | Отклоненные кандидаты — 2; 5 CAT-сегментов — 2; вывод — 1 | **5** |
| **Оформление репозитория, reproducibility** | Полнота файлов — 1; README — 1; `requirements.txt` — 1; чистая структура — 1 | **4** |
| **Итого** |  | **20** |
