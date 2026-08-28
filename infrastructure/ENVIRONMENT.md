# ENVIRONMENT.md

# Среда выполнения лабораторного практикума  
## Дисциплина «Основы машинного перевода»

**Образовательная организация:** ГАОУ ВО МГПУ  
**Направление:** 44.03.05 «Педагогическое образование (с двумя профилями подготовки)»  
**Профиль:** «Информатика и английский язык»  
**Объем лабораторных занятий:** 36 часов  
**Рекомендуемая версия Python:** 3.10–3.12  
**Актуализация документа:** 27.08.2026

---

## 1. Назначение документа

Документ определяет аппаратные и программные требования к рабочим станциям студентов и преподавателя для выполнения лабораторного практикума по дисциплине «Основы машинного перевода».

Практикум включает четыре сквозные лабораторные работы:

1. компьютерная лексикография, TBX/TMX и Smartcat;
2. локальный нейросетевой перевод с использованием MarianMT и NLLB;
3. контекстный перевод с помощью GigaChat, YandexGPT или открытой LLM;
4. автоматическое и экспертное оценивание качества перевода с помощью BLEU, chrF, COMET, MTPE и MQM.

Рекомендуемое распределение 36 аудиторных часов:

| Лабораторная работа | Основная вычислительная нагрузка | Рекомендуемое время |
|---|---|---:|
| ЛР 1. TBX/TMX и Smartcat | CPU, XML/XLSX, браузер | 8 ч |
| ЛР 2. MarianMT и NLLB | CPU/GPU, PyTorch, Transformers | 10 ч |
| ЛР 3. GigaChat / YandexGPT / OpenSource LLM | API или локальный GPU | 8 ч |
| ЛР 4. BLEU, chrF, COMET, MTPE/MQM | CPU/GPU | 10 ч |
| **Итого** |  | **36 ч** |

---

# 2. Системные требования

## 2.1. Минимальная конфигурация рабочей станции

Минимальная конфигурация рассчитана на выполнение основной части практикума локально, а наиболее ресурсоемких операций — в облачной или выделенной вычислительной среде.

| Компонент | Минимальное требование |
|---|---|
| Процессор | x86-64, 4 физических/логических ядра |
| ОЗУ | **8 ГБ** |
| Накопитель | не менее 20 ГБ свободного места |
| ОС | Windows 11 x64 / Ubuntu 22.04+ / совместимый Linux / macOS |
| GPU | не требуется |
| Интернет | стабильное соединение от 10 Мбит/с |
| Браузер | актуальная версия Chrome, Edge или Firefox |
| Python | 3.10+ |
| Git | 2.40+ рекомендуется |

На компьютере с 8 ГБ ОЗУ допускается:

- работа с Smartcat;
- создание и разбор TBX/TMX;
- выполнение токенизации;
- MarianMT EN→RU на CPU;
- вычисление BLEU и chrF;
- работа с GigaChat/YandexGPT через API;
- небольшие эксперименты с COMET при достаточном объеме памяти.

Для `facebook/nllb-200-distilled-600M`, COMET и локальных LLM рекомендуется использовать GPU или внешнюю вычислительную среду.

---

## 2.2. Рекомендуемая конфигурация

| Компонент | Рекомендуемое требование |
|---|---|
| Процессор | 6–8 ядер и более |
| ОЗУ | **16 ГБ** |
| Накопитель | SSD, 40–60 ГБ свободного места |
| GPU | NVIDIA GPU с CUDA, **8 ГБ VRAM и более** |
| Интернет | 50 Мбит/с и выше |
| Python | 3.10–3.12 |
| Git | актуальная стабильная версия |
| IDE | VS Code или JupyterLab |

16 ГБ ОЗУ рекомендуется считать базовой конфигурацией компьютерного класса, если предполагается параллельная работа:

- VS Code/JupyterLab;
- браузера;
- PyTorch;
- MarianMT/NLLB;
- COMET;
- нескольких DataFrame и результатов экспериментов.

---

## 2.3. Варианты выполнения ресурсоемких задач

Ресурсоемкие задания ЛР 2 и ЛР 4 допускается выполнять одним из четырех способов.

### Вариант A. Локальный GPU

Предпочтительный вариант при наличии учебной рабочей станции с NVIDIA GPU.

Рекомендуется:

- 8 ГБ VRAM и более;
- современный драйвер NVIDIA;
- версия PyTorch, совместимая с установленным CUDA runtime.

Проверка:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM, GB:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        )
    )
```

---

### Вариант B. Google Colab

Google Colab может использоваться как резервная среда для:

- NLLB;
- COMET;
- демонстрации локального инференса;
- экспериментов с небольшими open-source моделями.

Необходимо учитывать:

- тип доступного ускорителя может изменяться;
- бесплатный доступ к GPU не гарантируется;
- время сессии ограничено;
- модели и пакеты могут требовать повторной загрузки при новом запуске runtime.

В отчете студент обязан указать:

```text
Environment: Google Colab
Python: ...
PyTorch: ...
GPU: ...
Transformers: ...
```

---

### Вариант C. RuCode / вузовская образовательная среда

RuCode может использоваться как дополнительная образовательная платформа для материалов по Python, Jupyter, машинному обучению и обработке естественного языка.

Наличие конкретного GPU-runtime для произвольного пользовательского notebook зависит от используемой площадки, курса или соглашения с образовательной организацией. Поэтому RuCode **не следует считать гарантированной заменой локального GPU или Colab**, если преподавателем заранее не предоставлена соответствующая вычислительная среда.

Если МГПУ предоставляет студентам отдельный RuCode/MФТИ runtime, преподаватель должен до начала занятия проверить:

- Python 3.10+;
- возможность установки `pip`-пакетов;
- доступ к Hugging Face;
- объем RAM;
- наличие GPU;
- возможность выгрузки файлов результатов.

---

### Вариант D. Только CPU + API

Это наиболее универсальная конфигурация.

Локально выполняются:

- ЛР 1;
- MarianMT;
- токенизация;
- BLEU/chrF;
- подготовка данных;
- MQM/MTPE.

Внешний инференс выполняется через:

- GigaChat API;
- Yandex AI Studio / YandexGPT API.

NLLB и COMET при необходимости выполняются преподавателем демонстрационно или в выделенной GPU-среде.

---

# 3. Программное обеспечение

## 3.1. Обязательный стек

| ПО / сервис | Назначение |
|---|---|
| Python 3.10+ | основной язык лабораторных работ |
| JupyterLab | интерактивные эксперименты |
| VS Code | основной редактор/IDE |
| Git | контроль версий |
| GitHub или GitFlic | размещение отчетов |
| Smartcat | CAT/TMS, TM, glossary |
| Hugging Face Transformers | MarianMT, NLLB |
| PyTorch | нейросетевой инференс |
| GigaChat SDK | отечественный генеративный API |
| Yandex AI Studio SDK | YandexGPT |
| sacreBLEU | BLEU, chrF, TER |
| unbabel-comet | нейронные метрики качества |
| spaCy / NLTK | базовая NLP-обработка |
| pandas / openpyxl | таблицы, XLSX, результаты |

---

## 3.2. Python

Проверьте версию:

```bash
python --version
```

или:

```bash
python3 --version
```

Требование:

```text
Python 3.10+
```

Для курса рекомендуется использовать Python **3.10, 3.11 или 3.12**. Это снижает вероятность несовместимости научных пакетов и моделей.

---

## 3.3. JupyterLab

Установка:

```bash
pip install jupyterlab
```

Запуск:

```bash
jupyter lab
```

JupyterLab рекомендуется для:

- исследования токенизаторов;
- визуальной проверки DataFrame;
- экспериментов с attention;
- вычисления метрик;
- анализа ошибок.

Итоговый воспроизводимый код лабораторной работы рекомендуется дополнительно сохранять в `.py`-файлах.

---

## 3.4. VS Code

Рекомендуемые расширения:

- Python;
- Jupyter;
- GitHub Pull Requests;
- Markdown All in One — опционально.

Проверка Python interpreter:

```text
Ctrl+Shift+P
→ Python: Select Interpreter
→ выбрать .venv
```

---

# 4. Smartcat

## 4.1. Режим доступа

Для лабораторных работ необходим доступ к:

- CAT Editor;
- Translation Memory;
- glossary / terminology;
- проектам локализации.

На 27.08.2026 Smartcat предоставляет бесплатный тариф **Forever Free**, который включает базовый CAT Editor, Translation Memory и glossaries.

В учебных материалах рекомендуется использовать формулировку:

> «бесплатная учетная запись Smartcat для учебной работы»

а не обещать отдельный универсальный тариф `Smartcat Academic Account`, если такой доступ не предоставлен МГПУ по отдельному соглашению.

Если у университета имеется академическое/корпоративное соглашение Smartcat, преподаватель может заменить данный раздел локальной инструкцией входа.

Регистрация:

<https://www.smartcat.com/>

Справка:

<https://help.smartcat.com/>

---

## 4.2. Подготовка Smartcat до первой лабораторной

Каждый студент должен:

1. создать учетную запись;
2. войти в workspace;
3. создать тестовый EN→RU проект;
4. убедиться, что CAT Editor открывается;
5. проверить возможность создать glossary;
6. проверить возможность создать или импортировать Translation Memory.

Не следует расходовать доступный бесплатный AI-лимит на тестовые тексты до начала лабораторной работы.

---

# 5. Отечественные LLM API

## 5.1. GigaChat SDK

Установка:

```bash
pip install gigachat
```

Проверка установки:

```bash
python -c "import gigachat; print('gigachat: OK')"
```

Официальная документация:

<https://developers.sber.ru/docs/ru/gigachat/guides/using-sdks>

Для работы API требуется авторизационный ключ/credentials, полученный в проекте GigaChat API.

### Переменная окружения

Windows PowerShell:

```powershell
$env:GIGACHAT_CREDENTIALS="YOUR_KEY"
```

Linux/macOS:

```bash
export GIGACHAT_CREDENTIALS="YOUR_KEY"
```

Проверка без вывода секрета:

```python
import os

key = os.getenv("GIGACHAT_CREDENTIALS")

print(
    "GIGACHAT_CREDENTIALS:",
    "FOUND" if key else "MISSING"
)
```

**Запрещено:**

```python
print(os.getenv("GIGACHAT_CREDENTIALS"))
```

Такой вывод может случайно попасть в notebook, скриншот или Git history.

---

## 5.2. Yandex AI Studio / YandexGPT

В исходном практикуме сохраняется зависимость:

```text
yandex-cloud-ml-sdk
```

Однако на август 2026 г. пакет `yandex-cloud-ml-sdk` является **wrapper-пакетом обратной совместимости**. Актуальное развитие SDK перенесено в `yandex-ai-studio-sdk`.

Поэтому рекомендуется:

- для воспроизведения предоставленного практикума — установить `yandex-cloud-ml-sdk`;
- для нового кода преподавателя — ориентироваться на актуальный `yandex-ai-studio-sdk`;
- перед семестром проверить страницу миграции в официальной документации.

Минимальная установка в рамках требований курса:

```bash
pip install yandex-cloud-ml-sdk
```

Проверка:

```bash
python -c "import yandex_cloud_ml_sdk; print('Yandex Cloud ML SDK: OK')"
```

Для нового SDK при необходимости:

```bash
pip install yandex-ai-studio-sdk
```

Основные переменные окружения курса:

Windows PowerShell:

```powershell
$env:YANDEX_FOLDER_ID="YOUR_FOLDER_ID"
$env:YANDEX_API_KEY="YOUR_API_KEY"
```

Linux/macOS:

```bash
export YANDEX_FOLDER_ID="YOUR_FOLDER_ID"
export YANDEX_API_KEY="YOUR_API_KEY"
```

Проверка:

```python
import os

required = [
    "YANDEX_FOLDER_ID",
    "YANDEX_API_KEY",
]

for name in required:
    print(
        name,
        "FOUND" if os.getenv(name) else "MISSING"
    )
```

Документация Yandex AI Studio:

<https://yandex.cloud/ru/docs/ai-studio/>

---

# 6. Полный `requirements.txt`

Для курса используйте следующий базовый файл:

```text
torch
transformers
sentencepiece
sacremoses
sacrebleu
unbabel-comet
spacy
nltk
pandas
openpyxl
gigachat
yandex-cloud-ml-sdk
```

Файл должен находиться в корне студенческого репозитория:

```text
mt-course-project/
├── requirements.txt
├── ENVIRONMENT.md
├── README.md
├── data/
├── lab01/
├── lab02/
├── lab03/
└── lab04/
```

Установка:

```bash
pip install -r requirements.txt
```

### Дополнительные пакеты

Для Jupyter:

```bash
pip install jupyterlab ipykernel
```

Для XML/TBX:

```bash
pip install lxml
```

Если в коде используется `.env`:

```bash
pip install python-dotenv
```

Для актуального Yandex AI Studio SDK:

```bash
pip install yandex-ai-studio-sdk
```

После успешной настройки преподаватель может сформировать зафиксированное окружение:

```bash
pip freeze > requirements-lock.txt
```

`requirements.txt` в учебном репозитории рекомендуется оставлять читаемым, а `requirements-lock.txt` использовать для диагностики полной воспроизводимости.

---

# 7. Создание виртуального окружения

## 7.1. Вариант 1 — `venv`

### Windows 11 / PowerShell

Создание:

```powershell
python -m venv .venv
```

Активация:

```powershell
.venv\Scripts\Activate.ps1
```

Обновление pip:

```powershell
python -m pip install --upgrade pip
```

Установка зависимостей:

```powershell
pip install -r requirements.txt
```

Проверка:

```powershell
python --version
pip --version
```

---

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Выход из окружения:

```bash
deactivate
```

---

## 7.2. Вариант 2 — Conda

Создание:

```bash
conda create -n mt-course python=3.11
```

Активация:

```bash
conda activate mt-course
```

Установка:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Проверка:

```bash
python --version
python -c "import torch; print(torch.__version__)"
```

Не рекомендуется одновременно устанавливать один и тот же пакет через `conda` и `pip` без необходимости.

---

# 8. Первичная проверка окружения

После установки выполните:

```python
import sys
import torch
import transformers
import pandas
import sacrebleu
import spacy
import nltk

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("pandas:", pandas.__version__)
print("sacreBLEU:", sacrebleu.__version__)
print("CUDA:", torch.cuda.is_available())
```

Проверка `sentencepiece`:

```bash
python -c "import sentencepiece; print('sentencepiece: OK')"
```

Проверка GigaChat:

```bash
python -c "import gigachat; print('gigachat: OK')"
```

Проверка COMET:

```bash
python -c "import comet; print('COMET: OK')"
```

---

## 8.1. Ресурсы NLTK

При необходимости:

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
```

---

## 8.2. Модель spaCy

Для базовых экспериментов с английским языком:

```bash
python -m spacy download en_core_web_sm
```

Проверка:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Machine translation processes natural language.")

print([token.text for token in doc])
```

---

# 9. Проверка доступа к Hugging Face

Для публичных моделей курса специальный API-ключ обычно не требуется, однако доступ к домену Hugging Face должен быть разрешен в сети компьютерного класса.

Проверка MarianMT:

```python
from transformers import AutoTokenizer

name = "Helsinki-NLP/opus-mt-en-ru"

tokenizer = AutoTokenizer.from_pretrained(name)

print(
    tokenizer.tokenize(
        "Machine translation"
    )
)
```

Проверка NLLB:

```python
from transformers import AutoTokenizer

name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(
    name,
    src_lang="eng_Latn",
    tgt_lang="rus_Cyrl"
)

print("NLLB tokenizer: OK")
```

Модели желательно загрузить **до занятия**, если в аудитории ограничена пропускная способность сети.

---

# 10. Рекомендуемая предварительная настройка компьютерного класса

За 1–3 дня до начала лабораторного цикла преподаватель или администратор должен проверить:

- Python установлен;
- Git установлен;
- VS Code/JupyterLab запускаются;
- `pip install` разрешен;
- доступен PyPI;
- доступен Hugging Face;
- Smartcat открывается;
- GigaChat Developer Portal/API доступен;
- Yandex Cloud / AI Studio доступен;
- GitHub или GitFlic доступны;
- модели MarianMT/NLLB могут быть загружены;
- COMET может загрузить модель;
- на диске остается не менее 20 ГБ.

Для компьютерного класса полезно заранее прогреть кэш моделей.

Пример расположения Hugging Face cache:

```text
~/.cache/huggingface/
```

или на Windows:

```text
C:\Users\<USER>\.cache\huggingface\
```

Не следует вручную копировать кэш между учетными записями без учета прав доступа.

---

# 11. API-ключи и информационная безопасность

## 11.1. Что нельзя помещать в Git

Запрещено коммитить:

- API keys;
- OAuth tokens;
- GigaChat credentials;
- Yandex API keys;
- IAM tokens;
- файлы с персональными данными студентов;
- приватные translation memories, если они содержат реальные данные.

Обязательный `.gitignore`:

```gitignore
.venv/
.env
.env.*
__pycache__/
*.pyc
.ipynb_checkpoints/
.cache/
```

При необходимости можно оставить пример:

```text
.env.example
```

с пустыми значениями:

```text
GIGACHAT_CREDENTIALS=
YANDEX_FOLDER_ID=
YANDEX_API_KEY=
```

---

## 11.2. Проверка перед `git push`

```bash
git status
```

Дополнительно:

```bash
git diff --cached
```

Студент обязан убедиться, что секреты отсутствуют в staged files.

Если ключ уже был опубликован, недостаточно просто удалить строку из последнего коммита: ключ следует **отозвать/перевыпустить**.

---

# 12. Git: локальная настройка

Проверка:

```bash
git --version
```

Первичная настройка:

```bash
git config --global user.name "Student Name"
git config --global user.email "student@example.edu"
```

Проверка:

```bash
git config --global --list
```

Клонирование:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

Стандартный цикл:

```bash
git status
git add .
git commit -m "Complete lab 01"
git push
```

Рекомендуемые сообщения коммитов:

```text
init: create course repository
lab01: add TBX and TMX resources
lab02: add MarianMT and NLLB pipeline
lab03: add controlled LLM translation
lab04: add BLEU chrF COMET and MQM evaluation
docs: update README and environment instructions
```

---

# 13. Организация репозиториев студентов

## 13.1. Общий принцип

Для каждой учебной группы преподаватель поддерживает **эталонный шаблон**:

```text
machine-translation-course-template/
├── README.md
├── ENVIRONMENT.md
├── requirements.txt
├── data/
│   └── .gitkeep
├── lab01/
│   └── README.md
├── lab02/
│   └── README.md
├── lab03/
│   └── README.md
└── lab04/
    └── README.md
```

Для каждого студента создается отдельный репозиторий:

```text
mt-course-<group>-<surname>
```

Пример:

```text
mt-course-241-smirnov
```

Рекомендуемая политика:

- один студент — один репозиторий;
- `main` — сдаваемая версия;
- промежуточная работа может выполняться в feature branches;
- каждая лабораторная заканчивается отдельным коммитом/tag;
- отчеты хранятся в Markdown;
- крупные модели в Git **не загружаются**.

---

# 14. GitHub Classroom

> **Критическое обновление 2026:** GitHub официально объявил завершение работы GitHub Classroom **28 августа 2026 года**.

Поэтому для нового учебного цикла после этой даты **не следует строить основной процесс курса на GitHub Classroom**.

Ниже приведена схема только для уже существующего Classroom до прекращения работы сервиса.

## Legacy-сценарий

1. Создать GitHub Organization.
2. Создать Classroom.
3. Подключить template repository.
4. Создать Individual Assignment.
5. Установить private visibility.
6. Опубликовать invitation link.
7. После принятия задания Classroom автоматически создает персональный репозиторий студента.

GitHub Classroom исторически поддерживал:

- template repositories;
- individual/group assignments;
- feedback;
- autograding;
- private student repositories.

После 28.08.2026 рекомендуется перейти на один из двух вариантов:

### GitHub без Classroom

1. создать organization;
2. создать template repository;
3. создавать/генерировать персональные репозитории из template;
4. выдавать студенту доступ `Write`;
5. преподавателю оставлять `Maintain/Admin`;
6. использовать Issues/Pull Requests для обратной связи.

### GitFlic

Использовать отечественную платформу как основной вариант для нового семестра.

---

# 15. GitFlic

Официальная документация:

<https://docs.gitflic.ru/>

## 15.1. Создание преподавательского проекта

Преподаватель создает:

```text
mgpu-machine-translation-template
```

Рекомендуется включить:

- `README.md`;
- `ENVIRONMENT.md`;
- `.gitignore`;
- `requirements.txt`;
- каталоги четырех лабораторных.

GitFlic поддерживает создание проекта, импорт существующего Git-репозитория и форки.

---

## 15.2. Рекомендуемая схема для студентов

Вариант A — **fork** преподавательского проекта:

```text
mgpu-machine-translation-template
        ↓ fork
student-login/mt-course
```

Вариант B — создание нового проекта и подключение шаблона:

```bash
git clone <TEMPLATE_URL>
cd <TEMPLATE_DIRECTORY>

git remote remove origin
git remote add origin <STUDENT_GITFLIC_URL>

git push -u origin main
```

Если базовая ветка в проекте GitFlic называется `master`, используйте соответствующее имя.

Проверка:

```bash
git remote -v
```

---

## 15.3. Приватность

Для отчетов студентов рекомендуется:

```text
Private repository
```

Преподавателю должен быть предоставлен доступ.

GitFlic поддерживает роли:

- Guest;
- Reporter;
- Developer;
- Administrator;
- Owner.

Для проверки работы преподавателю обычно достаточно роли, позволяющей читать репозиторий и просматривать историю; конкретная политика может задаваться кафедрой.

Следует учитывать ограничения публичного облачного тарифа GitFlic по числу участников приватного проекта. Для массового курса предпочтительнее:

- персональные проекты студентов;
- GitFlic Enterprise/OnPremise при наличии у университета;
- централизованная организация/компания с заранее настроенными правами.

---

# 16. Рекомендуемый workflow студента

```mermaid
flowchart LR
    A[Получить шаблон] --> B[Создать локальное окружение]
    B --> C[Установить requirements]
    C --> D[Проверить Smartcat и API]
    D --> E[Выполнить ЛР]
    E --> F[Запустить проверки]
    F --> G[Обновить report.md]
    G --> H[git status]
    H --> I[Commit]
    I --> J[Push]
    J --> K[Отправить ссылку в Moodle]
```

Перед сдачей каждой лабораторной:

```bash
python --version
pip check
git status
```

`pip check` должен завершаться без конфликтов зависимостей.

---

# 17. Требования к воспроизводимости

Каждый студенческий репозиторий должен содержать:

```text
README.md
ENVIRONMENT.md
requirements.txt
```

В `README.md` студент указывает:

```text
OS:
Python:
CPU:
RAM:
GPU:
PyTorch:
Transformers:
Execution environment:
```

Пример:

```text
OS: Windows 11 24H2
Python: 3.11.x
CPU: Intel Core i5
RAM: 16 GB
GPU: NVIDIA RTX ..., 8 GB
PyTorch: ...
Transformers: ...
Execution environment: local
```

или:

```text
Execution environment: Google Colab
```

API-ключи в README не указываются.

---

# 18. Быстрая диагностика перед лабораторной

Создайте временный файл `check_environment.py`:

```python
import importlib
import os
import platform
import sys

packages = [
    "torch",
    "transformers",
    "sentencepiece",
    "sacremoses",
    "sacrebleu",
    "comet",
    "spacy",
    "nltk",
    "pandas",
    "openpyxl",
    "gigachat",
]

print("OS:", platform.platform())
print("Python:", sys.version.split()[0])
print()

for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "installed")
        print(f"[OK] {package}: {version}")
    except Exception as exc:
        print(f"[FAIL] {package}: {exc}")

print()
print(
    "GigaChat key:",
    "FOUND" if os.getenv("GIGACHAT_CREDENTIALS") else "MISSING"
)
print(
    "Yandex folder:",
    "FOUND" if os.getenv("YANDEX_FOLDER_ID") else "MISSING"
)
print(
    "Yandex key:",
    "FOUND" if os.getenv("YANDEX_API_KEY") else "MISSING"
)
```

Запуск:

```bash
python check_environment.py
```

Для PyTorch дополнительно:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 19. Рекомендуемая конфигурация для компьютерного класса МГПУ

Для стабильного проведения всех **36 часов** лабораторных занятий рекомендуется следующий профиль рабочего места:

```text
Windows 11 x64
CPU: 6+ ядер
RAM: 16 GB
SSD: 50 GB свободного пространства
Python: 3.11
Git
VS Code
JupyterLab
Chrome/Edge
локальный .venv
доступ к Smartcat
доступ к Hugging Face
доступ к GigaChat API
доступ к Yandex AI Studio
доступ к GitFlic
```

GPU не является обязательным для каждого рабочего места.

Оптимальная инфраструктура группы:

- все рабочие станции — 16 ГБ RAM;
- 1–2 преподавательские/лабораторные машины с NVIDIA GPU **или**
- заранее подготовленная внешняя GPU-среда;
- локальный CPU как fallback;
- облачные API для ЛР 3.

Такой вариант позволяет не делать наличие дорогостоящей GPU на каждом компьютере обязательным условием курса.

---

# 20. Контрольный чек-лист преподавателя

До первого занятия:

- [ ] Python 3.10+ установлен.
- [ ] Git установлен.
- [ ] VS Code или JupyterLab запускается.
- [ ] `pip install -r requirements.txt` выполняется.
- [ ] Smartcat доступен из сети университета.
- [ ] Hugging Face доступен.
- [ ] MarianMT загружается.
- [ ] NLLB загружается хотя бы на тестовой машине.
- [ ] COMET запускается хотя бы на одной доступной среде.
- [ ] GigaChat API проверен.
- [ ] Yandex AI Studio проверен.
- [ ] GitFlic/GitHub доступен.
- [ ] Подготовлен template repository.
- [ ] API-ключи не встроены в шаблон.
- [ ] Подготовлен резервный сценарий без GPU.
- [ ] Проверено не менее 20 ГБ свободного места на рабочих станциях.

---

# 21. Контрольный чек-лист студента

До начала ЛР 1:

- [ ] Репозиторий клонирован.
- [ ] `.venv` или Conda environment создан.
- [ ] `requirements.txt` установлен.
- [ ] `pip check` не показывает конфликтов.
- [ ] Smartcat открывается.
- [ ] Git push работает.
- [ ] `.env` исключен через `.gitignore`.

До ЛР 2:

- [ ] MarianMT загружается.
- [ ] NLLB tokenizer загружается.
- [ ] Определен CPU/GPU/Colab сценарий.

До ЛР 3:

- [ ] Получен хотя бы один разрешенный API-доступ.
- [ ] Переменная окружения определяется программой.
- [ ] Ключ не выводится на экран и не помещен в Git.

До ЛР 4:

- [ ] `sacrebleu` импортируется.
- [ ] `comet` импортируется.
- [ ] подготовлен единый evaluation dataset.
- [ ] зафиксирована версия программной среды.

---

# 22. Ссылки

- Python: <https://www.python.org/>
- JupyterLab: <https://jupyter.org/>
- Visual Studio Code: <https://code.visualstudio.com/>
- PyTorch: <https://pytorch.org/>
- Hugging Face Transformers: <https://huggingface.co/docs/transformers/>
- MarianMT: <https://huggingface.co/Helsinki-NLP/opus-mt-en-ru>
- NLLB: <https://huggingface.co/facebook/nllb-200-distilled-600M>
- Smartcat: <https://www.smartcat.com/>
- Smartcat Help Center: <https://help.smartcat.com/>
- GigaChat API: <https://developers.sber.ru/docs/ru/gigachat/>
- Yandex AI Studio: <https://yandex.cloud/ru/docs/ai-studio/>
- sacreBLEU: <https://github.com/mjpost/sacrebleu>
- COMET: <https://github.com/Unbabel/COMET>
- Git: <https://git-scm.com/>
- GitHub: <https://github.com/>
- GitFlic: <https://gitflic.ru/>
- GitFlic Documentation: <https://docs.gitflic.ru/>
- RuCode: <https://rucode.net/>
