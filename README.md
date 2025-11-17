# D&D Rule Assistant (RAG)

Персональный RAG-бот по правилам D&D (PHB/DMG): Docling → Qdrant → GPT. Этот документ описывает структуру проекта и назначение каталогов/файлов.

## Структура проекта
```text
dnd_rule_assistant/
├── docker-compose.yml
├── pyproject.toml
├── .gitignore
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
├── src/
│   ├── core/
│   │   ├── config.py
│   │   ├── io.py
│   │   ├── retriever.py
│   │   └── pipelines.py
│   ├── providers/
│   │   ├── llm.py
│   │   ├── embeddings.py
│   │   └── vectorstore.py
│   └── interfaces/
│       ├── bot.py
│       └── cli.py
├── configs/
├── notebooks/
├── tests/
├── scripts/
├── logs/
└── qdrant_storage/
```

## Назначение папок и файлов
- data/
  - raw/: исходные документы (PDF/DOCX) без изменений.
  - processed/: очищенный текст, чанки (~800 токенов) и метаданные.
  - embeddings/: сохранённые эмбеддинги для офлайн-анализа/кеша.
- src/
  - core/: бизнес-логика и пайплайны.
    - config.py: централизованные настройки (через .env/pydantic-settings).
    - io.py: парсинг Docling, нормализация текста, сохранение/чтение артефактов.
    - retriever.py: векторный/гибридный поиск, top-k, (опц.) rerank.
    - pipelines.py: сценарии ingest/index/query как переиспользуемые функции.
  - providers/: адаптеры внешних сервисов.
    - llm.py: клиент LLM (OpenAI/OpenRouter) и шаблоны вызовов.
    - embeddings.py: генерация эмбеддингов (OpenAI, опц. локальные модели).
    - vectorstore.py: клиент/обёртка Qdrant (создание коллекции, upsert, поиск).
  - interfaces/: точки входа.
    - bot.py: Telegram-бот (aiogram), команда /ask <вопрос>.
    - cli.py: CLI (Typer) — check-env, init-qdrant, ingest, query.
- configs/: (опционально) YAML/JSON-конфиги; приоритет у .env.
  - `ingest.yaml`: параметры чанкинга (размер, overlap, таблицы) и секция `llm.model_name` с моделью OpenAI по умолчанию (`gpt-5-mini`, можно переопределить через `INGEST_LLM_MODEL_NAME`).
- notebooks/: исследования — чанкинг, качество поиска, промпты.
- tests/: автоматические тесты (smoke/интеграционные для пайплайна и ретривера).
- scripts/: инженерные утилиты (тонкие обёртки над функциями из src/interfaces/cli.py).
- logs/: лог-файлы выполнения (не коммитятся).
- qdrant_storage/: volume для Qdrant, монтируется Docker'ом (не коммитится).
- docker-compose.yml: локальный Qdrant (порт 6333) с маунтом `./qdrant_storage`.
- pyproject.toml: управление зависимостями через Poetry.
- .gitignore: исключения из Git.
- README.md: это описание структуры проекта.

Примечание: часть файлов появится по мере реализации модулей (ингест, эмбеддинги, ретривер, бота и CLI).

## Нормализация Markdown
Ключевые правила нормализации включают:\n
- Декодирование HTML/`/uniXXXX` последовательностей\n
- Удаление шумных строк и склейка разрывов слов/строк\n
- Нормализация дефиса/тире и пробелов\n
- Восстановление буквиц при наличии `pymorphy2`\n
- Безопасная замена латинских гомоглифов в русских словах (в т.ч. `h→н`, `m→м`, `t→т`)\n
- Замена коротких латинских слов в русских строках (белый список; при `pymorphy2` — проверка словаря)\n
- Замена цифр внутри русских слов (`3/6/0` → «З/з, Б/б, О/о»; `4` в конце → «й/Й»; исключая нотации костей)\n

## Новый двухэтапный пайплайн (Sections → Chunks → Qdrant)

- Разделение на секции (без резки по длине):
  - `poetry run python -m dnd_rag.interfaces.cli sections --in data/processed/md_clean --out data/processed/sections`

- Чанкинг секций токеновым RCTS (параметры из `configs/ingest.yaml`):
  - `poetry run python -m dnd_rag.interfaces.cli chunks --in data/processed/sections --out data/processed/chunks --config configs/ingest.yaml`

- Инициализация коллекции Qdrant (локально):
  - `docker compose up -d qdrant`
  - `poetry run python -m dnd_rag.interfaces.cli init-qdrant --collection dnd_rule_assistant --host localhost --port 6333 --dim 1536`

- Индексация чанков (OpenAI text-embedding-3-small, ключ в `OPENAI_API_KEY`):
  - `poetry run python -m dnd_rag.interfaces.cli index --collection dnd_rule_assistant --chunks data/processed/chunks/DMG.jsonl data/processed/chunks/PHB.jsonl`

### CLI-вопросы (RAG)

- Спросить ассистента напрямую:
  - `poetry run python -m dnd_rag.interfaces.cli ask "Как создаются персонажи?" --k 4`
- Можно переопределить модель:
  - `poetry run python -m dnd_rag.interfaces.cli ask "What is advantage?" --llm-model gpt-5-mini`

### Telegram-бот

1. Задайте токен и параметры подключения (`TELEGRAM_BOT_TOKEN`, при необходимости `QDRANT_HOST/PORT/COLLECTION`).
2. Запустите `poetry run python -m dnd_rag.interfaces.bot`.
3. В Telegram используйте `/ask <вопрос>`; бот вернёт ответ и список источников.