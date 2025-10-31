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

