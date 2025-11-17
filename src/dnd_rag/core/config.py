# ==============================================================================
# Configuration module for document ingestion settings
# Модуль конфигурации для настроек загрузки документов
# ==============================================================================
# This file manages configuration settings for processing D&D documents.
# It loads settings from a YAML file and allows environment variables to override them.
#
# Этот файл управляет настройками конфигурации для обработки документов D&D.
# Он загружает настройки из YAML-файла и позволяет переменным окружения переопределять их.
# ==============================================================================

from __future__ import annotations

# Import Path for working with file paths in a cross-platform way
# Импортируем Path для работы с путями к файлам кросс-платформенным способом
from pathlib import Path

# Import Optional to indicate that a value can be None
# Импортируем Optional, чтобы указать, что значение может быть None
from typing import Optional

# Import yaml to read YAML configuration files
# Импортируем yaml для чтения конфигурационных файлов YAML
import yaml

# Import Pydantic models for data validation and settings management
# Импортируем модели Pydantic для валидации данных и управления настройками
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# ==============================================================================
# Main configuration class for document ingestion
# Основной класс конфигурации для загрузки документов
# ==============================================================================
class IngestConfig(BaseModel):
    """
    Configuration parameters for document processing.
    Параметры конфигурации для обработки документов.
    
    This class defines how documents should be split and processed.
    Этот класс определяет, как документы должны быть разделены и обработаны.
    """
    
    # Maximum size of each text chunk in tokens (roughly words)
    # Максимальный размер каждого фрагмента текста в токенах (примерно слова)
    # Default: 800 tokens per chunk
    # По умолчанию: 800 токенов на фрагмент
    chunk_size_tokens: int = 800
    
    # How many tokens should overlap between consecutive chunks
    # Сколько токенов должно перекрываться между последовательными фрагментами
    # This helps maintain context between chunks
    # Это помогает сохранять контекст между фрагментами
    # Default: 120 tokens overlap
    # По умолчанию: 120 токенов перекрытия
    chunk_overlap_tokens: int = 120
    
    # Minimum number of characters a paragraph must have to be processed
    # Минимальное количество символов, которое должен иметь параграф для обработки
    # Paragraphs shorter than this will be skipped or merged
    # Параграфы короче этого будут пропущены или объединены
    # Default: 40 characters
    # По умолчанию: 40 символов
    min_paragraph_chars: int = 40
    
    # Whether to keep tables as whole blocks (not split them)
    # Сохранять ли таблицы как целые блоки (не разделять их)
    # True = tables stay intact, False = tables can be split
    # True = таблицы остаются целыми, False = таблицы могут быть разделены
    # Default: True
    # По умолчанию: True
    keep_tables_as_blocks: bool = True
    
    # Whether to include image captions in the processed text
    # Включать ли подписи к изображениям в обработанный текст
    # True = include captions, False = ignore captions
    # True = включать подписи, False = игнорировать подписи
    # Default: True
    # По умолчанию: True
    include_image_captions: bool = False


# ==============================================================================
# Environment variable overrides class
# Класс переопределений через переменные окружения
# ==============================================================================
class EnvIngestOverrides(BaseSettings):
    """
    Allows overriding configuration using environment variables.
    Позволяет переопределять конфигурацию через переменные окружения.
    
    Example: Set INGEST_CHUNK_SIZE_TOKENS=1000 to change chunk size.
    Пример: Установите INGEST_CHUNK_SIZE_TOKENS=1000 чтобы изменить размер фрагмента.
    """
    
    # Configuration for how to read environment variables
    # Конфигурация для чтения переменных окружения
    model_config = SettingsConfigDict(
        # All environment variables must start with "INGEST_"
        # Все переменные окружения должны начинаться с "INGEST_"
        env_prefix="INGEST_",
        
        # Use "__" to access nested settings (e.g., INGEST_SUB__VALUE)
        # Используйте "__" для доступа к вложенным настройкам (например, INGEST_SUB__VALUE)
        env_nested_delimiter="__",
        
        # Ignore extra environment variables that don't match our fields
        # Игнорировать дополнительные переменные окружения, которые не соответствуют нашим полям
        extra="ignore",
    )

    # Optional overrides for each setting (None = use default from IngestConfig)
    # Опциональные переопределения для каждой настройки (None = использовать значение по умолчанию из IngestConfig)
    chunk_size_tokens: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None
    min_paragraph_chars: Optional[int] = None
    keep_tables_as_blocks: Optional[bool] = None
    include_image_captions: Optional[bool] = None


# ==============================================================================
# Helper function to find the default configuration file
# Вспомогательная функция для поиска файла конфигурации по умолчанию
# ==============================================================================
def _resolve_default_ingest_path() -> Path:
    """
    Find configs/ingest.yaml by searching upward from current file.
    Найти configs/ingest.yaml, поднимаясь вверх от текущего файла.
    
    This function searches for the configuration file in parent directories.
    Эта функция ищет конфигурационный файл в родительских директориях.
    
    Returns / Возвращает:
        Path to the ingest.yaml file / Путь к файлу ingest.yaml
    """
    # Get the absolute path of this Python file
    # Получаем абсолютный путь этого Python-файла
    start = Path(__file__).resolve()
    
    # Search in current directory and all parent directories
    # Ищем в текущей директории и всех родительских директориях
    for p in [start] + list(start.parents):
        # Check if configs/ingest.yaml exists in this directory
        # Проверяем, существует ли configs/ingest.yaml в этой директории
        cand = p / "configs" / "ingest.yaml"
        if cand.exists():
            return cand
    
    # Fallback: try to go up 3 levels (repository root) and build path
    # Резервный вариант: попытка подняться на 3 уровня (корень репозитория) и построить путь
    try:
        root = Path(__file__).resolve().parents[3]
        return root / "configs" / "ingest.yaml"
    except Exception:
        # If all else fails, return relative path
        # Если все остальное не сработало, вернуть относительный путь
        return Path("configs/ingest.yaml")


# Default path to the configuration file
# Путь по умолчанию к конфигурационному файлу
DEFAULT_CONFIG_PATH = _resolve_default_ingest_path()


# ==============================================================================
# Main function to load and merge configuration
# Основная функция для загрузки и объединения конфигурации
# ==============================================================================
def load_ingest_config(path: Optional[str | Path] = None) -> IngestConfig:
    """
    Load configuration from YAML file and apply environment variable overrides.
    Загрузить конфигурацию из YAML-файла и применить переопределения из переменных окружения.
    
    Parameters / Параметры:
        path: Optional custom path to config file. If None, uses default path.
              Опциональный пользовательский путь к файлу конфигурации. Если None, используется путь по умолчанию.
    
    Returns / Возвращает:
        IngestConfig object with all settings merged
        Объект IngestConfig со всеми объединенными настройками
    
    Priority / Приоритет (highest to lowest / от высшего к низшему):
        1. Environment variables (INGEST_*) / Переменные окружения (INGEST_*)
        2. YAML file settings / Настройки из YAML-файла
        3. Default values in IngestConfig class / Значения по умолчанию в классе IngestConfig
    """
    # Determine which config file to use
    # Определяем, какой конфигурационный файл использовать
    file_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    # If path is relative, try to find it relative to the nearest parent with configs/
    # Если путь относительный, пробуем найти его относительно ближайшего родителя с configs/
    if not file_path.is_absolute():
        for p in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
            cand = p / file_path
            if cand.exists():
                file_path = cand
                break

    # Try to load the YAML file
    # Пытаемся загрузить YAML-файл
    if not file_path.exists():
        # Diagnostics: report where we're looking
        # Диагностика: сообщаем, куда смотрим
        print(f"[CFG] ingest.yaml не найден по пути: {file_path}")
        print(f"[CFG] ingest.yaml not found at path: {file_path}")
        data = {}
    else:
        # File found, load it
        # Файл найден, загружаем его
        print(f"[CFG] ingest.yaml: {file_path} (exists=True)")
        with file_path.open("r", encoding="utf-8") as f:
            # Parse YAML file into a Python dictionary
            # Парсим YAML-файл в Python-словарь
            data = yaml.safe_load(f) or {}

    # Extract the 'ingest' section from YAML, or use the whole dict if no section exists
    # Извлекаем секцию 'ingest' из YAML, или используем весь словарь, если секции нет
    params = data.get("ingest", data) if isinstance(data, dict) else {}
    
    # Create IngestConfig object with parameters from YAML file
    # Создаём объект IngestConfig с параметрами из YAML-файла
    cfg = IngestConfig(**params)

    # Load environment variable overrides
    # Загружаем переопределения из переменных окружения
    overrides = EnvIngestOverrides()
    
    # Get only the overrides that were actually set (not None)
    # Получаем только те переопределения, которые были установлены (не None)
    override_dict = overrides.model_dump(exclude_none=True)
    
    # If there are any environment overrides, apply them
    # Если есть какие-либо переопределения из окружения, применяем их
    if override_dict:
        # Create a new config with overridden values
        # Создаём новую конфигурацию с переопределёнными значениями
        cfg = cfg.model_copy(update=override_dict)

    # Return the final merged configuration
    # Возвращаем итоговую объединённую конфигурацию
    return cfg

