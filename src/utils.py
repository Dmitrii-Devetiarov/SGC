"""
Вспомогательные функции: загрузка конфига с .env, логирование, метрики.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Загружает YAML-конфиг и подставляет переменные окружения.

    Сначала загружает .env файл (если есть), затем подставляет
    значения из переменных окружения вместо ${VAR_NAME} в конфиге.

    Args:
        config_path: путь к YAML-файлу конфигурации

    Returns:
        словарь с конфигурацией
    """
    # Загружаем .env в переменные окружения (если есть)
    _load_dotenv()

    # Читаем YAML
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Подстановка ${VAR_NAME} из переменных окружения
    def _subst(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(
                f"Переменная окружения '{var_name}' не установлена.\n"
                f"Создайте файл .env на основе .env.example или установите переменную."
            )
        return value

    resolved = re.sub(r'\$\{(\w+)\}', _subst, raw)
    config = yaml.safe_load(resolved)
    return config


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """
    Загружает переменные из .env файла в os.environ.
    Только если .env существует и переменная ещё не установлена.
    """
    env_file = Path(dotenv_path)
    if not env_file.exists():
        return

    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Пропускаем пустые строки и комментарии
            if not line or line.startswith("#"):
                continue
            # Формат: KEY=value
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Убираем кавычки, если есть
                value = value.strip("'").strip('"')
                # Устанавливаем только если ещё нет в окружении
                if key not in os.environ:
                    os.environ[key] = value


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Настраивает логирование из конфига.

    Args:
        config: словарь конфигурации (секция logging)
    """
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("file")

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    # Уменьшаем шум от сторонних библиотек
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def cosine_similarity(a, b) -> float:
    """Косинусное сходство между двумя векторами."""
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def ensure_dir(path: str) -> None:
    """Создаёт директорию, если её нет."""
    Path(path).mkdir(parents=True, exist_ok=True)