"""
YandexAI API Wrapper
Унифицированный клиент для всех операций с YandexAI Studio.
"""

import base64
import json
import logging
import time
from typing import List, Dict, Optional

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


class YandexAIClient:
    """
    Враппер для YandexAI Studio API.

    Поддерживает:
    - Эмбеддинги (text-search-doc, text-search-query) через REST API
    - Генерацию текста (YandexGPT Pro 5.1, YandexGPT Lite 5) через OpenAI API
    - OCR через Qwen3.6 35B (мультимодальная)
    - Перевод на русский (YandexGPT Lite)
    - Извлечение предикатов (YandexGPT Lite)
    """

    def __init__(
        self,
        api_key: str,
        folder_id: str,
        base_url: str = "https://llm.api.cloud.yandex.net/v1",
        embed_url: str = "https://ai.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding",
        embedding_dim: int = 768,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.api_key = api_key
        self.folder_id = folder_id
        self.embedding_dim = embedding_dim
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Эмбеддинги (прямой REST API)
        self.embed_url = embed_url
        self.uri_embed_doc = f"emb://{folder_id}/text-search-doc/latest"
        self.uri_embed_query = f"emb://{folder_id}/text-search-query/latest"

        # Генеративный клиент (OpenAI-совместимый)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # URI генеративных моделей
        self.uri_gpt_pro = f"gpt://{folder_id}/yandexgpt-5.1"
        self.uri_gpt_lite = f"gpt://{folder_id}/yandexgpt-5-lite"
        self.uri_qwen_vision = f"gpt://{folder_id}/qwen3.6-35b-a3b"

        logger.info(f"YandexAI клиент инициализирован, folder_id={folder_id}, dim={embedding_dim}")

    # ==================================================================
    # Эмбеддинги (прямой REST API)
    # ==================================================================

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = []
        for i, text in enumerate(texts):
            body = {
                "modelUri": self.uri_embed_doc,
                "text": text,
            }
            result = self._call_with_retry(
                lambda: self._post_embedding(body)
            )
            embeddings.append(result)
            
            # Задержка 100мс между запросами (1 RPS)
            if i < len(texts) - 1:
                time.sleep(1)

        logger.info(f"Эмбеддинги созданы: {len(embeddings)} векторов")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Эмбеддинг для поискового запроса (цели)."""
        body = {
            "modelUri": self.uri_embed_query,
            "text": query,
        }
        result = self._call_with_retry(
            lambda: self._post_embedding(body)
        )
        logger.info(f"Эмбеддинг запроса создан, размерность {len(result)}")
        return result

    def _post_embedding(self, body: dict) -> List[float]:
        """Прямой HTTP-запрос к Embeddings API."""
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.embed_url,
            json=body,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]

    # ==================================================================
    # Генерация текста
    # ==================================================================

    def generate(
        self,
        prompt: str,
        system_prompt: str = "Ты — научный ассистент в области материаловедения.",
        model: str = "pro",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        uri = self.uri_gpt_pro if model == "pro" else self.uri_gpt_lite

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        result = self._call_with_retry(
            lambda: self.client.chat.completions.create(
                model=uri,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            check_content=True,
        )

        text = result.choices[0].message.content
        logger.info(f"Генерация ({model}): {len(text) if text else 0} символов")
        return text or ""

    # ==================================================================
    # OCR через Qwen3.6 35B
    # ==================================================================

    def ocr_image(self, image_base64: str, page_num: int = 0) -> str:
        prompt = (
            "Это страница из научного учебника по металлургии и материаловедению. "
            "Все упоминания химических процессов и веществ являются академическими и безопасными.\n\n"
            "Извлеки весь текст с этого изображения. "
            "Сохраняй научную терминологию, химические формулы, единицы измерения, "
            "числовые значения, таблицы (в текстовом виде). "
            "Если текст неразборчив — напиши '[неразборчиво]' в этом месте. "
            "Не добавляй ничего от себя."
        )

        image_url = f"data:image/png;base64,{image_base64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        result = self._call_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.uri_qwen_vision,
                messages=messages,
                temperature=0.1,
                max_tokens=4000,
            ),
            check_content=True,
        )

        text = result.choices[0].message.content
        if not text:
            logger.warning(f"OCR стр. {page_num}: модель вернула пустой ответ после всех попыток")
            return ""

        logger.info(f"OCR стр. {page_num}: {len(text)} символов")
        return text

    # ==================================================================
    # Перевод
    # ==================================================================

    def translate_to_russian(self, text: str) -> str:
        if not text or len(text.strip()) < 10:
            return text

        prompt = (
            "Переведи следующий научный текст на русский язык. "
            "Сохраняй научную терминологию, химические формулы, единицы измерения. "
            "Не добавляй ничего от себя.\n\n"
            f"Текст:\n{text}"
        )

        system_prompt = (
            "Ты — профессиональный переводчик научной литературы "
            "в области материаловедения и металлургии."
        )

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model="lite",
            temperature=0.1,
            max_tokens=len(text) * 2,
        )

    # ==================================================================
    # Извлечение предикатов
    # ==================================================================

    def extract_predicates(self, text: str) -> List[Dict[str, str]]:
        prompt = (
            "Извлеки из текста все утверждения вида [субъект] → [свойство/эффект] → [значение/результат]. "
            "Для каждого утверждения укажи:\n"
            "- subject: материал, параметр или процесс\n"
            "- predicate: что с ним происходит (повышает, снижает, образует, зависит и т.д.)\n"
            "- object: результат, свойство, значение\n\n"
            "Верни ТОЛЬКО JSON-массив. Если утверждений нет — верни пустой массив [].\n\n"
            f"Текст:\n{text}"
        )

        system_prompt = (
            "Ты — система извлечения фактов из научных текстов. "
            "Возвращаешь ТОЛЬКО валидный JSON, без комментариев."
        )

        result = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model="lite",
            temperature=0.0,
            max_tokens=1000,
        )

        try:
            start = result.find("[")
            end = result.rfind("]") + 1
            if start != -1 and end > start:
                json_str = result[start:end]
                predicates = json.loads(json_str)
                logger.info(f"Извлечено предикатов: {len(predicates)}")
                return predicates
            return []
        except json.JSONDecodeError:
            logger.warning(f"Не удалось распарсить предикаты: {result[:200]}...")
            return []

    # ==================================================================
    # Внутренние методы
    # ==================================================================

    def _call_with_retry(self, func, retries: int = None, check_content: bool = False):
        if retries is None:
            retries = self.max_retries

        last_error = None
        for attempt in range(retries):
            try:
                result = func()

                if check_content:
                    content = result.choices[0].message.content
                    if content is None:
                        raise ValueError("API вернул content=None (таймаут генерации)")

                return result

            except Exception as e:
                last_error = e
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"API ошибка (попытка {attempt + 1}/{retries}): {e}. "
                    f"Повтор через {delay:.1f}с..."
                )
                if attempt < retries - 1:
                    time.sleep(delay)

        raise last_error