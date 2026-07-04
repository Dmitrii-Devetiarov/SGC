"""
Разбивка текстов на чанки, предфильтрация, кеширующий OCR.
"""

import hashlib
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class Chunker:
    """
    Разбивает текст на чанки фиксированного размера с перекрытием.
    Для pending_ocr — проверяет ChromaDB перед вызовом API.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_words: int = 15,
        yandex_api_client=None,
        vectordb=None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_words = min_chunk_words
        self.api = yandex_api_client
        self.vectordb = vectordb

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------

    def process(self, parsed_items: List[Dict]) -> List[Dict]:
        """
        Принимает результат парсера, возвращает список чанков:
        [
            {
                "text": str,
                "source": str,
                "page": int,
                "chunk_index": int,
                "is_ocr": bool,
            },
            ...
        ]
        """
        chunks = []

        for item in parsed_items:
            if item["pending_ocr"]:
                # Скан: проверяем кеш или делаем OCR
                ocr_chunks = self._process_ocr_item(item)
                chunks.extend(ocr_chunks)
            else:
                # Цифровой текст: просто разбиваем
                text_chunks = self._split_text(item["text"])
                for i, chunk_text in enumerate(text_chunks):
                    if self._is_meaningful(chunk_text):
                        chunks.append({
                            "text": chunk_text,
                            "source": item["source"],
                            "page": item["page"],
                            "chunk_index": i,
                            "is_ocr": False,
                        })

        logger.info(f"Всего чанков создано: {len(chunks)}")
        return chunks

    # ------------------------------------------------------------------
    # OCR с кешированием
    # ------------------------------------------------------------------

    def _process_ocr_item(self, item: Dict) -> List[Dict]:
        """OCR одной страницы скана с проверкой кеша в ChromaDB."""
        source = item["source"]
        page = item["page"]

        # Проверяем кеш: есть ли уже чанки для этой страницы в базе?
        if self.vectordb is not None:
            cached = self.vectordb.get_chunks_by_source_page(source, page)
            if cached:
                logger.info(f"Кеш: {source} стр. {page} — найдено {len(cached)} чанков")
                return cached

        # Кеша нет — делаем OCR
        if self.api is None:
            logger.warning(f"OCR недоступен, API-клиент отсутствует. {source} стр. {page} пропущена.")
            return []

        logger.info(f"OCR: {source} стр. {page}")
        try:
            text = self.api.ocr_image(item["image_base64"], page_num=page)
        except Exception as e:
            logger.error(f"Ошибка OCR: {source} стр. {page}: {e}")
            return []

        if not text or len(text.strip()) < 20:
            logger.warning(f"OCR вернул пустой/короткий текст: {source} стр. {page}")
            return []

        # Разбиваем результат OCR на чанки
        text_chunks = self._split_text(text)
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if self._is_meaningful(chunk_text):
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "chunk_index": i,
                    "is_ocr": True,
                })

        return chunks

    # ------------------------------------------------------------------
    # Разбивка текста
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> List[str]:
        """
        Простая разбивка по токенам (словам) с перекрытием.
        Без тяжёлых токенизаторов.
        """
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _is_meaningful(self, text: str) -> bool:
        """Отсев пустых и бессмысленных чанков."""
        if not text or len(text.strip()) < 20:
            return False
        words = text.split()
        if len(words) < self.min_chunk_words:
            return False
        # Отсев чанков, состоящих только из цифр/спецсимволов
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.3:
            return False
        return True
    
    def process_one(self, item: Dict) -> List[Dict]:
        """
        Обрабатывает один элемент (страницу) из парсера.
        Для pending_ocr — OCR с проверкой кеша.
        Для текста — разбивка на чанки.
        """
        if item["pending_ocr"]:
            return self._process_ocr_item(item)
        else:
            chunks = []
            text_chunks = self._split_text(item["text"])
            for i, chunk_text in enumerate(text_chunks):
                if self._is_meaningful(chunk_text):
                    chunks.append({
                        "text": chunk_text,
                        "source": item["source"],
                        "page": item["page"],
                        "chunk_index": i,
                        "is_ocr": False,
                    })
            return chunks