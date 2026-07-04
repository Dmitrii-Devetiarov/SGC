"""
Векторное хранилище на ChromaDB.
Сохранение, чтение, проверка кеша, выгрузка для gap_finder.
"""

import hashlib
import logging
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorDB:
    """
    Обёртка над ChromaDB для хранения чанков и метаданных.
    """

    def __init__(
        self,
        persist_path: str = "./db",
        collection_name: str = "documents",
        embedding_dim: int = 768,
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        self.client = chromadb.HttpClient(host="localhost", port=8000)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"VectorDB: {persist_path}, коллекция: {collection_name}")

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
    ) -> int:
        """
        Сохраняет чанки и их эмбеддинги. Пропускает дубликаты по ID.

        Returns:
            количество добавленных записей
        """
        if not chunks or not embeddings:
            return 0

        new_ids = []
        new_documents = []
        new_embeddings = []
        new_metadatas = []
        skipped = 0

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = self._make_id(chunk)
            existing = self.collection.get(ids=[chunk_id])
            if existing and existing["ids"]:
                skipped += 1
                continue

            new_ids.append(chunk_id)
            new_documents.append(chunk["text"])
            new_embeddings.append(emb)
            new_metadatas.append({
                "source": chunk.get("source", ""),
                "page": chunk.get("page", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "is_ocr": chunk.get("is_ocr", False),
                "type": chunk.get("type", ""),
                "original_language": chunk.get("original_language", "ru"),
                "char_count": len(chunk.get("text", "")),
            })

        if not new_ids:
            logger.info(f"Все {skipped} чанков уже в базе")
            return 0

        self.collection.add(
            ids=new_ids,
            documents=new_documents,
            embeddings=new_embeddings,
            metadatas=new_metadatas,
        )

        logger.info(f"Добавлено: {len(new_ids)}, пропущено: {skipped}")
        return len(new_ids)

    # ------------------------------------------------------------------
    # Кеш OCR
    # ------------------------------------------------------------------

    def get_chunks_by_source_page(self, source: str, page: int) -> List[Dict]:
        """Проверяет наличие чанков для source + page (кеш OCR)."""
        try:
            result = self.collection.get(
                where={"$and": [{"source": source}, {"page": page}]}
            )
        except Exception as e:
            logger.warning(f"Ошибка запроса кеша: {e}")
            return []

        if not result or not result["ids"]:
            return []

        chunks = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            if doc:
                chunks.append({
                    "text": doc,
                    "source": meta.get("source", source),
                    "page": meta.get("page", page),
                    "chunk_index": meta.get("chunk_index", 0),
                    "is_ocr": meta.get("is_ocr", False),
                    "type": meta.get("type", ""),
                    "original_language": meta.get("original_language", "ru"),
                    "char_count": meta.get("char_count", 0),
                })

        return chunks

    # ------------------------------------------------------------------
    # Выгрузка для gap_finder
    # ------------------------------------------------------------------

    def get_all_chunks_with_embeddings(self) -> tuple:
        result = self.collection.get(include=["documents", "embeddings", "metadatas"])

        if not result or not result["ids"]:
            logger.warning("База пуста")
            return [], []

        chunks = []
        embeddings = []

        for doc, emb, meta in zip(
            result["documents"],
            result["embeddings"],
            result["metadatas"],
        ):
            if doc and emb is not None and len(emb) > 0:
                # Проверяем на NaN
                import numpy as np
                if np.isnan(emb).any():
                    logger.warning(f"Пропущен чанк с NaN: {meta.get('source', '?')} стр. {meta.get('page', '?')}")
                    continue
                chunks.append({
                    "text": doc,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "is_ocr": meta.get("is_ocr", False),
                    "type": meta.get("type", ""),
                    "original_language": meta.get("original_language", "ru"),
                    "char_count": meta.get("char_count", 0),
                })
                embeddings.append(emb)

        logger.info(f"Выгружено: {len(chunks)} чанков (NaN пропущены)")
        return chunks, embeddings

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filter_source: Optional[str] = None,
    ) -> List[Dict]:
        """Поиск ближайших чанков к вектору запроса."""
        where = {"source": filter_source} if filter_source else None

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not result or not result["ids"][0]:
            return []

        chunks = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "is_ocr": meta.get("is_ocr", False),
                "type": meta.get("type", ""),
                "distance": dist,
            })

        return chunks

    # ------------------------------------------------------------------
    # Статистика и утилиты
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_path": self.persist_path,
            "embedding_dim": self.embedding_dim,
        }

    def _make_id(self, chunk: Dict) -> str:
        raw = f"{chunk['source']}_{chunk['page']}_{chunk['chunk_index']}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def reset(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("База очищена")