"""
Индексация документов: парсинг → чанкинг → эмбеддинги → ChromaDB.
С инкрементальной записью: каждый чанк сохраняется сразу после обработки.
"""

import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parser import DocumentParser
from chunker import Chunker
from vectordb import VectorDB
from yandex_api import YandexAIClient
from utils import load_config, setup_logging, ensure_dir

logger = logging.getLogger("indexer")


def main():
    config = load_config()
    setup_logging(config)

    api = YandexAIClient(
        api_key=config["yandex_api"]["api_key"],
        folder_id=config["yandex_api"]["folder_id"],
        base_url=config["yandex_api"]["base_url"],
        embedding_dim=config["yandex_api"]["embedding_dim"],
    )

    vectordb = VectorDB(
        persist_path=config["paths"]["db"],
        collection_name="documents",
        embedding_dim=config["yandex_api"]["embedding_dim"],
    )

    parser = DocumentParser(
        ocr_dpi=config["chunking"]["ocr_dpi"],
        ocr_max_width=config["chunking"]["ocr_max_width"],
    )

    chunker = Chunker(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        min_chunk_words=config["chunking"]["min_chunk_words"],
        yandex_api_client=api,
        vectordb=vectordb,
    )

    input_dir = Path(config["paths"]["data_input"])
    if not input_dir.exists():
        logger.error(f"Папка не найдена: {input_dir}")
        return

    extensions = [".pdf", ".txt", ".csv", ".xlsx", ".xls"]
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

    if not files:
        logger.error(f"Нет файлов в {input_dir}")
        return

    logger.info(f"Найдено файлов: {len(files)}")

    for file_idx, file_path in enumerate(sorted(files), start=1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{file_idx}/{len(files)}] {file_path.name}")
        logger.info(f"{'='*60}")

        # Парсинг
        parsed = parser.parse(str(file_path))
        digital = sum(1 for p in parsed if not p["pending_ocr"])
        scans = sum(1 for p in parsed if p["pending_ocr"])
        logger.info(f"  Страниц: {len(parsed)} (цифровых: {digital}, сканов: {scans})")

        # Обрабатываем постранично, сразу сохраняем
        for page_idx, item in enumerate(parsed, start=1):
            try:
                chunks = chunker.process_one(item)

                if not chunks:
                    continue

                # Эмбеддинги
                texts = [c["text"] for c in chunks]
                embeddings = api.embed_documents(texts)

                # Сразу в БД
                added = vectordb.add_chunks(chunks, embeddings)
                if added > 0:
                    logger.debug(f"  Стр. {item['page']}: сохранено {added} чанков")

            except Exception as e:
                logger.error(f"  Ошибка на стр. {item['page']}: {e}")
                continue

        # Статистика после файла
        stats = vectordb.get_stats()
        logger.info(f"  Всего в БД: {stats['total_chunks']} чанков")

    logger.info(f"\n{'='*60}")
    logger.info(f"Индексация завершена. Чанков в БД: {vectordb.get_stats()['total_chunks']}")


if __name__ == "__main__":
    main()