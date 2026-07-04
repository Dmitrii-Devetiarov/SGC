"""
Парсер документов: извлекает текст из цифровых PDF, TXT, CSV, Excel.
Для сканированных PDF возвращает изображения страниц в base64 (pending OCR).
"""

import re
import logging
from pathlib import Path
from typing import List, Dict
from io import BytesIO

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Парсер документов. Для цифровых страниц возвращает текст,
    для сканов — изображение в base64 с флагом pending_ocr=True.
    """

    def __init__(self, ocr_dpi: int = 200, ocr_max_width: int = 1500):
        self.ocr_dpi = ocr_dpi
        self.ocr_max_width = ocr_max_width

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------

    def parse(self, file_path: str) -> List[Dict]:
        """
        Возвращает список словарей:
        [
            {
                "text": str | None,
                "page": int,
                "source": str,
                "type": str,
                "pending_ocr": bool,
                "image_base64": str | None,
            },
            ...
        ]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(path)
        elif suffix == ".txt":
            return self._parse_txt(path)
        elif suffix == ".csv":
            return self._parse_csv(path)
        elif suffix in (".xlsx", ".xls"):
            return self._parse_excel(path)
        else:
            logger.warning(f"Неподдерживаемый формат: {suffix}")
            return []

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def _parse_pdf(self, path: Path) -> List[Dict]:
        """Обрабатывает PDF постранично."""
        results = []
        doc = fitz.open(str(path))

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            
            if self._is_scan_page(page):
                # Большое изображение на всю страницу → скан
                image_base64 = self._page_to_base64(page)
                results.append({
                    "text": None,
                    "page": page_num,
                    "source": path.name,
                    "type": "pdf",
                    "pending_ocr": True,
                    "image_base64": image_base64,
                })
            elif self._needs_ocr(text):
                # Текст пустой или кракозябры → OCR
                image_base64 = self._page_to_base64(page)
                results.append({
                    "text": None,
                    "page": page_num,
                    "source": path.name,
                    "type": "pdf",
                    "pending_ocr": True,
                    "image_base64": image_base64,
                })
            else:
                # Цифровой текст
                results.append({
                    "text": text,
                    "page": page_num,
                    "source": path.name,
                    "type": "pdf",
                    "pending_ocr": False,
                    "image_base64": None,
                })

        doc.close()
        digital = sum(1 for r in results if not r["pending_ocr"])
        scans = sum(1 for r in results if r["pending_ocr"])
        logger.info(f"PDF обработан: {path.name}, страниц {len(results)} (цифровых: {digital}, сканов: {scans})")
        return results

    def _is_scan_page(self, page) -> bool:
        """
        Определяет, является ли страница сканом.
        Скан = есть изображение, занимающее >70% площади страницы.
        """
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        if page_area == 0:
            return False

        for img in page.get_images(full=True):
            img_rects = page.get_image_rects(img[0])
            for rect in img_rects:
                img_area = rect.width * rect.height
                if img_area / page_area > 0.7:
                    return True
        return False

    def _needs_ocr(self, text: str) -> bool:
        """Определяет, нужен ли OCR: мало текста или кракозябры."""
        if len(text) < 50:
            return True
        if self._is_garbled(text):
            return True
        return False

    def _is_garbled(self, text: str) -> bool:
        """Эвристика: слишком много нечитаемых символов → кракозябры."""
        normal = len(re.findall(r'[а-яА-ЯёЁa-zA-Z0-9\s.,;:!?\-()\[\]%°+=/]', text))
        total = len(text)
        return total == 0 or (normal / total) < 0.7

    def _page_to_base64(self, page) -> str:
        """Рендерит страницу в PNG, сжимает, возвращает base64."""
        import base64

        pix = page.get_pixmap(dpi=self.ocr_dpi)
        img_bytes = pix.tobytes("png")
        img_bytes = self._compress_image(img_bytes)
        return base64.b64encode(img_bytes).decode("utf-8")

    def _compress_image(self, img_bytes: bytes) -> bytes:
        """Сжимает изображение до max_width, сохраняя пропорции."""
        try:
            img = Image.open(BytesIO(img_bytes))
            if img.width > self.ocr_max_width:
                ratio = self.ocr_max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((self.ocr_max_width, new_height), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        except Exception:
            return img_bytes

    # ------------------------------------------------------------------
    # TXT
    # ------------------------------------------------------------------

    def _parse_txt(self, path: Path) -> List[Dict]:
        results = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    results.append({
                        "text": line,
                        "page": line_num,
                        "source": path.name,
                        "type": "txt",
                        "pending_ocr": False,
                        "image_base64": None,
                    })
        logger.info(f"TXT обработан: {path.name}, строк: {len(results)}")
        return results

    # ------------------------------------------------------------------
    # CSV / Excel
    # ------------------------------------------------------------------

    def _parse_csv(self, path: Path) -> List[Dict]:
        try:
            df = pd.read_csv(path)
            return self._table_to_dicts(df, path.name, "csv")
        except Exception as e:
            logger.error(f"Ошибка чтения CSV {path.name}: {e}")
            return []

    def _parse_excel(self, path: Path) -> List[Dict]:
        try:
            sheets = pd.read_excel(path, sheet_name=None)
            results = []
            for sheet_name, df in sheets.items():
                results.extend(self._table_to_dicts(df, f"{path.name}[{sheet_name}]", "excel"))
            return results
        except Exception as e:
            logger.error(f"Ошибка чтения Excel {path.name}: {e}")
            return []

    def _table_to_dicts(self, df: pd.DataFrame, source: str, file_type: str) -> List[Dict]:
        df = df.fillna("")
        results = []
        for row_num, (_, row) in enumerate(df.iterrows(), start=1):
            parts = []
            for col in df.columns:
                value = str(row[col]).strip()
                if value:
                    parts.append(f"{col}: {value}")
            if parts:
                results.append({
                    "text": "; ".join(parts),
                    "page": row_num,
                    "source": source,
                    "type": file_type,
                    "pending_ocr": False,
                    "image_base64": None,
                })
        logger.info(f"Таблица обработана: {source}, строк: {len(results)}")
        return results