"""
Verbaizer — RAG-вербализация гипотез.
Превращает граничные чанки войда в читаемую гипотезу через YandexGPT Pro.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class Verbaizer:
    """
    Генерирует человекочитаемые гипотезы на основе граничных чанков.
    Использует YandexGPT Pro 5.1 с жёсткими ограничениями:
    - только предоставленные чанки
    - никаких выдуманных фактов
    - structured output
    """

    def __init__(self, yandex_api_client):
        self.api = yandex_api_client

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    def verbaize(self, gap: Dict, target_query: str = "") -> Dict:
        """
        Генерирует гипотезу по одному войду.

        Args:
            gap: словарь с ключами:
                - boundary_chunks: список граничных чанков [{text, source, page}, ...]
                - depth: глубина войда
                - cluster_ids: смежные кластеры
            target_query: целевой запрос (опционально)

        Returns:
            словарь с полями гипотезы
        """
        boundary = gap.get("boundary_chunks", [])
        if len(boundary) < 2:
            return self._empty_hypothesis("Недостаточно граничных чанков")

        # Собираем контекст
        context = self._build_context(boundary, target_query)

        # Промпт
        prompt = self._build_prompt(context, gap.get("depth", 0))

        # Отправляем в YandexGPT Pro
        system = (
            "Ты — научный ассистент в области металлургии и материаловедения. "
            "Твоя задача — формулировать проверяемые научные гипотезы СТРОГО на основе "
            "предоставленных фрагментов документов. "
            "ЗАПРЕЩЕНО выдумывать факты, которых нет в контексте. "
            "Если данных недостаточно — честно напиши 'Недостаточно данных'."
        )

        try:
            raw_response = self.api.generate(
                prompt=prompt,
                system_prompt=system,
                model="pro",
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error(f"Ошибка вербализации: {e}")
            return self._empty_hypothesis(f"Ошибка API: {e}")

        # Парсим ответ
        hypothesis = self._parse_response(raw_response, boundary)
        return hypothesis

    # ------------------------------------------------------------------
    # Сборка контекста
    # ------------------------------------------------------------------

    def _build_context(self, boundary_chunks: List[Dict], target_query: str) -> str:
        """Собирает контекст из граничных чанков для промпта."""
        parts = []

        if target_query:
            parts.append(f"ЦЕЛЕВОЙ ЗАПРОС: {target_query}\n")

        parts.append("ГРАНИЧНЫЕ ДОКУМЕНТЫ (на стыке областей знаний):\n")

        for i, chunk in enumerate(boundary_chunks, start=1):
            source = chunk.get("source", "?")
            page = chunk.get("page", "?")
            text = chunk.get("text", "")
            # Обрезаем слишком длинные чанки
            if len(text) > 800:
                text = text[:800] + "..."
            parts.append(f"[{i}] {source} (стр. {page}):\n{text}\n")

        return "\n".join(parts)

    def _build_prompt(self, context: str, gap_depth: float) -> str:
        """Собирает финальный промпт."""
        return f"""{context}

На основе ТОЛЬКО этих фрагментов предложи научную гипотезу. 
Эти фрагменты находятся на границе РАЗНЫХ областей знаний (глубина разрыва: {gap_depth:.2f}).
Гипотеза должна описывать, что можно исследовать на стыке этих областей.

Верни ответ СТРОГО в формате:

ГИПОТЕЗА: [одно предложение — что именно предлагается проверить]
МЕХАНИЗМ: [как это должно работать физически/химически, на основе фрагментов]
ОБОСНОВАНИЕ: [какие фрагменты на это указывают, с номерами в квадратных скобках]
НОВИЗНА: [почему это не описано в известных источниках]
РИСКИ: [технические риски проверки]
ПЛАН ПРОВЕРКИ: [2-3 шага, критерии успеха/провала]

Если данных недостаточно — напиши: НЕДОСТАТОЧНО ДАННЫХ."""

    # ------------------------------------------------------------------
    # Парсинг ответа
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, boundary: List[Dict]) -> Dict:
        """Парсит структурированный ответ модели."""
        if not raw or "НЕДОСТАТОЧНО ДАННЫХ" in raw.upper():
            return self._empty_hypothesis("Недостаточно данных в граничных чанках")

        fields = {
            "hypothesis": "",
            "mechanism": "",
            "justification": "",
            "novelty": "",
            "risks": "",
            "verification_plan": "",
        }

        current_field = None
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("ГИПОТЕЗА:"):
                current_field = "hypothesis"
                fields[current_field] = line.split(":", 1)[1].strip()
            elif upper.startswith("МЕХАНИЗМ:"):
                current_field = "mechanism"
                fields[current_field] = line.split(":", 1)[1].strip()
            elif upper.startswith("ОБОСНОВАНИЕ:"):
                current_field = "justification"
                fields[current_field] = line.split(":", 1)[1].strip()
            elif upper.startswith("НОВИЗНА:"):
                current_field = "novelty"
                fields[current_field] = line.split(":", 1)[1].strip()
            elif upper.startswith("РИСКИ:"):
                current_field = "risks"
                fields[current_field] = line.split(":", 1)[1].strip()
            elif upper.startswith("ПЛАН ПРОВЕРКИ:"):
                current_field = "verification_plan"
                fields[current_field] = line.split(":", 1)[1].strip()
            elif current_field:
                fields[current_field] += " " + line

        # Добавляем ссылки на источники
        sources = list(set(
            f"{c.get('source', '?')} (стр. {c.get('page', '?')})"
            for c in boundary
        ))

        return {
            "hypothesis": fields["hypothesis"] or "Не удалось извлечь гипотезу",
            "mechanism": fields["mechanism"] or "Не указан",
            "justification": fields["justification"] or "Не указано",
            "novelty": fields["novelty"] or "Не указана",
            "risks": fields["risks"] or "Не оценены",
            "verification_plan": fields["verification_plan"] or "Не предложен",
            "sources": sources,
            "raw_response": raw,
        }

    def _empty_hypothesis(self, reason: str) -> Dict:
        """Заглушка для пустой гипотезы."""
        return {
            "hypothesis": f"Гипотеза не сформирована: {reason}",
            "mechanism": "",
            "justification": "",
            "novelty": "",
            "risks": "",
            "verification_plan": "",
            "sources": [],
            "raw_response": "",
        }