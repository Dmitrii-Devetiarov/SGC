"""
Gap Finder — ядро системы.
Находит научные пробелы (войды) в пространстве эмбеддингов.

Алгоритм:
1. Загружает чанки и эмбеддинги из ChromaDB
2. Снижает размерность через UMAP (2D)
3. Кластеризует через HDBSCAN
4. Строит 2D-гистограмму плотности
5. Ищет локальные минимумы между кластерами (войды)
6. Для каждого войда находит граничные чанки
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import umap
import hdbscan
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


@dataclass
class Gap:
    """Найденный научный пробел."""
    id: int
    center: Tuple[float, float]        # Центр войда в 2D UMAP
    depth: float                         # Глубина войда (процентиль плотности)
    boundary_chunks: List[Dict]          # Граничные чанки
    cluster_ids: List[int]               # ID смежных кластеров
    density: float                       # Плотность в центре войда


class GapFinder:
    """
    Ищет научные пробелы (антикластеры) в латентном пространстве.
    """

    def __init__(
        self,
        # UMAP
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        # HDBSCAN
        min_cluster_size: int = 10,
        min_samples: int = 5,
        # Gap detection
        grid_size: int = 100,
        smooth_sigma: float = 1.5,
        min_gap_depth: float = 0.05,
        boundary_top_n: int = 10,
        overlap_threshold: float = 0.75
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.grid_size = grid_size
        self.smooth_sigma = smooth_sigma
        self.min_gap_depth = min_gap_depth
        self.boundary_top_n = boundary_top_n
        self.overlap_threshold = overlap_threshold

        self.reducer = None       # UMAP reducer (обученный)
        self.clusterer = None     # HDBSCAN clusterer (обученный)
        self.embeddings_2d = None # 2D проекция
        self.labels = None        # Метки кластеров
        self.chunks = None        # Чанки
        self.embeddings = None    # Исходные эмбеддинги
        self.gaps = None          # Найденные войды

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    def find_gaps(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        target_vector: Optional[List[float]] = None,
    ) -> List[Gap]:
        """
        Главный метод: поиск войдов.

        Args:
            chunks: список чанков (из VectorDB)
            embeddings: список эмбеддингов
            target_vector: опциональный вектор цели для фокусировки поиска

        Returns:
            список Gap
        """
        if len(embeddings) < 20:
            logger.warning("Слишком мало чанков для поиска войдов (нужно >= 20)")
            return []

        self.chunks = chunks
        self.embeddings = np.array(embeddings)
        logger.info(f"Поиск войдов: {len(self.embeddings)} чанков, размерность {self.embeddings.shape[1]}")

        # Шаг 1: UMAP
        self._reduce_dimensions()

        # Шаг 2: HDBSCAN кластеризация
        self._cluster()

        # Шаг 3: Поиск войдов
        all_gaps = self._detect_gaps()

        # Шаг 4: Фильтрация по цели (если задана)
        if target_vector is not None:
            all_gaps = self._filter_by_target(all_gaps, target_vector)

        # Шаг 5: Сортировка по глубине
        all_gaps.sort(key=lambda g: g.depth, reverse=True)

        self.gaps = all_gaps
        
        logger.info(f"Найдено войдов: {len(all_gaps)}")

        
        return all_gaps
    
    def _filter_overlapping_gaps(self, gaps: List[Gap], overlap_threshold: float = 0.75) -> List[Gap]:
        """
        Убирает дублирующиеся войды: если два круга перекрываются > overlap_threshold,
        оставляет более глубокий.
        """
        if not gaps:
            return gaps
        
        import math
        
        # Считаем площадь пересечения двух кругов
        def circle_overlap(r1, r2, d):
            """d — расстояние между центрами, r1, r2 — радиусы."""
            if d >= r1 + r2:
                return 0.0
            if d <= abs(r1 - r2):
                return math.pi * min(r1, r2) ** 2
            
            part1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
            part2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
            part3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
            return part1 + part2 - part3
        
        # Сортируем по глубине (глубокие — вперёд)
        sorted_gaps = sorted(gaps, key=lambda g: g.depth, reverse=True)
        kept = []
        
        for gap in sorted_gaps:
            # Считаем радиус = расстояние до ближайшей кластерной точки
            center = np.array(gap.center)
            cluster_mask = self.labels != -1
            points_clustered = self.embeddings_2d[cluster_mask]
            r = np.min(np.linalg.norm(points_clustered - center, axis=1)) if len(points_clustered) > 0 else 0.2
            area = math.pi * r ** 2
            
            overlaps = False
            for kept_gap in kept:
                kept_center = np.array(kept_gap.center)
                kept_r = np.min(np.linalg.norm(points_clustered - kept_center, axis=1)) if len(points_clustered) > 0 else 0.2
                d = np.linalg.norm(center - kept_center)
                overlap_area = circle_overlap(r, kept_r, d)
                
                if overlap_area / min(area, math.pi * kept_r ** 2) > overlap_threshold:
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(gap)
        
        logger.info(f"После фильтрации перекрытий: {len(kept)} из {len(gaps)} войдов")
        return kept

    # ------------------------------------------------------------------
    # Шаг 1: UMAP
    # ------------------------------------------------------------------

    def _reduce_dimensions(self):
        logger.info("UMAP: снижение размерности...")
        import numpy as np

        # Собираем 2D массив вручную
        clean_embeddings = []
        clean_chunks = []
        for i, emb in enumerate(self.embeddings):
            try:
                arr = np.array(emb, dtype=np.float64)
                if arr.ndim == 1 and len(arr) > 0:
                    clean_embeddings.append(arr)
                    clean_chunks.append(self.chunks[i])
            except Exception:
                pass

        if len(clean_embeddings) < 10:
            logger.error(f"Недостаточно валидных эмбеддингов: {len(clean_embeddings)}")
            return

        self.embeddings = np.array(clean_embeddings)
        self.chunks = clean_chunks
        logger.info(f"Валидных эмбеддингов: {len(self.embeddings)}")

        self.reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=42,
            verbose=False,
        )
        self.embeddings_2d = self.reducer.fit_transform(self.embeddings)
        logger.info(f"UMAP: готово, форма {self.embeddings_2d.shape}")

    # ------------------------------------------------------------------
    # Шаг 2: HDBSCAN
    # ------------------------------------------------------------------

    def _cluster(self):
        """Кластеризует 2D-проекцию."""
        logger.info("HDBSCAN: кластеризация...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
        )
        self.labels = self.clusterer.fit_predict(self.embeddings_2d)

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = sum(1 for l in self.labels if l == -1)
        logger.info(f"HDBSCAN: {n_clusters} кластеров, {n_noise} шумовых точек")

    # ------------------------------------------------------------------
    # Шаг 3: Поиск войдов
    # ------------------------------------------------------------------

    def _detect_gaps(self) -> List[Gap]:
        """
        Поиск войдов геометрическим методом:
        войд = точка в 2D, максимально удалённая от всех кластерных точек.
        """
        logger.info("Поиск войдов геометрическим методом...")

        points = self.embeddings_2d
        cluster_mask = self.labels != -1
        points_clustered = points[cluster_mask]

        if len(points_clustered) < 10:
            logger.warning("Слишком мало кластерных точек")
            return []

        # Строим сетку кандидатов по всему пространству
        x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
        y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1

        grid_x = np.linspace(x_min, x_max, self.grid_size)
        grid_y = np.linspace(y_min, y_max, self.grid_size)
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Для каждой точки сетки — расстояние до ближайшей кластерной точки
        distances = cdist(grid_points, points_clustered)
        min_distances = distances.min(axis=1)

        # Ищем локальные максимумы расстояния (центры войдов)
        dist_map = min_distances.reshape(self.grid_size, self.grid_size)
        local_maxima = self._find_local_maxima(dist_map)

        # Нормируем глубину
        max_dist = min_distances.max() if min_distances.max() > 0 else 1.0

        gaps = []
        for gy, gx in local_maxima:
            center_x = grid_x[gx]
            center_y = grid_y[gy]
            center = np.array([center_x, center_y])

            dist_to_nearest = dist_map[gy, gx]
            depth = dist_to_nearest / max_dist

            # Фильтруем мелкие войды
            if depth < self.min_gap_depth:
                continue

            # Проверяем, что вокруг центра есть точки из ≥2 разных кластеров
            nearby_labels = self._get_nearby_clusters(center, radius=dist_to_nearest * 1.2)
            if len(nearby_labels) < 2:
                continue

            # Граничные чанки
            boundary = self._find_boundary_chunks(center)

            if len(boundary) < 2:
                continue

            gaps.append(Gap(
                id=len(gaps),
                center=(float(center_x), float(center_y)),
                depth=float(depth),
                boundary_chunks=boundary,
                cluster_ids=nearby_labels,
                density=float(dist_to_nearest),
            ))

        # Сортировка по глубине
        gaps.sort(key=lambda g: g.depth, reverse=True)

        # Фильтрация перекрывающихся войдов
        all_gaps = self._filter_overlapping_gaps(gaps, overlap_threshold=0.75)

        logger.info(f"Найдено войдов: {len(all_gaps)}")
        return all_gaps

    def _find_local_maxima(self, arr: np.ndarray) -> List[Tuple[int, int]]:
        """Находит локальные максимумы на 2D-карте расстояний."""
        maxima = []
        h, w = arr.shape
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = arr[y, x]
                neighbors = arr[y-1:y+2, x-1:x+2].flatten()
                neighbors = np.delete(neighbors, 4)  # Убираем центр
                if center > neighbors.max():
                    maxima.append((y, x))
        return maxima

    def _find_local_minima(self, hist: np.ndarray) -> List[Tuple[int, int]]:
        """
        Находит локальные минимумы на 2D-гистограмме.
        Минимум = ячейка, значение которой меньше всех 8 соседей.
        """
        minima = []
        h, w = hist.shape
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = hist[y, x]
                neighbors = hist[y-1:y+2, x-1:x+2].flatten()
                # Исключаем центр из соседей
                neighbors = np.delete(neighbors, 4)  # центр в позиции 4
                if center < neighbors.min():
                    minima.append((y, x))
        return minima

    def _get_nearby_clusters(self, center: np.ndarray, radius: float = 1.0) -> List[int]:
        """
        Возвращает ID кластеров, чьи точки находятся в радиусе от центра.
        """
        distances = cdist([center], self.embeddings_2d)[0]
        nearby_mask = distances < radius
        nearby_labels = set(self.labels[nearby_mask])
        # Убираем шум (-1)
        nearby_labels.discard(-1)
        return sorted(nearby_labels)

    def _find_boundary_chunks(self, center: np.ndarray) -> List[Dict]:
        """
        Находит топ-N чанков, ближайших к центру войда.
        """
        distances = cdist([center], self.embeddings_2d)[0]
        top_indices = np.argsort(distances)[:self.boundary_top_n]

        boundary = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["distance_to_gap"] = float(distances[idx])
            chunk["cluster_id"] = int(self.labels[idx])
            boundary.append(chunk)

        return boundary

    # ------------------------------------------------------------------
    # Шаг 4: Фильтрация по цели
    # ------------------------------------------------------------------

    def _filter_by_target(self, gaps: List[Gap], target_vector: List[float]) -> List[Gap]:
        """
        Оставляет только войды, смежные с кластером, ближайшим к целевому вектору.
        """
        if not gaps or target_vector is None:
            return gaps

        # Эмбеддим цель в 2D через обученный UMAP
        target_2d = self.reducer.transform([target_vector])[0]

        # Находим ближайший кластер к цели
        cluster_centers = {}
        for label in set(self.labels):
            if label == -1:
                continue
            mask = self.labels == label
            cluster_centers[label] = self.embeddings_2d[mask].mean(axis=0)

        if not cluster_centers:
            return gaps

        distances_to_target = {
            label: np.linalg.norm(center - target_2d)
            for label, center in cluster_centers.items()
        }
        target_cluster = min(distances_to_target, key=distances_to_target.get)
        logger.info(f"Целевой кластер: {target_cluster} (расстояние до цели: {distances_to_target[target_cluster]:.3f})")

        # Оставляем войды, смежные с целевым кластером
        filtered = [g for g in gaps if target_cluster in g.cluster_ids]
        logger.info(f"После фильтрации по цели: {len(filtered)} войдов (из {len(gaps)})")
        return filtered

    # ------------------------------------------------------------------
    # Визуализация (для Streamlit)
    # ------------------------------------------------------------------

    def get_visualization_data(self) -> Dict:
        """
        Возвращает данные для построения UMAP-карты с войдами.
        """
        return {
            "points_2d": self.embeddings_2d.tolist() if self.embeddings_2d is not None else [],
            "labels": self.labels.tolist() if self.labels is not None else [],
            "gaps": [
                {
                    "id": g.id,
                    "center": list(g.center),
                    "depth": g.depth,
                    "cluster_ids": g.cluster_ids,
                }
                for g in (self.gaps or [])
            ],
        }