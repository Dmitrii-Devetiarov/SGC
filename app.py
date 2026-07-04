"""
SGC — Scientific Gap Cartography
Streamlit-интерфейс для генерации научных гипотез.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
import chromadb
import subprocess
import os

from utils import load_config
from yandex_api import YandexAIClient
from verbaizer import Verbaizer
from gap_finder import GapFinder

# ------------------------------------------------------------------
# Инициализация
# ------------------------------------------------------------------

@st.cache_resource
def init_app():
    config = load_config()
    api = YandexAIClient(
        api_key=config["yandex_api"]["api_key"],
        folder_id=config["yandex_api"]["folder_id"],
        embedding_dim=config["yandex_api"]["embedding_dim"],
    )
    return config, api

config, api = init_app()


def get_vectordb():
    """Подключается к ChromaDB через HTTP (WSL fix)."""
    client = chromadb.HttpClient(host="localhost", port=8000)
    try:
        collection = client.get_collection("documents")
    except Exception:
        collection = client.create_collection("documents", metadata={"hnsw:space": "cosine"})
    return client, collection


client, collection = get_vectordb()

# ------------------------------------------------------------------
# Боковая панель
# ------------------------------------------------------------------

st.sidebar.title("🧪 SGC")
st.sidebar.markdown("*Scientific Gap Cartography*")
st.sidebar.markdown("Интерпретируемый движок научных гипотез")

# Режим
mode = st.sidebar.radio("Режим работы", ["🎯 По запросу", "🔍 Автогенерация"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚️ Параметры поиска")

n_gaps = st.sidebar.slider("Количество гипотез", 1, 10, 3)
min_gap_depth = st.sidebar.slider(
    "Глубина разрыва", 0.01, 0.30, 0.03, 0.01,
    help="Глубже — неожиданные, рискованные гипотезы на стыке далёких областей. Мельче — осторожные, близкие к известному."
)
min_cluster_size = st.sidebar.slider(
    "Дробность тем", 3, 20, 5,
    help="Меньше — больше узких тем и детальных гипотез. Больше — только крупные научные направления."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💎 Ценность гипотезы")
st.sidebar.info(
    "Гипотезы ранжируются по **глубине разрыва** — "
    "насколько далеко центр войда от известных точек знания. "
    "Чем глубже разрыв, тем более неожиданной и потенциально прорывной может быть гипотеза."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Управление базой")
with st.sidebar.expander("🔄 Индексация документов"):
    data_dir = st.text_input("Папка с документами", "./data/input")
    st.caption("Поддерживаются: PDF, TXT, CSV, XLSX, XLS")
    if st.button("🚀 Запустить индексацию", use_container_width=True):
        with st.spinner("Индексация..."):
            result = subprocess.run(
                [sys.executable, "src/indexer.py"],
                capture_output=True, text=True, timeout=600,
                env={**os.environ, "DATA_INPUT_DIR": data_dir},
            )
        if result.returncode == 0:
            st.success("Индексация завершена.")
            client, collection = get_vectordb()
        else:
            st.error(f"Ошибка:\n```\n{result.stderr[-500:]}\n```")

# ------------------------------------------------------------------
# Основной экран
# ------------------------------------------------------------------

st.title("🔬 Scientific Gap Cartography")
st.markdown(
    "Поиск научных пробелов и генерация проверяемых гипотез. "
    "**Каждая гипотеза трассируется до исходных документов.**"
)

# Статистика
count = collection.count()
if count == 0:
    st.warning(
        "⚠️ База знаний пуста. "
        "Загрузите документы через панель «Индексация документов» слева, "
        "либо распакуйте демо-базу: `tar -xzf db_demo.tar.gz`"
    )
    st.stop()

st.info(f"📚 Чанков в базе: **{count}**")

# Запрос
query = ""
if "По запросу" in mode:
    query = st.text_input(
        "🎯 Целевой запрос",
        placeholder="Например: повысить производительность флотации золота на 15%",
    )

can_run = "Автогенерация" in mode or ("По запросу" in mode and query.strip())
run = st.button("🚀 Сгенерировать гипотезы", type="primary", disabled=not can_run, use_container_width=True)

if not run:
    st.markdown("---")
    st.markdown("### 💡 Как это работает")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Векторное облако**\n\nДокументы → эмбеддинги → облако точек в латентном пространстве")
    c2.markdown("**2. Поиск антикластеров**\n\nUMAP + HDBSCAN → кластеры знаний → структурные разрывы между ними")
    c3.markdown("**3. Вербализация**\n\nRAG на граничных документах → проверяемая гипотеза с обоснованием и источниками")
    st.stop()

# ------------------------------------------------------------------
# Пайплайн
# ------------------------------------------------------------------

# Загружаем данные
with st.spinner("📡 Загрузка данных..."):
    result = collection.get(include=["documents", "embeddings", "metadatas"])
    chunks = []
    embeddings = []
    for doc, emb, meta in zip(result["documents"], result["embeddings"], result["metadatas"]):
        chunks.append({"text": doc, "source": meta.get("source", ""), "page": meta.get("page", 0)})
        embeddings.append(emb)

if len(chunks) < 20:
    st.error("⚠️ Недостаточно данных. Проиндексируйте хотя бы 20 чанков.")
    st.stop()

with st.spinner("🔍 Поиск научных пробелов..."):
    query_embedding = api.embed_query(query) if query else None

    finder = GapFinder(
        n_neighbors=config["umap"]["n_neighbors"],
        min_dist=config["umap"]["min_dist"],
        metric=config["umap"]["metric"],
        min_cluster_size=min_cluster_size,
        min_samples=config["hdbscan"]["min_samples"],
        grid_size=config["gaps"]["grid_size"],
        smooth_sigma=config["gaps"]["smooth_sigma"],
        min_gap_depth=min_gap_depth,
        boundary_top_n=config["gaps"]["boundary_top_n"],
    )

    gaps = finder.find_gaps(chunks, [np.array(e) for e in embeddings], target_vector=query_embedding)

if not gaps:
    st.warning("🔍 Не найдено значимых пробелов. Уменьшите глубину разрыва или измените запрос.")
    st.stop()

st.success(f"🎯 Найдено пробелов: **{len(gaps)}**. Показаны топ-{min(n_gaps, len(gaps))}.")

# ------------------------------------------------------------------
# Вербализация
# ------------------------------------------------------------------

verbaizer = Verbaizer(api)
top_gaps = gaps[:n_gaps]

st.markdown("---")
st.header("📋 Гипотезы")

for i, gap in enumerate(top_gaps, start=1):
    with st.spinner(f"📝 Формулировка гипотезы #{i}..."):
        result = verbaizer.verbaize({
            "boundary_chunks": gap.boundary_chunks,
            "depth": gap.depth,
            "cluster_ids": gap.cluster_ids,
        }, target_query=query)

    with st.expander(
        f"**Гипотеза #{i}** | Глубина разрыва: {gap.depth:.3f} | Кластеры: {gap.cluster_ids}",
        expanded=(i == 1),
    ):
        st.markdown(f"### {result.get('hypothesis', '—')}")

        c1, c2 = st.columns(2)
        c1.markdown("**⚙️ Механизм:**"); c1.info(result.get("mechanism", "—"))
        c1.markdown("**📖 Обоснование:**"); c1.info(result.get("justification", "—"))
        c2.markdown("**🆕 Новизна:**"); c2.info(result.get("novelty", "—"))
        c2.markdown("**⚠️ Риски:**"); c2.info(result.get("risks", "—"))

        st.markdown("**🧪 План проверки:**")
        st.success(result.get("verification_plan", "—"))

        if result.get("sources"):
            st.markdown("**📄 Источники:**")
            for s in result["sources"]:
                st.caption(f"• {s}")

# ------------------------------------------------------------------
# Визуализация
# ------------------------------------------------------------------

st.markdown("---")
st.header("🗺️ Карта знаний")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

cmask = finder.labels != -1
pc = finder.embeddings_2d[cmask]
pn = finder.embeddings_2d[~cmask] if (~cmask).any() else np.empty((0, 2))

for label in sorted(set(finder.labels)):
    mask = finder.labels == label
    if label == -1:
        ax1.scatter(pn[:, 0], pn[:, 1], c='gray', s=6, alpha=0.3, edgecolors='none')
    else:
        ax1.scatter(finder.embeddings_2d[mask, 0], finder.embeddings_2d[mask, 1],
                    s=10, alpha=0.5, edgecolors='none')

if query_embedding is not None:
    q2d = finder.reducer.transform([query_embedding])[0]
    ax1.scatter(q2d[0], q2d[1], c='red', marker='*', s=350, edgecolors='black', linewidths=1.5, label='🎯 Запрос', zorder=10)

for gap in gaps:
    c = np.array(gap.center)
    r = np.min(np.linalg.norm(pc - c, axis=1)) if len(pc) > 0 else 0.2
    ax1.add_patch(Circle(gap.center, r, fill=False, edgecolor='red', linewidth=1.5, alpha=0.8, zorder=10))

ax1.set_title(f"Кластеры и войды ({len(chunks)} чанков, {len(gaps)} войдов)")
ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2")
if query: ax1.legend(fontsize=8)
ax1.grid(alpha=0.15); ax1.set_aspect('equal')

if len(pc) > 0:
    hist, xe, ye = np.histogram2d(pc[:, 0], pc[:, 1], bins=80,
        range=[[pc[:, 0].min()-1, pc[:, 0].max()+1], [pc[:, 1].min()-1, pc[:, 1].max()+1]])
    hist_s = gaussian_filter(hist, sigma=1.5)
    im = ax2.imshow(hist_s.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                    cmap='hot', aspect='auto', interpolation='bilinear')

    if query_embedding is not None:
        ax2.scatter(q2d[0], q2d[1], c='cyan', marker='*', s=350, edgecolors='white', linewidths=1.5, zorder=10)

    for gap in gaps:
        c = np.array(gap.center)
        r = np.min(np.linalg.norm(pc - c, axis=1)) if len(pc) > 0 else 0.2
        ax2.add_patch(Circle(gap.center, r, fill=False, edgecolor='cyan', linewidth=1.2, linestyle='--', alpha=0.8, zorder=10))

    plt.colorbar(im, ax=ax2, label="Плотность", shrink=0.8)

ax2.set_title("Тепловая карта плотности")
ax2.set_xlabel("UMAP 1"); ax2.set_ylabel("UMAP 2")
plt.tight_layout()
st.pyplot(fig)

# ------------------------------------------------------------------
# Экспорт
# ------------------------------------------------------------------

# Вместо блока экспорта в app.py:

st.markdown("---")
st.header("📥 Экспорт")

export_dir = st.text_input("Папка для сохранения", "./output", help="Куда сохранить файлы с гипотезами")

c1, c2 = st.columns(2)

if c1.button("📄 Скачать DOCX", use_container_width=True):
    from exporter import export_to_docx
    os.makedirs(export_dir, exist_ok=True)
    hyps = []
    for i, gap in enumerate(top_gaps, start=1):
        r = verbaizer.verbaize({
            "boundary_chunks": gap.boundary_chunks,
            "depth": gap.depth,
            "cluster_ids": gap.cluster_ids,
        }, target_query=query)
        r["gap_id"] = i
        r["gap_depth"] = gap.depth
        hyps.append(r)
    filepath = os.path.join(export_dir, "sgc_hypotheses.docx")
    export_to_docx(hyps, query, filepath)
    st.success(f"Сохранено: {filepath}")

if c2.button("📊 Скачать JSON", use_container_width=True):
    import json
    os.makedirs(export_dir, exist_ok=True)
    hyps = []
    for i, gap in enumerate(top_gaps, start=1):
        r = verbaizer.verbaize({
            "boundary_chunks": gap.boundary_chunks,
            "depth": gap.depth,
            "cluster_ids": gap.cluster_ids,
        }, target_query=query)
        r["gap_id"] = i
        r["gap_depth"] = gap.depth
        r.pop("raw_response", None)
        hyps.append(r)
    filepath = os.path.join(export_dir, "sgc_hypotheses.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(hyps, f, ensure_ascii=False, indent=2)
    st.success(f"Сохранено: {filepath}")
st.markdown("---")
st.caption("🧪 SGC — Scientific Gap Cartography | Хакатон 2026| Все гипотезы трассируются до исходных документов")