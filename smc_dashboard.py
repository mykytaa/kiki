# trader_notes_full.py
# Полная система, оттюненная:
# - Добавить видео → Конспектировать (2 шага)
# - oEmbed title + превью YouTube (без API-ключа)
# - Quill-редактор (широкие, высокие) в экспандерах
# - Вложения к заметкам (attachments/{note_id})
# - Просмотр заметок блоками (экспандеры с текстом и вложениями)
# - Библиотека видео: фильтры, правки, удаление (с вложениями), экспорт CSV
# - Дашборд: крупные превью карточками + табличный режим
# - Импорт/экспорт CSV (видео и заметки)
# - Фиксы Streamlit: session_state и обязательные submit-кнопки

import os, sqlite3, pathlib, shutil
from datetime import datetime, date
from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st
import requests
from urllib.parse import urlparse, parse_qs
from streamlit_quill import st_quill

APP_TITLE = "📚 Trader Notes — Full"
APP_DIR   = os.path.dirname(__file__)
DB_PATH   = os.path.join(APP_DIR, "notes_hub.db")
HERO_PATH = os.path.join(APP_DIR, "hero.jpg")
ATT_DIR   = os.path.join(APP_DIR, "attachments")
os.makedirs(ATT_DIR, exist_ok=True)

STATUS = ["Не просмотрено","В процессе","Законспектировано"]
IMP    = ["🟢 Низкая","🟡 Средняя","🔴 Высокая"]
SETUPS = ["","BOS","CHOCH","OB","iFVG","FVG","Sweep","Liquidity grab","Break&Retest","Other"]
INSTR  = ["","EURUSD","XAUUSD","GBPUSD","USDJPY","GER40","US500","BTCUSD","ETHUSD"]

# -------------------- Styling --------------------
def css():
    st.markdown("""
    <style>
      .wrap{max-width:1180px;margin:0 auto}
      .card{background:#0f172a0A;border:1px solid #e5e7eb33;border-radius:16px;padding:18px 20px;margin:10px 0 18px;box-shadow:0 2px 14px rgba(0,0,0,0.05)}
      .title-xl{font-size:34px;font-weight:800;margin:6px 0}
      .subtitle{color:#6b7280;font-size:14px;margin:0 0 10px}
      .stButton>button{border-radius:10px;padding:10px 16px}
      /* Высокие редакторы Quill */
      .stQuill .ql-container{min-height:320px !important}
      .stQuill .ql-editor{min-height:280px !important}
    </style>
    """, unsafe_allow_html=True)

# -------------------- YouTube helpers --------------------
def normalize_youtube_url(url: str) -> str:
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        if "youtu.be" in host:
            vid = u.path.lstrip("/")
            return f"https://www.youtube.com/watch?v={vid}"
        if "youtube.com" in host:
            if u.path.startswith("/shorts/"):
                vid = u.path.split("/shorts/")[1].split("/")[0]
                return f"https://www.youtube.com/watch?v={vid}"
            if host.startswith("m."):
                return url.replace("m.youtube.com", "www.youtube.com")
        return url
    except:
        return url

def youtube_id(url: str) -> Optional[str]:
    try:
        u = urlparse(normalize_youtube_url(url))
        if u.hostname and "youtube.com" in u.hostname:
            v = parse_qs(u.query).get("v", [""])[0]
            return v or None
        if u.hostname and "youtu.be" in u.hostname:
            return u.path.lstrip("/") or None
    except:
        pass
    return None

def youtube_thumbnail(url: str) -> Optional[str]:
    vid = youtube_id(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def youtube_title_oembed(url: str) -> Optional[str]:
    try:
        nurl = normalize_youtube_url(url)
        r = requests.get("https://www.youtube.com/oembed",
                         params={"url": nurl, "format": "json"}, timeout=6)
        if r.ok:
            return r.json().get("title")
    except:
        pass
    return None

# -------------------- DB --------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS videos(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      url   TEXT,
      instrument TEXT,
      topic TEXT,
      tags  TEXT,
      status TEXT,
      importance TEXT,
      added_at TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS notes(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      video_id INTEGER NOT NULL,
      watched_on TEXT,
      timecode   TEXT,
      key_point  TEXT,
      setup      TEXT,
      context_md TEXT,  -- храним HTML из Quill
      idea_md    TEXT,  -- храним HTML
      notes_md   TEXT,  -- храним HTML
      created_at TEXT,
      updated_at TEXT,
      FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
    )""")
    return conn

def df_videos(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM videos ORDER BY id DESC", conn)

def df_notes(conn, video_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM notes WHERE video_id=? ORDER BY id DESC", conn, params=[video_id]
    )

def insert_video(conn, row: Dict[str, Any]):
    row.setdefault("added_at", datetime.utcnow().isoformat())
    cols = ",".join(row.keys()); q = ",".join("?" * len(row))
    conn.execute(f"INSERT INTO videos({cols}) VALUES({q})", list(row.values()))
    conn.commit()

def update_video(conn, vid: int, row: Dict[str, Any]):
    if not row: return
    sets = ",".join([f"{k}=?" for k in row.keys()])
    conn.execute(f"UPDATE videos SET {sets} WHERE id=?", [*row.values(), vid])
    conn.commit()

def insert_note(conn, row: Dict[str, Any]) -> int:
    now = datetime.utcnow().isoformat()
    row.setdefault("created_at", now); row.setdefault("updated_at", now)
    cols = ",".join(row.keys()); q = ",".join("?" * len(row))
    cur = conn.cursor()
    cur.execute(f"INSERT INTO notes({cols}) VALUES({q})", list(row.values()))
    conn.commit()
    return cur.lastrowid

def delete_notes_with_attachments(conn, ids: List[int]):
    if not ids: return
    for nid in ids:
        ndir = pathlib.Path(ATT_DIR) / str(int(nid))
        if ndir.exists():
            shutil.rmtree(ndir, ignore_errors=True)
    qmarks = ",".join(["?"] * len(ids))
    conn.execute(f"DELETE FROM notes WHERE id IN ({qmarks})", ids)
    conn.commit()

def delete_videos_with_attachments(conn, ids: List[int]):
    if not ids: return
    qmarks = ",".join(["?"] * len(ids))
    note_ids = pd.read_sql_query(
        f"SELECT id FROM notes WHERE video_id IN ({qmarks})",
        conn, params=ids
    )["id"].tolist()
    delete_notes_with_attachments(conn, note_ids)
    conn.execute(f"DELETE FROM videos WHERE id IN ({qmarks})", ids)
    conn.commit()

# -------------------- Pages --------------------
def page_home():
    css()
    st.title(APP_TITLE)
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.header("🏠 Главная")
    if os.path.exists(HERO_PATH):
        st.image(HERO_PATH, use_column_width=True, caption="Твоя мотивация")
    else:
        st.info("Загрузи мотивационное фото (hero.jpg в папку приложения).")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Разделы")
    st.markdown("- **🎥 Добавить видео** — сохранить ссылку и метки\n"
                "- **📝 Конспект** — выбрать видео, открыть редакторы и прикрепить файлы\n"
                "- **📚 Видео-библиотека** — фильтры, правки, удаление, экспорт\n"
                "- **📊 Дашборд** — крупные превью YouTube / табличный режим\n"
                "- **⬇️⬆️ Импорт/Экспорт** — CSV")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def sidebar_filters_videos(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Фильтры видео")
    q = st.sidebar.text_input("Поиск (название/теги/тема)")
    statuses = st.sidebar.multiselect("Статус", STATUS, default=STATUS)
    imps = st.sidebar.multiselect("Важность", IMP, default=IMP)
    inst = sorted([x for x in df["instrument"].dropna().unique() if x]) if not df.empty else []
    ins = st.sidebar.multiselect("Инструменты", inst, default=inst)
    topics = sorted([x for x in df["topic"].dropna().unique() if x]) if not df.empty else []
    tsel = st.sidebar.multiselect("Темы", topics, default=topics)

    out = df.copy() if not df.empty else df
    if not df.empty:
        if q:
            out = out[
                out["title"].str.contains(q, case=False, na=False) |
                out["tags"].str.contains(q, case=False, na=False) |
                out["topic"].str.contains(q, case=False, na=False)
            ]
        if statuses: out = out[out["status"].isin(statuses)]
        if imps: out = out[out["importance"].isin(imps)]
        if ins: out = out[out["instrument"].isin(ins)]
        if tsel: out = out[out["topic"].isin(tsel)]
    st.sidebar.markdown("---")
    st.sidebar.metric("Видео в выборке", len(out))
    return out

def page_add_video(conn):
    css(); st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.header("🎥 Добавить видео")

    # Инициализация session_state заранее
    st.session_state.setdefault("vid_title", "")
    st.session_state.setdefault("vid_url", "")

    # --- ВНЕ ФОРМЫ: URL и подтягивание названия ---
    st.subheader("Шаг 1: Ссылка на видео")
    st.text_input("Ссылка на YouTube/материал", key="vid_url",
                  placeholder="https://www.youtube.com/watch?v=...")

    c1, c2 = st.columns([1,3])
    with c1:
        if st.button("🎯 Подтянуть название с YouTube"):
            url = (st.session_state.get("vid_url") or "").strip()
            if url:
                t = youtube_title_oembed(url)
                if t:
                    st.session_state["vid_title"] = t  # меняем ДО создания инпута названия
                    st.success("Название обновлено.")
                    st.rerun()  # безопасный перерендер
                else:
                    st.info("Не удалось подтянуть — проверь ссылку.")
            else:
                st.warning("Сначала вставь ссылку.")
    with c2:
        thumb = youtube_thumbnail(st.session_state.get("vid_url", ""))
        if thumb: st.image(thumb, width=320, caption="Превью YouTube")

    st.markdown("---")
    st.subheader("Шаг 2: Детали видео")

    # --- ФОРМА: поля читают уже готовый session_state ---
    with st.form("add_vid_form", clear_on_submit=False):
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            st.text_input("Название*", key="vid_title", placeholder="Название ролика")
            topic = st.text_input("Тема/курс", placeholder="Smart Money / Playlist")
        with col2:
            instrument = st.selectbox("Инструмент", INSTR, index=0)
            status     = st.selectbox("Статус", STATUS, index=0)
            importance = st.selectbox("Важность", IMP, index=1)
        with col3:
            tags = st.text_input("Теги", placeholder="FVG, OB, BOS")
            st.caption("URL указан выше")

        submit_add = st.form_submit_button("Добавить", type="primary")
        if submit_add:
            title_val = (st.session_state.get("vid_title") or "").strip()
            if not title_val:
                st.error("Название обязательно.")
            else:
                insert_video(conn, {
                    "title": title_val,
                    "url":   (st.session_state.get("vid_url") or "").strip(),
                    "instrument": instrument.strip(),
                    "topic": topic.strip(),
                    "tags":  tags.strip(),
                    "status": status,
                    "importance": importance
                })
                st.success("Видео добавлено! Перейди в «📝 Конспект».")

def page_library(conn):
    css(); st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.header("📚 Видео-библиотека")

    df = df_videos(conn)
    df_view = sidebar_filters_videos(df if not df.empty else pd.DataFrame(columns=[
        "id","title","url","instrument","topic","tags","status","importance","added_at"
    ]))

    if df_view.empty:
        st.info("Пока нет видео или не попало под фильтры.")
        st.markdown('</div>', unsafe_allow_html=True); return

    df_edit = df_view.copy()
    df_edit.insert(0, "🗑 удалить?", False)

    edited = st.data_editor(
        df_edit, use_container_width=True, hide_index=True, num_rows="dynamic",
        column_config={
            "status": st.column_config.SelectboxColumn("status", options=STATUS),
            "importance": st.column_config.SelectboxColumn("importance", options=IMP),
            "instrument": st.column_config.SelectboxColumn("instrument", options=INSTR),
            "🗑 удалить?": st.column_config.CheckboxColumn("🗑 удалить?"),
        }, key="lib_editor"
    )

    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        if st.button("💾 Сохранить правки"):
            for _, r in edited.iterrows():
                update_video(conn, int(r["id"]), {
                    "title": str(r["title"]), "url": str(r["url"]),
                    "instrument": str(r["instrument"]), "topic": str(r["topic"]),
                    "tags": str(r["tags"]), "status": str(r["status"]), "importance": str(r["importance"])
                })
            st.success("Сохранено.")
    with c2:
        ids = [int(x) for x in edited.loc[edited.get("🗑 удалить?", False)==True, "id"].tolist()]
        st.button(
            f"🗑 Удалить отмеченные ({len(ids)})",
            disabled=(len(ids)==0),
            on_click=lambda: (delete_videos_with_attachments(conn, ids), st.success(f"Удалено: {len(ids)}"))
        )
    with c3:
        if not df_view.empty and isinstance(df_view.iloc[0].get("url"), str) and df_view.iloc[0]["url"]:
            st.link_button("▶️ Открыть 1-е видео", df_view.iloc[0]["url"])
    with c4:
        csv = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Экспорт CSV", csv, "videos_export.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

def page_notes(conn):
    css(); st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.header("📝 Конспект")

    vids = df_videos(conn)
    if vids.empty:
        st.info("Сначала добавь видео.")
        return

    vids["display"] = vids.apply(lambda r: f'[{r["id"]}] {r["title"]}', axis=1)
    choice = st.selectbox("Выбери видео", vids["display"].tolist())
    video_id = int(choice.split("]")[0][1:])
    row = vids[vids["id"]==video_id].iloc[0]

    head1, head2 = st.columns([3,1])
    with head1:
        st.caption(f"Ссылка: {row['url'] or '—'} | Инструмент: {row['instrument'] or '—'} | Тема: {row['topic'] or '—'} | Теги: {row['tags'] or '—'}")
    with head2:
        if row["url"]:
            st.link_button("▶️ Открыть видео", row["url"], use_container_width=True)

    # ----- Создание записи -----
    with st.form("note_form"):
        c1, c2 = st.columns([1,1])
        with c1:
            watched_on = st.date_input("Дата просмотра", value=date.today())
            timecode   = st.text_input("Таймкод", "00:00")
        with c2:
            key_point  = st.text_input("Ключевая мысль", placeholder="Короткая выжимка")
            setup      = st.selectbox("Сетап (SMC)", SETUPS, index=0)

        st.markdown("#### ✍️ Разделы конспекта (широкие/высокие редакторы)")
        with st.expander("Контекст рынка — открыть/закрыть", expanded=True):
            context_html = st_quill(value="", placeholder="Опиши контекст рынка...",
                                    html=True, key=f"ctx_{video_id}")
        with st.expander("Идея входа — открыть/закрыть", expanded=True):
            idea_html    = st_quill(value="", placeholder="Опиши сетап/логику входа...",
                                    html=True, key=f"idea_{video_id}")
        with st.expander("Заметки/детали — открыть/закрыть", expanded=True):
            notes_html   = st_quill(value="", placeholder="Заметки, выводы, пометки...",
                                    html=True, key=f"notes_{video_id}")

        files = st.file_uploader("Прикрепить файлы (png/jpg/pdf/txt)", 
                                 type=["png","jpg","jpeg","pdf","txt"], accept_multiple_files=True)

        submit_note = st.form_submit_button("➕ Добавить конспект", type="primary")
        if submit_note:
            nid = insert_note(conn, {
                "video_id":  video_id,
                "watched_on": str(watched_on),
                "timecode":   timecode.strip(),
                "key_point":  key_point.strip(),
                "setup":      setup.strip(),
                "context_md": context_html or "",
                "idea_md":    idea_html or "",
                "notes_md":   notes_html or "",
            })
            note_dir = pathlib.Path(ATT_DIR) / str(nid)
            note_dir.mkdir(parents=True, exist_ok=True)
            for f in files or []:
                (note_dir / f.name).write_bytes(f.read())
            if row["status"] == "Не просмотрено":
                update_video(conn, video_id, {"status": "В процессе"})
            st.success("Конспект добавлен!")

            # Кнопки после сохранения
            b1, b2 = st.columns(2)
            with b1:
                if st.form_submit_button("➕ Добавить ещё конспект к этому видео"):
                    for k in (f"ctx_{video_id}", f"idea_{video_id}", f"notes_{video_id}"):
                        if k in st.session_state: del st.session_state[k]
                    st.experimental_rerun()
            with b2:
                if st.form_submit_button("✅ Пометить видео как законспектировано"):
                    update_video(conn, video_id, {"status": "Законспектировано"})
                    st.success("Видео помечено как законспектировано.")
                    st.experimental_rerun()

    # ----- Просмотр как блоки -----
    st.markdown("### 📚 Конспекты по этому видео")
    notes = df_notes(conn, video_id)
    if notes.empty:
        st.info("Пока нет заметок.")
    else:
        for _, r in notes.iterrows():
            header = f"🗓 {r['watched_on'] or '—'}  |  ⏱ {r['timecode'] or '—'}  |  💡 {r['key_point'] or '—'}"
            with st.expander(header, expanded=False):
                st.markdown(f"**Сетап:** `{r['setup'] or '—'}`")
                st.markdown("---")
                if r["context_md"]: st.markdown(r["context_md"], unsafe_allow_html=True)
                if r["idea_md"]:    st.markdown(r["idea_md"], unsafe_allow_html=True)
                if r["notes_md"]:   st.markdown(r["notes_md"], unsafe_allow_html=True)
                note_dir = pathlib.Path(ATT_DIR) / str(int(r["id"]))
                if note_dir.exists():
                    st.markdown("#### 📎 Вложения")
                    imgs, others = [], []
                    for p in sorted(note_dir.glob("*")):
                        if p.suffix.lower() in [".png",".jpg",".jpeg"]:
                            imgs.append(p)
                        else:
                            others.append(p)
                    if imgs:
                        st.image([str(p) for p in imgs], width=320, caption=[p.name for p in imgs])
                    for p in others:
                        with open(p, "rb") as fh:
                            st.download_button(f"Скачать {p.name}", fh.read(), file_name=p.name, key=f"dl_{p.name}_{r['id']}")

def page_dashboard(conn):
    css(); st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.header("📊 Дашборд")

    vids = df_videos(conn)
    total_v = len(vids)
    v_not = int((vids["status"] == "Не просмотрено").sum()) if not vids.empty else 0
    v_inp = int((vids["status"] == "В процессе").sum()) if not vids.empty else 0
    v_done = int((vids["status"] == "Законспектировано").sum()) if not vids.empty else 0

    cols = st.columns(4)
    cols[0].metric("Видео всего", total_v)
    cols[1].metric("Не просмотрено", v_not)
    cols[2].metric("В процессе", v_inp)
    cols[3].metric("Законспектировано", v_done)
    if total_v: st.progress(v_done / total_v)

    st.markdown("---")
    view_mode = st.radio("Режим отображения", ["🖼 Карточки (крупные превью)", "📋 Таблица"], horizontal=True)

    if vids.empty:
        st.info("Нет данных.")
        return

    if view_mode.startswith("🖼"):
        # --- Большие превью карточками ---
        st.markdown("#### Видео (крупные превью)")
        disp = vids.copy()
        n_cols = 3
        cols_grid = st.columns(n_cols)
        for i, (_, r) in enumerate(disp.iterrows()):
            with cols_grid[i % n_cols]:
                thumb = youtube_thumbnail(r["url"]) or ""
                if thumb:
                    st.image(thumb, use_column_width=True)  # КРУПНО
                st.markdown(f"**{r['title']}**")
                if isinstance(r["url"], str) and r["url"]:
                    st.link_button("▶️ Смотреть", r["url"], use_container_width=True)
                st.caption(f"{r['status']} • {r['importance']} • {r['instrument'] or '—'} • {r['topic'] or '—'}")
                st.markdown("---")
    else:
        # --- Табличный вид ---
        disp = vids.copy()
        disp["thumb"] = disp["url"].apply(lambda u: youtube_thumbnail(u) or "")
        disp["title_link"] = disp.apply(lambda r: r["url"] if isinstance(r["url"], str) and r["url"] else "", axis=1)
        disp = disp[["thumb","title","title_link","status","importance","instrument","topic"]]
        st.data_editor(
            disp, use_container_width=True, hide_index=True, disabled=True,
            column_config={
                "thumb": st.column_config.ImageColumn("Превью", help="Миниатюра YouTube"),
                "title": st.column_config.TextColumn("Название"),
                "title_link": st.column_config.LinkColumn("▶️ Смотреть", display_text="Открыть"),
                "status": st.column_config.TextColumn("Статус"),
                "importance": st.column_config.TextColumn("Важность"),
                "instrument": st.column_config.TextColumn("Инструмент"),
                "topic": st.column_config.TextColumn("Тема"),
            },
        )

def page_import_export(conn):
    css(); st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.header("⬇️⬆️ Импорт / Экспорт")

    vids = df_videos(conn)
    st.subheader("Экспорт CSV")
    c1, c2 = st.columns(2)
    with c1:
        if not vids.empty:
            st.download_button("Экспортировать ВИДЕО (CSV)",
                               vids.to_csv(index=False).encode("utf-8"),
                               "videos_export.csv", "text/csv")
        else:
            st.caption("Нет видео для экспорта.")
    all_notes = pd.read_sql_query("SELECT * FROM notes ORDER BY id DESC", conn)
    with c2:
        if not all_notes.empty:
            st.download_button("Экспортировать ЗАМЕТКИ (CSV)",
                               all_notes.to_csv(index=False).encode("utf-8"),
                               "notes_export.csv", "text/csv")
        else:
            st.caption("Нет заметок для экспорта.")

    st.markdown("---")
    st.subheader("Импорт CSV")

    upv = st.file_uploader("Импорт ВИДЕО CSV", type=["csv"], key="upv")
    if upv:
        try:
            df_in = pd.read_csv(upv)
            rename = {}
            for c in df_in.columns:
                lc = c.lower().strip()
                if lc in ["название","title","лекция","video","лекция / видео"]: rename[c] = "title"
                elif lc in ["url","ссылка","link"]: rename[c] = "url"
                elif lc in ["instrument","инструмент"]: rename[c] = "instrument"
                elif lc in ["тема","topic"]: rename[c] = "topic"
                elif lc in ["теги","tags"]: rename[c] = "tags"
                elif lc in ["статус","status"]: rename[c] = "status"
                elif lc in ["важность","importance"]: rename[c] = "importance"
            df_in = df_in.rename(columns=rename)
            need = {"title","url","instrument","topic","tags","status","importance"}
            for m in need:
                if m not in df_in.columns: df_in[m] = ""
            for _, r in df_in.iterrows():
                insert_video(conn, {k: str(r[k]) for k in need})
            st.success(f"Импортировано видео: {len(df_in)}")
        except Exception as e:
            st.error(f"Ошибка импорта видео: {e}")

    upn = st.file_uploader("Импорт ЗАМЕТОК CSV (обязателен столбец video_id)", type=["csv"], key="upn")
    if upn:
        try:
            df_in = pd.read_csv(upn)
            need = {"video_id","watched_on","timecode","key_point","setup","context_md","idea_md","notes_md"}
            for m in need:
                if m not in df_in.columns: df_in[m] = ""
            for _, r in df_in.iterrows():
                insert_note(conn, {k: (int(r[k]) if k=="video_id" and str(r[k]).isdigit() else str(r[k])) for k in need})
            st.success(f"Импортировано заметок: {len(df_in)}")
        except Exception as e:
            st.error(f"Ошибка импорта заметок: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- App --------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    conn = get_conn()
    menu = st.sidebar.radio("Навигация", [
        "🏠 Главная",
        "🎥 Добавить видео",
        "📝 Конспект",
        "📚 Видео-библиотека",
        "📊 Дашборд",
        "⬇️⬆️ Импорт/Экспорт",
    ])
    if menu == "🏠 Главная":             page_home()
    elif menu == "🎥 Добавить видео":    page_add_video(conn)
    elif menu == "📝 Конспект":          page_notes(conn)
    elif menu == "📚 Видео-библиотека":  page_library(conn)
    elif menu == "📊 Дашборд":           page_dashboard(conn)
    else:                                page_import_export(conn)

if __name__ == "__main__":
    main()
