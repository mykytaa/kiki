import os, sqlite3, pathlib, shutil
from datetime import datetime, date
from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st
import requests
from urllib.parse import urlparse, parse_qs

APP_TITLE = "📚 Trader Notes (Full Cloud Version)"
APP_DIR   = os.path.dirname(__file__)
DB_PATH   = os.path.join(APP_DIR, "notes_hub.db")
HERO_PATH = os.path.join(APP_DIR, "hero.jpg")
ATT_DIR   = os.path.join(APP_DIR, "attachments")
os.makedirs(ATT_DIR, exist_ok=True)

STATUS = ["Не просмотрено","В процессе","Законспектировано"]
IMP    = ["🟢 Низкая","🟡 Средняя","🔴 Высокая"]
SETUPS = ["","BOS","CHOCH","OB","iFVG","FVG","Sweep","Liquidity grab","Break&Retest","Other"]
INSTR  = ["","EURUSD","XAUUSD","GBPUSD","USDJPY","GER40","US500","BTCUSD","ETHUSD"]

# -------------------- CSS --------------------
def css():
    st.markdown("""
    <style>
      .wrap{max-width:1180px;margin:0 auto}
      .card{background:#0f172a0A;border:1px solid #e5e7eb33;border-radius:16px;
            padding:18px 20px;margin:10px 0 18px;box-shadow:0 2px 14px rgba(0,0,0,0.05)}
      .stTextArea textarea {min-height: 220px !important}
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
      context_md TEXT,
      idea_md    TEXT,
      notes_md   TEXT,
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
    st.header("🏠 Главная")
    if os.path.exists(HERO_PATH):
        st.image(HERO_PATH, use_column_width=True, caption="Твоя мотивация")
    else:
        st.info("Загрузи hero.jpg в папку приложения.")
    st.markdown("Добро пожаловать трейдер!")

def page_add_video(conn):
    css(); st.header("🎥 Добавить видео")

    st.session_state.setdefault("vid_title", "")
    st.session_state.setdefault("vid_url", "")

    st.text_input("Ссылка на YouTube", key="vid_url")
    if st.button("🎯 Подтянуть название"):
        url = (st.session_state.get("vid_url") or "").strip()
        if url:
            t = youtube_title_oembed(url)
            if t:
                st.session_state["vid_title"] = t
                st.success("Название подтянуто.")
                st.rerun()

    thumb = youtube_thumbnail(st.session_state.get("vid_url", ""))
    if thumb: st.image(thumb, width=320)

    with st.form("add_vid_form"):
        title = st.text_input("Название*", key="vid_title")
        topic = st.text_input("Тема/курс")
        instrument = st.selectbox("Инструмент", INSTR, index=0)
        status     = st.selectbox("Статус", STATUS, index=0)
        importance = st.selectbox("Важность", IMP, index=1)
        tags = st.text_input("Теги")
        submit_add = st.form_submit_button("Добавить")
        if submit_add:
            if not title.strip():
                st.error("Название обязательно.")
            else:
                insert_video(conn, {
                    "title": title.strip(),
                    "url":   (st.session_state.get("vid_url") or "").strip(),
                    "instrument": instrument.strip(),
                    "topic": topic.strip(),
                    "tags":  tags.strip(),
                    "status": status,
                    "importance": importance
                })
                st.success("Видео добавлено!")

def page_library(conn):
    css(); st.header("📚 Видео-библиотека")
    df = df_videos(conn)
    if df.empty:
        st.info("Пока нет видео.")
        return
    df_edit = df.copy()
    df_edit.insert(0, "🗑 удалить?", False)
    edited = st.data_editor(
        df_edit, use_container_width=True, hide_index=True, num_rows="dynamic",
        column_config={
            "status": st.column_config.SelectboxColumn("status", options=STATUS),
            "importance": st.column_config.SelectboxColumn("importance", options=IMP),
            "instrument": st.column_config.SelectboxColumn("instrument", options=INSTR),
            "🗑 удалить?": st.column_config.CheckboxColumn("🗑 удалить?"),
        }
    )
    if st.button("💾 Сохранить правки"):
        for _, r in edited.iterrows():
            update_video(conn, int(r["id"]), {
                "title": str(r["title"]), "url": str(r["url"]),
                "instrument": str(r["instrument"]), "topic": str(r["topic"]),
                "tags": str(r["tags"]), "status": str(r["status"]), "importance": str(r["importance"])
            })
        st.success("Сохранено.")
    ids = [int(x) for x in edited.loc[edited.get("🗑 удалить?", False)==True, "id"].tolist()]
    if ids and st.button("🗑 Удалить отмеченные"):
        delete_videos_with_attachments(conn, ids)
        st.success(f"Удалено: {len(ids)}")

def page_notes(conn):
    css(); st.header("📝 Конспект")

    vids = df_videos(conn)
    if vids.empty:
        st.info("Сначала добавь видео.")
        return
    vids["display"] = vids.apply(lambda r: f'[{r["id"]}] {r["title"]}', axis=1)
    choice = st.selectbox("Выбери видео", vids["display"].tolist())
    video_id = int(choice.split("]")[0][1:])
    row = vids[vids["id"]==video_id].iloc[0]

    st.caption(f"Ссылка: {row['url'] or '—'} | Инструмент: {row['instrument'] or '—'}")

    with st.form("note_form"):
        watched_on = st.date_input("Дата просмотра", value=date.today())
        timecode   = st.text_input("Таймкод", "00:00")
        key_point  = st.text_input("Ключевая мысль")
        setup      = st.selectbox("Сетап", SETUPS, index=0)
        context_html = st.text_area("Контекст рынка", key=f"ctx_{video_id}")
        idea_html    = st.text_area("Идея входа", key=f"idea_{video_id}")
        notes_html   = st.text_area("Заметки/детали", key=f"notes_{video_id}")
        files = st.file_uploader("Вложения", type=["png","jpg","jpeg","pdf","txt"], accept_multiple_files=True)
        submit_note = st.form_submit_button("💾 Сохранить конспект")
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
            st.success("Конспект добавлен!")
            if row["status"] == "Не просмотрено":
                update_video(conn, video_id, {"status": "В процессе"})

    st.subheader("📚 Конспекты")
    notes = df_notes(conn, video_id)
    if notes.empty:
        st.info("Нет заметок.")
    else:
        for _, r in notes.iterrows():
            with st.expander(f"{r['watched_on']} | {r['timecode']} | {r['key_point']}"):
                st.markdown(f"**Сетап:** {r['setup']}")
                st.write(r['context_md'])
                st.write(r['idea_md'])
                st.write(r['notes_md'])
                note_dir = pathlib.Path(ATT_DIR) / str(r['id'])
                if note_dir.exists():
                    for p in note_dir.glob("*"):
                        st.write("📎", p.name)

def page_dashboard(conn):
    css(); st.header("📊 Дашборд")
    vids = df_videos(conn)
    if vids.empty:
        st.info("Нет данных.")
        return
    total_v = len(vids)
    v_not = int((vids["status"] == "Не просмотрено").sum())
    v_inp = int((vids["status"] == "В процессе").sum())
    v_done = int((vids["status"] == "Законспектировано").sum())
    cols = st.columns(4)
    cols[0].metric("Видео всего", total_v)
    cols[1].metric("Не просмотрено", v_not)
    cols[2].metric("В процессе", v_inp)
    cols[3].metric("Законспектировано", v_done)
    if total_v: st.progress(v_done / total_v)
    n_cols = 3
    cols_grid = st.columns(n_cols)
    for i, (_, r) in enumerate(vids.iterrows()):
        with cols_grid[i % n_cols]:
            thumb = youtube_thumbnail(r["url"]) or ""
            if thumb: st.image(thumb, use_column_width=True)
            st.markdown(f"**{r['title']}**")
            if r["url"]: st.link_button("▶️ Смотреть", r["url"], use_container_width=True)
            st.caption(f"{r['status']} • {r['importance']} • {r['instrument'] or '—'}")

def page_import_export(conn):
    css(); st.header("⬇️⬆️ Импорт / Экспорт")
    vids = df_videos(conn)
    all_notes = pd.read_sql_query("SELECT * FROM notes ORDER BY id DESC", conn)
    if not vids.empty:
        st.download_button("⬇️ Экспорт видео", vids.to_csv(index=False).encode("utf-8"),
                           "videos_export.csv", "text/csv")
    if not all_notes.empty:
        st.download_button("⬇️ Экспорт заметок", all_notes.to_csv(index=False).encode("utf-8"),
                           "notes_export.csv", "text/csv")
    upv = st.file_uploader("Импорт видео CSV", type=["csv"], key="upv")
    if upv:
        df_in = pd.read_csv(upv)
        for _, r in df_in.iterrows():
            insert_video(conn, {
                "title": str(r.get("title","")),
                "url": str(r.get("url","")),
                "instrument": str(r.get("instrument","")),
                "topic": str(r.get("topic","")),
                "tags": str(r.get("tags","")),
                "status": str(r.get("status","")),
                "importance": str(r.get("importance",""))
            })
        st.success("Видео импортированы.")
    upn = st.file_uploader("Импорт заметок CSV", type=["csv"], key="upn")
    if upn:
        df_in = pd.read_csv(upn)
        for _, r in df_in.iterrows():
            insert_note(conn, {
                "video_id": int(r.get("video_id",0)),
                "watched_on": str(r.get("watched_on","")),
                "timecode": str(r.get("timecode","")),
                "key_point": str(r.get("key_point","")),
                "setup": str(r.get("setup","")),
                "context_md": str(r.get("context_md","")),
                "idea_md": str(r.get("idea_md","")),
                "notes_md": str(r.get("notes_md",""))
            })
        st.success("Заметки импортированы.")

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
        "⬇️⬆️ Импорт / Экспорт"
    ])
    if menu == "🏠 Главная": page_home()
    elif menu == "🎥 Добавить видео": page_add_video(conn)
    elif menu == "📝 Конспект": page_notes(conn)
    elif menu == "📚 Видео-библиотека": page_library(conn)
    elif menu == "📊 Дашборд": page_dashboard(conn)
    else: page_import_export(conn)

if __name__ == "__main__":
    main()
