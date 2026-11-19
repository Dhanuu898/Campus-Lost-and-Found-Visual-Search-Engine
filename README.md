# Campus-Lost-and-Found-Visual-Search-Engine

import streamlit as st
import sqlite3, os, io, uuid, hashlib
import numpy as np
from PIL import Image
from datetime import datetime
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Campus Lost & Found", page_icon="🔎", layout="wide")

os.makedirs("images", exist_ok=True)
conn = sqlite3.connect("lostfound.db", check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS items(
    id TEXT PRIMARY KEY, kind TEXT, title TEXT, description TEXT,
    contact TEXT, location TEXT, path TEXT, emb BLOB, created TEXT,
        status TEXT, img_hash TEXT)""")
columns = [row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()]
if "img_hash" not in columns:
    conn.execute("ALTER TABLE items ADD COLUMN img_hash TEXT")
conn.commit()

@st.cache_resource
def load_model():
    return SentenceTransformer("clip-ViT-B-32")

st.title("🔎 Campus Lost & Found – Visual Search")
dashboard = st.container()  # filled at the bottom so the numbers are always fresh

with st.spinner("Loading AI model (first time takes a few minutes)..."):
    model = load_model()

# ---------- Sidebar ----------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Match threshold (%)", 40, 95, 65) / 100
st.sidebar.caption("Higher = stricter. Matches below this are hidden.")

# ---------- Helpers ----------
def embed_image(img):
    return model.encode(img, normalize_embeddings=True).astype(np.float32)

def embed_text(t):
    return model.encode(t, normalize_embeddings=True).astype(np.float32)

def save_item(kind, title, desc, contact, loc, img, emb, img_hash):
    iid = str(uuid.uuid4())[:8]
    path = f"images/{iid}.png"
    img.save(path)
    conn.execute("INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (iid, kind, title, desc, contact, loc, path, emb.tobytes(),
         datetime.now().strftime("%d %b %Y %H:%M"), "open", img_hash))
    conn.execute("CREATE INDEX IF NOT EXISTS idx_items_kind_status ON items(kind, status)")
    conn.commit()

def search(query_emb, kind, min_score, top_k=5):
    rows = conn.execute(
        "SELECT id,title,description,contact,location,path,emb,created FROM items "
        "WHERE kind=? AND status='open'", (kind,)).fetchall()
    results = []
    for r in rows:
        score = float(np.dot(query_emb, np.frombuffer(r[6], dtype=np.float32)))
        if score >= min_score:
            results.append((score, r))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def label_for(pct):
    if pct >= 85:
        return "🟢 Strong match"
    if pct >= 75:
        return "🟡 Possible match"
    return "🟠 Uncertain - please verify"

def show_results(results, ctx, text_mode=False):
    if not results:
        st.info("No matches above the threshold. Try lowering it in the sidebar.")
        return
    for score, r in results:
        c1, c2 = st.columns([1, 3])
        c1.image(r[5], use_container_width=True)
        if text_mode:
            c2.markdown(f"**{r[1]}**  \n🔤 Text relevance score: **{score:.3f}**")
        else:
            pct = max(0, min(100, int(score * 100)))
            tag = label_for(pct)
            c2.markdown(f"**{r[1]}**  \n{tag}  \nImage similarity: **{score:.3f}**")
            c2.progress(pct / 100)
        if r[2]:
            c2.write(r[2])
        c2.caption(f"📍 {r[4]} | 🕒 {r[7]} | 📞 {r[3]}")
        if c2.button("Mark as returned", key=f"ret_{ctx}_{r[0]}"):
            conn.execute("UPDATE items SET status='returned' WHERE id=?", (r[0],))
            conn.commit()
            st.rerun()
        st.divider()

def count(query, params=()):
    return conn.execute(query, params).fetchone()[0]

def pending_matches():
    lost = conn.execute("SELECT emb FROM items WHERE kind='lost' AND status='open'").fetchall()
    found = conn.execute("SELECT emb FROM items WHERE kind='found' AND status='open'").fetchall()
    found_embs = [np.frombuffer(f[0], dtype=np.float32) for f in found]
    n = 0
    for l in lost:
        le = np.frombuffer(l[0], dtype=np.float32)
        if any(float(np.dot(le, fe)) >= threshold for fe in found_embs):
            n += 1
    return n

def item_form(kind, place_label):
    other = "lost" if kind == "found" else "found"
    with st.form(f"form_{kind}", clear_on_submit=True):
        f = st.file_uploader("Photo (required)", type=["jpg", "jpeg", "png"])
        title = st.text_input("Item name *")
        desc = st.text_area("Details (optional)")
        loc = st.text_input(place_label + " *")
        contact = st.text_input("Your contact *")
        submitted = st.form_submit_button("Post & search")

    if submitted:
        missing = []
        if f is None: missing.append("photo")
        if not title.strip(): missing.append("item name")
        if not loc.strip(): missing.append("location")
        if not contact.strip(): missing.append("contact")
        if missing:
            st.error("Please provide: " + ", ".join(missing))
        else:
            data = f.getvalue()
            h = hashlib.md5(data + kind.encode()).hexdigest()
            if count("SELECT COUNT(*) FROM items WHERE img_hash=? AND kind=?", (h, kind)):
                st.warning("This photo was already posted. Duplicate ignored.")
            else:
                try:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    emb = embed_image(img)
                    save_item(kind, title.strip(), desc.strip(),
                              contact.strip(), loc.strip(), img, emb, h)
                    st.session_state[f"q_{kind}"] = emb
                    st.success("Posted successfully.")
                except Exception as e:
                    st.error(f"Unable to process this image. Please upload another image. Details: {e}")

    q = st.session_state.get(f"q_{kind}")
    if q is not None:
        st.subheader(f"Matching {other.upper()} posts")
        show_results(search(q, other, threshold), ctx=kind)



with st.expander("ℹ️ How does the matching work?"):
    st.markdown("""
1. **CLIP** converts each photo into a numerical feature vector (an embedding).
2. Each embedding is **normalized** to unit length, so cosine similarity compares the *direction* of the feature vectors.
3. **Cosine similarity** between two vectors measures how alike the items look.
4. Results are **ranked** from highest to lowest similarity, and anything below the threshold is hidden.

The score is a similarity measure, not a probability. The owner should always verify the item.
""")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📸 I found something", "😢 I lost something", "🔤 Search by text", "📋 All posts"])

with tab1:
    st.subheader("Upload a photo of the found item")
    item_form("found", "Where did you find it?")

with tab2:
    st.subheader("Upload a photo of your lost item")
    item_form("lost", "Where did you lose it?")

with tab3:
    q = st.text_input("Describe it (e.g. 'black calculator', 'red water bottle')")
    kind = st.radio("Search in", ["found", "lost"], horizontal=True)
    st.caption("Text scores are raw similarity scores, not accuracy percentages. Results are ranked by relative similarity.")
    if q.strip():
        show_results(search(embed_text(q), kind, 0.18), ctx="text", text_mode=True)

with tab4:
    for k in ["lost", "found"]:
        st.subheader(k.capitalize())
        rows = conn.execute(
            "SELECT title,location,path,status FROM items WHERE kind=?", (k,)).fetchall()
        if not rows:
            st.caption("Nothing posted yet.")
        cols = st.columns(4)
        for i, r in enumerate(rows):
            with cols[i % 4]:
                st.image(r[2], caption=f"{r[0]} - {r[1]} ({r[3]})")

# ---------- Dashboard (rendered last, shown at top) ----------
with dashboard:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lost items", count("SELECT COUNT(*) FROM items WHERE kind='lost'"))
    c2.metric("Found items", count("SELECT COUNT(*) FROM items WHERE kind='found'"))
    c3.metric("Returned", count("SELECT COUNT(*) FROM items WHERE status='returned'"))
    c4.metric("Lost items with a possible match", pending_matches())
