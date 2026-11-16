import html

import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

from rag_core import answer_from_documents, answer_with_local_model, set_llm_model
from vector_db import get_retrievers

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "mxbai-embed-large"

SOURCE_DATA = "Business Data (CSV)"
SOURCE_DOCS = "Business Documents (PDF)"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Invyra | Business Analytics",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("selected_source", "CSV")
st.session_state.setdefault("pending_prompt", None)
st.session_state.setdefault("light_mode", False)
st.session_state.setdefault("llm_fallback", True)
st.session_state.setdefault("data", None)

# ---------------------------------------------------------------------------
# Theme (dark / light)
# ---------------------------------------------------------------------------
DARK = {
    "scheme": "dark",
    "bg": "#070A18", "sidebar": "#0A0E22",
    "surface": "rgba(20, 26, 56, 0.72)", "surface2": "rgba(30, 38, 78, 0.90)",
    "line": "rgba(126, 146, 255, 0.22)", "grid": "rgba(126, 146, 255, 0.05)",
    "text": "#EAEEFF", "muted": "#98A2D4",
    "a1": "#22E6FF", "a2": "#9170FF",
    "glow1": "rgba(34, 230, 255, 0.15)", "glow2": "rgba(145, 112, 255, 0.22)",
    "user": "rgba(145, 112, 255, 0.16)", "user_line": "rgba(145, 112, 255, 0.45)",
    "code": "#0C1230", "on_accent": "#070A18", "ok": "#3DFFB0", "chip": "rgba(34, 230, 255, 0.10)",
}
LIGHT = {
    "scheme": "light",
    "bg": "#F1F4FF", "sidebar": "#FFFFFF",
    "surface": "rgba(255, 255, 255, 0.86)", "surface2": "#FFFFFF",
    "line": "rgba(70, 82, 190, 0.20)", "grid": "rgba(70, 82, 190, 0.07)",
    "text": "#141A3D", "muted": "#566096",
    "a1": "#0087AD", "a2": "#6A3EF0",
    "glow1": "rgba(0, 135, 173, 0.12)", "glow2": "rgba(106, 62, 240, 0.14)",
    "user": "rgba(106, 62, 240, 0.08)", "user_line": "rgba(106, 62, 240, 0.34)",
    "code": "#E9EDFF", "on_accent": "#FFFFFF", "ok": "#0A9B64", "chip": "rgba(0, 135, 173, 0.09)",
}
tokens = LIGHT if st.session_state.light_mode else DARK
root_vars = ":root{color-scheme:" + tokens["scheme"] + ";" + "".join(
    f"--{k.replace('_', '-')}:{v};" for k, v in tokens.items() if k != "scheme") + "}"

STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;500;600&family=Orbitron:wght@600;800&display=swap');

html, body, .stApp { font-family: 'Exo 2', sans-serif; color: var(--text); }
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(820px 460px at 88% -8%, var(--glow2), transparent 62%),
        radial-gradient(700px 420px at 0% 105%, var(--glow1), transparent 62%),
        linear-gradient(var(--grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid) 1px, transparent 1px);
    background-size: auto, auto, 48px 48px, 48px 48px;
    background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }
.block-container { max-width: 880px; padding-top: 1.8rem; padding-bottom: 7rem; }

.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp span,
[data-testid="stMarkdownContainer"] { color: var(--text); }
h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif; letter-spacing: 0.01em; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }
[data-testid="stSpinner"], [data-testid="stSpinner"] p { color: var(--muted); }
.stApp pre { background: var(--code); border: 1px solid var(--line); border-radius: 10px; }
.stApp code { color: var(--a1); background: var(--code); border-radius: 6px; }

/* ---------- Brand ---------- */
.brand-row { display: flex; align-items: center; gap: 12px; margin-bottom: 1.3rem; }
.brand-orb {
    width: 32px; height: 32px; border-radius: 50%; flex: none;
    background: radial-gradient(circle at 32% 30%, #FFFFFF 0%, var(--a1) 28%, var(--a2) 72%, #1A1450 100%);
    box-shadow: 0 0 18px var(--a2), 0 0 40px var(--glow1);
    animation: orb-breathe 5s ease-in-out infinite;
}
@keyframes orb-breathe {
    0%, 100% { box-shadow: 0 0 14px var(--a2), 0 0 32px var(--glow1); }
    50%      { box-shadow: 0 0 26px var(--a2), 0 0 64px var(--glow2); }
}
.brand-name { font-family: 'Orbitron', sans-serif; font-weight: 800; font-size: 1.15rem; letter-spacing: 0.14em; }
.pill {
    margin-left: auto; display: inline-flex; align-items: center; gap: 8px;
    padding: 5px 12px; border-radius: 999px; font-size: 0.8rem;
    background: var(--surface); border: 1px solid var(--line); color: var(--muted);
}
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.dot.on  { background: var(--ok); box-shadow: 0 0 8px var(--ok); }
.dot.off { background: var(--muted); opacity: .55; }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden; background: var(--surface);
    -webkit-backdrop-filter: blur(14px); backdrop-filter: blur(14px);
    border: 1px solid var(--line); border-radius: 20px; padding: 2rem 2rem 1.8rem; margin-bottom: 1rem;
}
.hero::before {
    content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--a1), var(--a2), transparent);
    background-size: 200% 100%; animation: sweep 6s linear infinite;
}
@keyframes sweep { from { background-position: 200% 0; } to { background-position: -200% 0; } }
.hero-title {
    font-family: 'Orbitron', sans-serif; font-weight: 800;
    font-size: clamp(1.4rem, 4vw, 2.05rem); line-height: 1.25; margin: 0 0 0.8rem 0;
    background: linear-gradient(100deg, var(--a2) 0%, var(--a1) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: var(--muted); font-size: 1.02rem; max-width: 620px; line-height: 1.6; }

.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 1.2rem; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; }
.tile-label { color: var(--muted); font-size: 0.78rem; margin-bottom: 4px; }
.tile-value { font-family: 'Orbitron', sans-serif; font-weight: 600; font-size: 0.9rem; }
@media (max-width: 640px) { .tiles { grid-template-columns: 1fr; } }
.ask-title { font-family: 'Orbitron', sans-serif; font-size: 0.82rem; font-weight: 600; margin: 0.2rem 0 0.6rem; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--line); }
.side-heading { font-family: 'Orbitron', sans-serif; font-weight: 600; font-size: 0.76rem; letter-spacing: 0.08em; margin: 0.7rem 0 0.6rem; }
.side-note { color: var(--muted); font-size: 0.82rem; line-height: 1.5; }

[data-testid="stSidebar"] div[role="radiogroup"] { gap: 8px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: var(--surface2); border: 1px solid var(--line); border-radius: 12px;
    padding: 12px 14px; transition: border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { border-color: var(--a1); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    border-color: var(--a1); box-shadow: 0 0 0 1px var(--a1), 0 0 18px var(--glow1);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }

.status-row {
    display: flex; align-items: center; gap: 10px; padding: 10px 12px; margin-bottom: 8px;
    background: var(--surface2); border: 1px solid var(--line); border-radius: 12px; font-size: 0.88rem;
}
.status-count { margin-left: auto; color: var(--muted); font-size: 0.8rem; }

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--surface2); color: var(--text); border: 1px solid var(--line);
    border-radius: 12px; padding: 0.7rem 0.9rem; font-family: 'Exo 2', sans-serif; font-weight: 500;
    min-height: 3.6rem; line-height: 1.3;
    transition: border-color .15s ease, box-shadow .15s ease, transform .1s ease;
}
.stButton > button:hover { border-color: var(--a1); box-shadow: 0 0 16px var(--glow1); color: var(--text); }
.stButton > button:active { transform: scale(0.98); }
.stButton > button:focus-visible { outline: 2px solid var(--a1); outline-offset: 2px; }
.stButton > button p { color: inherit; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: var(--surface); -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
    border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.15rem; margin-bottom: 0.9rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--user); border-color: var(--user-line);
}
[data-testid="stChatMessage"] p { line-height: 1.65; }
[data-testid="stChatMessageAvatarAssistant"] {
    background: radial-gradient(circle at 32% 30%, #FFFFFF 0%, var(--a1) 30%, var(--a2) 85%); color: var(--on-accent);
}
[data-testid="stChatMessageAvatarUser"] { background: var(--a2); color: #FFFFFF; }

.tag {
    display: inline-block; margin: 0.7rem 0.4rem 0 0; padding: 3px 10px; border-radius: 999px;
    font-size: 0.78rem; border: 1px solid var(--line); color: var(--muted);
}
.tag.llm  { border-color: var(--a2); color: var(--a2); }
.tag.docs { border-color: var(--a1); color: var(--a1); }
.chip {
    display: inline-block; margin: 0.45rem 0.4rem 0 0; padding: 3px 10px; border-radius: 8px;
    font-size: 0.78rem; background: var(--chip); border: 1px solid var(--line); color: var(--text);
}

[data-testid="stBottom"], [data-testid="stBottom"] > div { background: transparent; }
[data-testid="stChatInput"] { background: var(--surface2); border: 1px solid var(--line); border-radius: 16px; }
[data-testid="stChatInput"]:focus-within { border-color: var(--a1); box-shadow: 0 0 0 3px var(--glow1), 0 0 24px var(--glow1); }
[data-testid="stChatInput"] textarea { color: var(--text); background: transparent; font-family: 'Exo 2', sans-serif; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); }

hr { border-color: var(--line); }
div[data-testid="stAlert"] { border-radius: 12px; }

@media (prefers-reduced-motion: reduce) {
    .brand-orb, .hero::before { animation: none; }
    .stButton > button, [data-testid="stSidebar"] div[role="radiogroup"] > label { transition: none; }
}
"""
st.markdown(f"<style>{root_vars}{STATIC_CSS}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Models (thinking off, larger context so retrieved passages are not cut off)
# ---------------------------------------------------------------------------
def load_ollama_models():
    try:
        llm_instance = OllamaLLM(model=LLM_MODEL, reasoning=False, num_ctx=8192, temperature=0.2)
    except Exception:
        # Older langchain-ollama versions don't know `reasoning`; the prompt adds /no_think instead.
        llm_instance = OllamaLLM(model=LLM_MODEL, num_ctx=8192, temperature=0.2)
    embeddings_instance = OllamaEmbeddings(model=EMBED_MODEL)
    return llm_instance, embeddings_instance


llm, embeddings = load_ollama_models()
set_llm_model(llm)


def load_data(force=False):
    """Loads (and indexes new files for) both sources. Runs once per session, or on Refresh."""
    if force or st.session_state.data is None:
        with st.spinner("Invyra is reading your business files. New files can take a minute."):
            try:
                csv_r, pdf_r, stats = get_retrievers(embeddings)
                st.session_state.data = {"csv": csv_r, "pdf": pdf_r, "stats": stats}
            except Exception as e:
                st.error(
                    f"Invyra could not start: {e}\n\n"
                    f"Make sure Ollama is running (`ollama serve`) and both models are installed: "
                    f"`ollama pull {LLM_MODEL}` and `ollama pull {EMBED_MODEL}`."
                )
                st.stop()
    return st.session_state.data


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------
def respond(question, source, retriever):
    result = answer_from_documents(question, retriever)
    if result["found"]:
        return {"content": result["answer"], "origin": "docs", "sources": result["sources"]}

    if st.session_state.llm_fallback:
        general = answer_with_local_model(question)
        if general:
            return {"content": general, "origin": "llm", "sources": []}

    folder = "local_csv" if source == "CSV" else "local_pdf"
    return {
        "content": f"I couldn't find that in your business files. Try rephrasing the question, "
                   f"or add more files to `{folder}` and press Refresh data.",
        "origin": None, "sources": [],
    }


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return
        if msg.get("origin") == "llm":
            st.markdown(f'<span class="tag llm">Not found in your business files. Answered by local model {html.escape(LLM_MODEL)}</span>',
                        unsafe_allow_html=True)
        elif msg.get("origin") == "docs":
            chips = "".join(f'<span class="chip">{html.escape(s)}</span>' for s in msg.get("sources", []))
            st.markdown(f'<span class="tag docs">From your business files</span>{chips}', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
data = load_data()
stats = data["stats"]

with st.sidebar:
    st.markdown('<div class="brand-row"><div class="brand-orb"></div><div class="brand-name">INVYRA</div></div>',
                unsafe_allow_html=True)
    st.toggle("Light mode", key="light_mode")

    st.markdown('<div class="side-heading">DATA SOURCE</div>', unsafe_allow_html=True)
    source_choice = st.radio(
        "Choose data source", (SOURCE_DATA, SOURCE_DOCS),
        index=0 if st.session_state.selected_source == "CSV" else 1,
        key="source_radio", label_visibility="collapsed",
    )
    st.session_state.selected_source = "CSV" if source_choice == SOURCE_DATA else "PDF"
    retriever = data[st.session_state.selected_source.lower()]

    st.toggle(
        "Use local AI when my files have no answer",
        key="llm_fallback",
        help="Applies to both sources. If your CSV or PDF files don't contain the answer, the question goes to the local model.",
    )

    st.markdown("---")
    st.markdown('<div class="side-heading">KNOWLEDGE BASE</div>', unsafe_allow_html=True)

    def status_row(label, count, unit):
        cls = "on" if count else "off"
        text = f"{count:,} {unit}" if count else "empty"
        return (f'<div class="status-row"><span class="dot {cls}"></span><span>{label}</span>'
                f'<span class="status-count">{text}</span></div>')

    st.markdown(status_row("Business data", stats["csv"], "records")
                + status_row("Business documents", stats["pdf"], "passages"), unsafe_allow_html=True)
    if not stats["csv"]:
        st.caption("Add CSV files to `local_csv`, then press Refresh data.")
    if not stats["pdf"]:
        st.caption("Add PDF files to `local_pdf`, then press Refresh data.")

    if st.button("Refresh data", use_container_width=True):
        st.session_state.data = None
        st.rerun()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

# ---------------------------------------------------------------------------
# Hero (until the first question)
# ---------------------------------------------------------------------------
is_data = st.session_state.selected_source == "CSV"
source_label = "your business data" if is_data else "your business documents"
ready = int(stats["csv"] > 0) + int(stats["pdf"] > 0)

if not st.session_state.messages:
    st.markdown(
        '<div class="brand-row"><div class="brand-orb"></div><div class="brand-name">INVYRA</div>'
        f'<div class="pill"><span class="dot on"></span>Local AI online</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><div class="hero-title">Ask Invyra about your business.</div>'
        f'<div class="hero-sub">Invyra searches {source_label} on this computer and answers in plain language, '
        'with the files it used. Nothing leaves your machine.</div></div>',
        unsafe_allow_html=True,
    )

    mode = "Files, then local AI" if st.session_state.llm_fallback else "Your files only"
    st.markdown(
        '<div class="tiles">'
        f'<div class="tile"><div class="tile-label">Sources ready</div><div class="tile-value">{ready} of 2</div></div>'
        f'<div class="tile"><div class="tile-label">Answer mode</div><div class="tile-value">{mode}</div></div>'
        f'<div class="tile"><div class="tile-label">Local model</div><div class="tile-value">{html.escape(LLM_MODEL)}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ask-title">Try asking</div>', unsafe_allow_html=True)
    suggestions = {
        "CSV": ["What are the main themes in the data?", "What stands out as the biggest issue?", "Which areas are performing best?"],
        "PDF": ["Summarise the key points of these documents", "What risks or concerns are raised?", "List the important figures and deadlines"],
    }[st.session_state.selected_source]
    for i, (col, text) in enumerate(zip(st.columns(len(suggestions)), suggestions)):
        if col.button(text, key=f"suggest_{i}", use_container_width=True):
            st.session_state.pending_prompt = text
            st.rerun()

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    render_message(message)

prompt = st.chat_input("Ask about your business data..." if is_data else "Ask about your business documents...")
if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.spinner(f"Invyra is searching {source_label}..."):
        try:
            reply = respond(prompt, st.session_state.selected_source, retriever)
        except Exception as e:
            st.error(f"Something went wrong: {e}. Check that Ollama is running and `{LLM_MODEL}` is installed.")
            reply = {"content": "I couldn't answer that. Please try again.", "origin": None, "sources": []}

    assistant_msg = {"role": "assistant", **reply}
    render_message(assistant_msg)
    st.session_state.messages.append(assistant_msg)