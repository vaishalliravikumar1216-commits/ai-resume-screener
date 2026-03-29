import streamlit as st
import pdfplumber
import re
import nltk
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

st.set_page_config(
    page_title="ResumeIQ — AI Resume Screener",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;700&family=Inter:wght@400;500;600&display=swap');

    .stApp { background-color: #FFFFFF !important; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #FFFFFF; }
    .main { background-color: #FFFFFF !important; }
    .block-container { padding: 2rem 3rem; max-width: 1150px; }

    .navbar {
        display: flex; align-items: center;
        justify-content: space-between;
        padding-bottom: 1.1rem;
        margin-bottom: 2rem;
        border-bottom: 1.5px solid #E2E8F0;
    }
    .navbar-title { font-family: 'Lora', serif; font-size: 1.2rem; font-weight: 700; color: #1E293B; letter-spacing: 0.03em; }
    .navbar-sub { font-size: 0.72rem; color: #94A3B8; letter-spacing: 0.03em; }
    .navbar-badge { background: #F1F5F9; color: #475569; padding: 0.3rem 0.9rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; border: 1px solid #E2E8F0; }

    .hero { background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #93C5FD; border-radius: 8px; padding: 1.75rem 2rem; margin-bottom: 1.75rem; display: flex; align-items: center; justify-content: space-between; gap: 2rem; }
    .hero h1 { font-family: 'Lora', serif; font-size: 1.3rem; font-weight: 700; color: #1E293B; margin-bottom: 0.4rem; }
    .hero p { font-size: 0.84rem; color: #64748B; line-height: 1.65; max-width: 500px; }
    .steps { display: flex; gap: 2rem; flex-shrink: 0; }
    .step { text-align: center; }
    .step-num { width: 30px; height: 30px; border-radius: 50%; background: #EFF6FF; border: 1.5px solid #BFDBFE; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.8rem; color: #3B82F6; margin: 0 auto 0.4rem; }
    .step-lbl { font-size: 0.65rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; }

    .card-title { font-family: 'Lora', serif; font-size: 0.9rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .card-sub { font-size: 0.72rem; color: #94A3B8; margin-bottom: 0.85rem; }

    .metric-card { background: white; border: 1px solid #E2E8F0; border-top: 3px solid #BFDBFE; border-radius: 8px; padding: 1rem; text-align: center; }
    .metric-label { font-size: 0.65rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
    .metric-value { font-family: 'Lora', serif; font-size: 1.8rem; font-weight: 700; color: #1E293B; }
    .metric-value.green { color: #16A34A; }
    .metric-value.red { color: #DC2626; }
    .metric-value.slate { color: #475569; }

    .section-title { font-size: 0.67rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.85rem; }

    .pill-green { display: inline-block; background: #F0FDF4; color: #166534; border: 1px solid #BBF7D0; border-radius: 4px; padding: 0.2rem 0.65rem; font-size: 0.72rem; font-weight: 500; margin: 0.2rem; }
    .pill-red { display: inline-block; background: #FFF1F2; color: #9F1239; border: 1px solid #FECDD3; border-radius: 4px; padding: 0.2rem 0.65rem; font-size: 0.72rem; font-weight: 500; margin: 0.2rem; }

    .feedback-strong { background: #F0FDF4; border-left: 3px solid #22C55E; border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; color: #166534; font-size: 0.84rem; line-height: 1.65; }
    .feedback-moderate { background: #FEFCE8; border-left: 3px solid #EAB308; border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; color: #854D0E; font-size: 0.84rem; line-height: 1.65; }
    .feedback-weak { background: #FFF1F2; border-left: 3px solid #F43F5E; border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; color: #9F1239; font-size: 0.84rem; line-height: 1.65; }

    .stButton > button { background: #1E293B !important; color: white !important; border: none !important; border-radius: 6px !important; padding: 0.8rem 2rem !important; font-size: 0.82rem !important; font-weight: 600 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; width: 100% !important; }
    .stButton > button:hover { background: #334155 !important; }

    .stTextArea textarea { border-radius: 6px !important; border: 1px solid #E2E8F0 !important; font-size: 0.85rem !important; background-color: #FAFAFA !important; color: #1E293B !important; }
    .stTextArea textarea:focus { border-color: #93C5FD !important; box-shadow: 0 0 0 2px rgba(147,197,253,0.2) !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .progress-bar-bg { background: #F1F5F9; border-radius: 3px; height: 8px; margin-top: 6px; }
    .progress-bar-green { background: #86EFAC; height: 8px; border-radius: 3px; }
    .progress-bar-red { background: #FCA5A5; height: 8px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="navbar">
    <div>
        <div class="navbar-title">ResumeIQ</div>
        <div class="navbar-sub">AI-Powered Resume Analysis</div>
    </div>
    <div class="navbar-badge">AI Powered</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div>
        <h1>Match Your Resume to Any Job Description</h1>
        <p>Paste a job description and upload your resume to receive an instant compatibility score, keyword analysis, and tailored recommendations.</p>
    </div>
    <div class="steps">
        <div class="step"><div class="step-num">1</div><div class="step-lbl">Paste JD</div></div>
        <div class="step"><div class="step-num">2</div><div class="step-lbl">Upload CV</div></div>
        <div class="step"><div class="step-num">3</div><div class="step-lbl">Get Score</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Helper functions ─────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def get_keywords(jd_text, resume_text, top_n=20):
    vectorizer = TfidfVectorizer(max_features=60, ngram_range=(1, 2))
    vectorizer.fit([jd_text])
    jd_keywords = set(vectorizer.get_feature_names_out())
    resume_words = set(resume_text.lower().split())
    matched = [k for k in jd_keywords if any(w in resume_words for w in k.split())]
    missing = [k for k in jd_keywords if k not in matched]
    return matched[:top_n], missing[:top_n]

def get_match_score(jd_text, resume_text, matched, missing):
    # TF-IDF cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_text, resume_text])
    tfidf_score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100

    # Keyword match ratio
    total = len(matched) + len(missing)
    keyword_score = (len(matched) / total * 100) if total > 0 else 0

    # Weighted — 40% TF-IDF, 60% keyword match
    final_score = (tfidf_score * 0.4) + (keyword_score * 0.6)
    return round(final_score, 1)

def word_count(text):
    return len(text.split())

# ── Inputs ───────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card-title">Job Description</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Paste the complete job description including responsibilities and requirements</div>', unsafe_allow_html=True)
    jd_input = st.text_area("", height=220, placeholder="Copy and paste the full job description here...", label_visibility="collapsed")
    if jd_input:
        st.caption(f"{word_count(jd_input)} words detected")

with col2:
    st.markdown('<div class="card-title">Your Resume</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Upload your resume in PDF format</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    if resume_file:
        st.success(f"{resume_file.name} uploaded successfully")

st.markdown("<br>", unsafe_allow_html=True)
analyse = st.button("Analyse My Resume")

# ── Results ──────────────────────────────────────────────────
if analyse:
    if not jd_input:
        st.error("Please paste a job description before analysing.")
    elif not resume_file:
        st.error("Please upload your resume in PDF format.")
    else:
        with st.spinner("Analysing your resume..."):
            resume_text = extract_text_from_pdf(resume_file)
            jd_clean = clean_text(jd_input)
            resume_clean = clean_text(resume_text)
            matched, missing = get_keywords(jd_clean, resume_clean)
            score = get_match_score(jd_clean, resume_clean, matched, missing)

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Match Score</div><div class="metric-value">{score}%</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Keywords Matched</div><div class="metric-value green">{len(matched)}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Keywords Missing</div><div class="metric-value red">{len(missing)}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Resume Words</div><div class="metric-value slate">{word_count(resume_text)}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        g1, g2 = st.columns(2, gap="large")

        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={'suffix': '%', 'font': {'size': 38, 'color': '#1E293B', 'family': 'Lora, serif'}},
                title={'text': "Compatibility Score", 'font': {'size': 12, 'color': '#94A3B8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#CBD5E1', 'tickfont': {'size': 10}},
                    'bar': {'color': '#93C5FD', 'thickness': 0.22},
                    'bgcolor': 'white', 'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40], 'color': '#FEE2E2'},
                        {'range': [40, 70], 'color': '#FEF9C3'},
                        {'range': [70, 100], 'color': '#DCFCE7'},
                    ],
                }
            ))
            fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20), paper_bgcolor='white', plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            total = len(matched) + len(missing)
            matched_pct = int((len(matched) / total) * 100) if total > 0 else 0
            missing_pct = 100 - matched_pct
            st.markdown('<div class="section-title">Keyword Coverage</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-bottom:1.25rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:0.78rem;color:#166534;font-weight:600;">Matched Keywords</span>
                    <span style="font-size:0.78rem;color:#166534;">{len(matched)}</span>
                </div>
                <div class="progress-bar-bg"><div class="progress-bar-green" style="width:{matched_pct}%;"></div></div>
            </div>
            <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:0.78rem;color:#9F1239;font-weight:600;">Missing Keywords</span>
                    <span style="font-size:0.78rem;color:#9F1239;">{len(missing)}</span>
                </div>
                <div class="progress-bar-bg"><div class="progress-bar-red" style="width:{missing_pct}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        k1, k2 = st.columns(2, gap="large")
        with k1:
            st.markdown('<div class="section-title">Matched Keywords</div>', unsafe_allow_html=True)
            if matched:
                pills = "".join([f'<span class="pill-green">{k}</span>' for k in matched])
                st.markdown(f'<div style="line-height:2.4">{pills}</div>', unsafe_allow_html=True)
            else:
                st.warning("No strong keyword matches found.")

        with k2:
            st.markdown('<div class="section-title">Missing Keywords</div>', unsafe_allow_html=True)
            if missing:
                pills = "".join([f'<span class="pill-red">{k}</span>' for k in missing])
                st.markdown(f'<div style="line-height:2.4">{pills}</div>', unsafe_allow_html=True)
            else:
                st.success("No major keywords missing.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recommendation</div>', unsafe_allow_html=True)
        if score >= 60:
            st.markdown('<div class="feedback-strong">Strong match. Your resume aligns well with this job description and is likely to pass initial screening filters. Consider incorporating the missing keywords naturally to strengthen your application further.</div>', unsafe_allow_html=True)
        elif score >= 35:
            st.markdown('<div class="feedback-moderate">Moderate match. Your resume has a reasonable fit but could be improved. Review the missing keywords listed above and incorporate them naturally into your experience and skills sections.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="feedback-weak">Low match. Your resume requires significant tailoring for this role. Carefully review the missing keywords and update your resume to reflect those skills and experiences where applicable.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("View Extracted Resume Text"):
            st.text(resume_text[:3000] + "..." if len(resume_text) > 3000 else resume_text)