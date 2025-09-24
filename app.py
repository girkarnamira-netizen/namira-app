# streamlit_internship_recommender.py
# Single-file Streamlit app: Internship Recommendation + beautiful CSS cards
# Run: pip install streamlit pandas scikit-learn && streamlit run streamlit_internship_recommender.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

st.set_page_config(page_title="Internship Recommender", layout="wide")

# ---------- Custom CSS (beautiful card layout) ----------
css = """
<style>
:root{--bg:#0f1724;--card:#0b1220;--accent:#7c3aed;--muted:#9aa4b2;--glass:rgba(255,255,255,0.03)}
body {background: linear-gradient(180deg, #071226 0%, #081a2a 100%); color: #e6eef8}
.header {display:flex;align-items:center;gap:18px}
.logo {width:72px;height:72px;border-radius:16px;background:linear-gradient(135deg,var(--accent),#06b6d4);display:flex;align-items:center;justify-content:center;font-weight:700;color:white;font-size:26px}
.title {font-size:28px;font-weight:700}
.subtitle {color:var(--muted);margin-top:6px}
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px;margin-top:18px}
.card{background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));border-radius:14px;padding:16px;box-shadow:0 6px 18px rgba(2,6,23,0.6);border:1px solid rgba(255,255,255,0.03)}
.card .row{display:flex;align-items:center;justify-content:space-between}
.company{font-weight:700;font-size:18px}
.role{font-size:15px;color:#cfe9ff;margin-top:6px}
.tags{margin-top:10px;display:flex;flex-wrap:wrap;gap:8px}
.tag{background:var(--glass);padding:6px 10px;border-radius:999px;font-size:13px;color:var(--muted);border:1px solid rgba(255,255,255,0.02)}
.relevance{font-weight:700;color:var(--accent)}
.explain{color:var(--muted);font-size:13px;margin-top:8px}
.filter-box{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));padding:14px;border-radius:12px;border:1px solid rgba(255,255,255,0.02)}
.footer{color:var(--muted);font-size:13px;margin-top:18px}
.btn{display:inline-block;padding:8px 14px;border-radius:10px;background:var(--accent);color:white;text-decoration:none}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# ---------- Header ----------
header_html = """
<div class='header'>
  <div class='logo'>IR</div>
  <div>
    <div class='title'>Internship Recommender</div>
    <div class='subtitle'>Personalized internship matches based on your skills and preferences — beautiful, fast, explainable.</div>
  </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ---------- Sidebar: User inputs ----------
with st.sidebar.form('profile'):
    st.markdown("## Build your profile")
    name = st.text_input('Name')
    skills_text = st.text_area('Enter skills (comma separated) or paste resume text', placeholder='e.g. Python, Machine Learning, SQL, pandas, communication')
    location_pref = st.selectbox('Location preference', ['Any', 'Remote', 'On-site', 'Hybrid'])
    domain_pref = st.multiselect('Interested domains', ['Data Science','Web Dev','Marketing','Design','Finance','Research'], default=['Data Science'])
    min_stipend = st.slider('Minimum stipend (₹)', 0, 50000, 0, step=500)
    submitted = st.form_submit_button('Update profile')

if not skills_text:
    st.info('Tip: paste your resume text or list of skills in the sidebar to get personalized recommendations.')

# ---------- Sample internship data (you can replace with CSV/upload) ----------
sample_data = [
    { 'company': 'DeepLearn Labs', 'role': 'Machine Learning Intern', 'location':'Remote', 'stipend':10000,
      'description':'Work on machine learning models, Python, pandas, scikit-learn, model evaluation, data cleaning, SQL.'},
    { 'company': 'WebWave Studio', 'role': 'Frontend Intern', 'location':'On-site', 'stipend':8000,
      'description':'HTML, CSS, JavaScript, React, UI/UX design, responsive web design.'},
    { 'company': 'MarketMinds', 'role': 'Digital Marketing Intern', 'location':'Hybrid', 'stipend':5000,
      'description':'Social media campaigns, content creation, SEO, analytics, communication skills.'},
    { 'company': 'FinEdge', 'role': 'Data Analyst Intern', 'location':'Remote', 'stipend':9000,
      'description':'SQL, Excel, Python, data visualization, business analytics, stakeholder communication.'},
    { 'company': 'CreativeKids', 'role': 'Educational Content Creator', 'location':'On-site', 'stipend':6000,
      'description':'Create educational cartoons for children, scriptwriting, basic animation tools, creativity.'}
]

df = pd.DataFrame(sample_data)

# Allow user to upload their own CSV of internships
st.markdown('---')
col_upload, col_hint = st.columns([1,3])
with col_upload:
    uploaded = st.file_uploader('Upload internships CSV (optional)', type=['csv'])
with col_hint:
    st.markdown('If you have a CSV with columns: company, role, location, stipend, description — upload it to replace the sample listings.')

if uploaded:
    try:
        user_df = pd.read_csv(uploaded)
        # basic safety: require description column
        if 'description' in user_df.columns:
            df = user_df
            st.success('Uploaded internships loaded')
        else:
            st.error('CSV must include a `description` column. Using sample data.')
    except Exception as e:
        st.error('Could not parse uploaded CSV — using sample data.')

# ---------- Simple recommender (TF-IDF + cosine similarity) ----------
@st.cache_data
def build_matrix(descriptions):
    vec = TfidfVectorizer(stop_words='english', max_features=1000)
    mat = vec.fit_transform(descriptions)
    return vec, mat

vec, mat = build_matrix(df['description'].astype(str).tolist())

# Build user vector
user_profile = skills_text if skills_text else ''
user_vec = vec.transform([user_profile]) if user_profile.strip() else None

# compute similarity
if user_vec is not None and user_profile.strip():
    sim = cosine_similarity(user_vec, mat).flatten()
    df['score'] = sim
else:
    df['score'] = 0.0

# Apply filters
filtered = df.copy()
if location_pref and location_pref != 'Any':
    filtered = filtered[filtered['location'].str.contains(location_pref, case=False, na=False)]
if min_stipend:
    filtered = filtered[filtered['stipend'] >= min_stipend]
if domain_pref:
    domain_mask = filtered['description'].apply(lambda s: any(d.lower() in s.lower() for d in domain_pref))
    # keep ones that match at least one domain OR keep all if none matched
    if domain_mask.any():
        filtered = filtered[domain_mask]

# sort by score
filtered = filtered.sort_values('score', ascending=False)

# ---------- Display results in beautiful cards ----------
st.markdown('<div style="margin-top:10px"></div>', unsafe_allow_html=True)
results_html = """
<div style='display:flex;gap:18px'>
  <div style='flex:1'>
    <div class='filter-box'>
      <h3 style='margin:0'>Your profile</h3>
      <div style='color:var(--muted);margin-top:8px'>Name: {name}</div>
      <div style='color:var(--muted);margin-top:6px'>Skills: {skills}</div>
      <div style='color:var(--muted);margin-top:6px'>Location pref: {loc}</div>
      <div style='color:var(--muted);margin-top:6px'>Domains: {doms}</div>
      <div style='margin-top:10px'><a class='btn' href='#' onclick="window.location.reload();">Refresh</a></div>
    </div>
  </div>
  <div style='flex:3'>
    <div class='card-grid'>
""".format(name=name or '—', skills=skills_text or '—', loc=location_pref, doms=','.join(domain_pref) if domain_pref else 'Any')

# construct cards
cards = ''
for _, row in filtered.iterrows():
    tags = ''
    # extract simple tags from description (first 5 words that look like tech)
    for t in row['description'].split(',')[:5]:
        tags += f"<div class='tag'>{t.strip()}</div>"
    explain = ''
    if user_profile.strip():
        explain = f"Matches your skills: top keywords overlap (score {row['score']:.2f})"
    else:
        explain = 'Set your skills to get personalized scores.'
    card = f"""
    <div class='card'>
      <div class='row'>
        <div>
          <div class='company'>{row['company']}</div>
          <div class='role'>{row['role']}</div>
        </div>
        <div style='text-align:right'>
          <div class='relevance'>{row['stipend'] and '₹'+str(row['stipend'])}</div>
          <div style='font-size:12px;color:var(--muted)'>{row['location']}</div>
        </div>
      </div>
      <div class='tags'>{tags}</div>
      <div class='explain'>{explain}</div>
    </div>
    """
    cards += card

if cards.strip() == '':
    cards = "<div class='card'><div class='company'>No matches found</div><div class='explain'>Try changing filters or enter your skills to get recommendations.</div></div>"

results_html += cards + "</div></div></div>"

components.html(results_html, height=600)

st.markdown("""
<div class='footer'>Made with ❤️ — pick, filter, and click to apply. Want advanced matching with embeddings, resume parsing, or a database backend? Tell me which feature next and I will add it.</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---- Custom CSS for styling ----
st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        text-align: center;
        color: #2E86C1;
        font-weight: bold;
    }
    .card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .internship-title {
        font-size: 22px;
        font-weight: bold;
        color: #1B4F72;
    }
    .internship-details {
        font-size: 16px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown('<p class="main-title">🎓 Internship Recommendation System</p>', unsafe_allow_html=True)

# ---- User Input ----
st.sidebar.header("Your Profile")
skills = st.sidebar.text_area("Enter your skills (comma separated):", "Python, Data Analysis, Machine Learning")
location = st.sidebar.text_input("Preferred Location:", "Remote")
domain = st.sidebar.text_input("Preferred Domain:", "Data Science")

# ---- Sample Internship Data ----
data = {
    "Title": ["Data Science Intern", "Web Development Intern", "Marketing Intern"],
    "Skills": ["Python, Machine Learning, Data Analysis", "HTML, CSS, JavaScript", "SEO, Social Media, Communication"],
    "Location": ["Remote", "Mumbai", "Delhi"],
    "Stipend": ["₹10,000", "₹5,000", "₹3,000"]
}
df = pd.DataFrame(data)

# ---- Recommendation Logic ----
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["Skills"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf.transform([skills]))
df["Score"] = cosine_sim.flatten()
recommendations = df.sort_values(by="Score", ascending=False)

# ---- Display Results ----
st.subheader("🔎 Recommended Internships")
for _, row in recommendations.iterrows():
    st.markdown(f"""
    <div class="card">
        <div class="internship-title">{row['Title']}</div>
        <div class="internship-details">📍 {row['Location']} | 💰 {row['Stipend']}</div>
        <div class="internship-details"><b>Required Skills:</b> {row['Skills']}</div>
    </div>
    """, unsafe_allow_html=True)
