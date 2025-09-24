import streamlit as st
import pandas as pd

# Define a sample dataset for recommendations.
# In a real app, you would load this from a database or a CSV file.
internship_data = {
    "online": [
        {"name": "Remote Software Engineer Intern", "company": "Tech Innovators", "skills": "Python, React, API", "mode": "Online"},
        {"name": "Virtual Data Analyst Intern", "company": "Data Insights Co.", "skills": "SQL, Excel, Data Visualization", "mode": "Online"},
        {"name": "Digital Marketing Intern", "company": "Growth Wizards", "skills": "SEO, SEM, Social Media", "mode": "Online"},
        {"name": "Online UX/UI Design Intern", "company": "Creative Minds Studio", "skills": "Figma, Sketch, User Research", "mode": "Online"},
        {"name": "Remote Product Management Intern", "company": "Future Solutions", "skills": "Market Analysis, Agile", "mode": "Online"},
    ],
    "offline": [
        {"name": "On-site Electrical Engineering Intern", "company": "PowerGrid Inc.", "skills": "Circuit Design, AutoCAD", "mode": "Offline"},
        {"name": "In-person Civil Engineer Intern", "company": "MegaBuild Corp.", "skills": "Structural Analysis, CAD", "mode": "Offline"},
        {"name": "Local PR and Communications Intern", "company": "Media Hub", "skills": "Public Relations, Event Planning", "mode": "Offline"},
        {"name": "Laboratory Research Intern", "company": "BioGen Labs", "skills": "Cell Biology, Lab Techniques", "mode": "Offline"},
        {"name": "Field Operations Intern", "company": "Logistics Pro", "skills": "Supply Chain, Inventory Management", "mode": "Offline"},
    ]
}

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="InternMate",
    page_icon="🔎",
    layout="wide",
)

# You can add custom CSS to make it even more beautiful
st.markdown("""
<style>
    .main-header {
        color: #2c3e50;
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 3.5em;
        font-weight: bold;
        text-shadow: 2px 2px 4px #ccc;
        margin-bottom: 20px;
    }
    .st-emotion-cache-1r65d3m {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-1r65d3m h3 {
        color: #34495e;
    }
    .st-emotion-cache-1r65d3m .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .st-emotion-cache-1r65d3m .stButton button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
    }
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .card-title {
        color: #2980b9;
        font-size: 1.2em;
        font-weight: bold;
    }
    .card-company {
        color: #7f8c8d;
        font-size: 0.9em;
        margin-top: -5px;
    }
    .card-skills {
        background-color: #ecf0f1;
        border-radius: 8px;
        padding: 5px 10px;
        font-size: 0.8em;
        color: #34495e;
        margin-top: 10px;
        display: inline-block;
    }
    .st-emotion-cache-1r65d3m .stSelectbox {
        color: #34495e;
    }
    .st-emotion-cache-1r65d3m .stTextInput {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

# --- Main Page Layout ---
st.markdown("<h1 class='main-header'>InternMate - Find Your Internship</h1>", unsafe_allow_html=True)
st.write("Welcome to the internship recommendation platform. We'll help you find the perfect match!")

# --- Sidebar for user input ---
st.sidebar.header("Your Profile")
user_name = st.sidebar.text_input("Enter your name:", "Your Name")
skills_input = st.sidebar.text_area("Your skills (comma-separated):", "Python, Machine Learning, Data Analysis")

# --- New 'Work Mode' Feature ---
work_mode = st.sidebar.selectbox(
    "Select your preferred work mode:",
    ("Online", "Offline")
)

st.sidebar.button("Update Profile")

# --- Main Content Area ---
st.header(f"Recommendations for {user_name}")

# Filter internships based on the selected work mode
mode_filtered_internships = internship_data[work_mode.lower()]
df = pd.DataFrame(mode_filtered_internships)

# Search functionality (case-insensitive)
search_query = st.text_input("Search for internships...", "")
if search_query:
    filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
else:
    filtered_df = df

st.subheader(f"{work_mode} Internships ({len(filtered_df)} results)")
st.write("Based on your profile, here are some recommendations:")

# Display recommendations in a grid of cards
if not filtered_df.empty:
    columns = st.columns(3) # Display 3 cards per row
    col_index = 0
    for index, row in filtered_df.iterrows():
        with columns[col_index]:
            st.markdown(
                f"""
                <div class="card">
                    <p class="card-title">{row['name']}</p>
                    <p class="card-company">at {row['company']}</p>
                    <p><b>Mode:</b> {row['mode']}</p>
                    <p class="card-skills"><b>Skills:</b> {row['skills']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        col_index = (col_index + 1) % 3
else:
    st.info("No internships found matching your criteria.")

st.markdown("---")
st.write("Powered by InternMate")
st.write("Made by: Girkar Namira Siddique")
