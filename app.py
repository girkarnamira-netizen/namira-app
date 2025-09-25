import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Language Strings ---
# A dictionary to hold all text strings for different languages
text_strings = {
    "en": {
        "page_title": "InternMate",
        "header_title": "InternMate",
        "header_tagline": "Find your next big opportunity with InternMate—a personalized search experience.",
        "sidebar_title": "Search and Filter",
        "sidebar_tagline": "Use the filters below to get personalized recommendations.",
        "select_language": "Select Language",
        "work_mode": "Select Work Mode",
        "online": "Online",
        "offline": "Offline",
        "any": "Any",
        "enter_skills": "Enter your skills (e.g., Python, AI, Web Development)",
        "min_stipend": "Minimum stipend (₹)",
        "show_recommendations": "Show Recommendations",
        "recommendations_title": "Your Internship Recommendations",
        "stipend_label": "Stipend",
        "no_results": "No internships found matching your criteria.",
        "help_title": "Help and Support Centre",
        "faq_1_q": "How does the recommendation engine work?",
        "faq_1_a": "Our recommendation engine uses a combination of natural language processing (NLP) and machine learning algorithms. It analyzes the skills and preferences you provide and compares them to the internship descriptions and required skills. It then uses a technique called **Cosine Similarity** to find the most relevant matches.",
        "faq_2_q": "How do I filter for internships?",
        "faq_2_a": "You can use the sidebar to filter internships by **Work Mode**, **Skills**, and **Minimum Stipend**. For skills, you can enter multiple keywords separated by a comma (e.g., 'Python, Data Analysis, SQL').",
        "faq_3_q": "Can I upload my resume for personalized recommendations?",
        "faq_3_a": "Currently, this feature is not available. We are working to add a resume upload feature in future updates to provide even better recommendations.",
        "faq_4_q": "What is the difference between 'Online' and 'Offline' work modes?",
        "faq_4_a": "Selecting **'Online'** will show you internships with a **'Remote'** location. Selecting **'Offline'** will show you internships with an **'On-site'** or **'Hybrid'** location.",
        "faq_5_q": "How is the stipend calculated?",
        "faq_5_a": "The stipend displayed for each internship is a fixed amount specified in the dataset. The **'Minimum stipend'** slider in the sidebar filters the internships to show only those that meet or exceed your selected amount.",
        "footer": "InternMate - Made by Girkar Namira Siddique"
    },
    "hi": {
        "page_title": "इंटरनमैट",
        "header_title": "इंटरनमैट",
        "header_tagline": "इंटरनमैट के साथ अपना अगला बड़ा अवसर खोजें—एक व्यक्तिगत खोज अनुभव।",
        "sidebar_title": "खोजें और फ़िल्टर करें",
        "sidebar_tagline": "व्यक्तिगत सिफारिशों के लिए नीचे दिए गए फ़िल्टर का उपयोग करें।",
        "select_language": "भाषा चुनें",
        "work_mode": "कार्य मोड चुनें",
        "online": "ऑनलाइन",
        "offline": "ऑफ़लाइन",
        "any": "कोई भी",
        "enter_skills": "अपने कौशल दर्ज करें (उदा. Python, AI, Web Development)",
        "min_stipend": "न्यूनतम वजीफा (₹)",
        "show_recommendations": "सिफारिशें दिखाएं",
        "recommendations_title": "आपकी इंटर्नशिप सिफारिशें",
        "stipend_label": "वजीफा",
        "no_results": "आपके मानदंडों से मेल खाने वाली कोई इंटर्नशिप नहीं मिली।",
        "help_title": "सहायता और समर्थन केंद्र",
        "faq_1_q": "सिफारिश इंजन कैसे काम करता है?",
        "faq_1_a": "हमारा सिफारिश इंजन प्राकृतिक भाषा प्रसंस्करण (NLP) और मशीन लर्निंग एल्गोरिदम के संयोजन का उपयोग करता है। यह आपके द्वारा प्रदान किए गए कौशल और वरीयताओं का विश्लेषण करता है और उनकी इंटर्नशिप विवरण और आवश्यक कौशल से तुलना करता है। फिर यह सबसे प्रासंगिक मिलान खोजने के लिए **Cosine Similarity** नामक तकनीक का उपयोग करता है।",
        "faq_2_q": "मैं इंटर्नशिप कैसे फ़िल्टर करूँ?",
        "faq_2_a": "आप **कार्य मोड**, **कौशल**, और **न्यूनतम वजीफा** द्वारा इंटर्नशिप को फ़िल्टर करने के लिए साइडबार का उपयोग कर सकते हैं। कौशल के लिए, आप एक अल्पविराम (,) से अलग करके कई कीवर्ड दर्ज कर सकते हैं (उदा. 'Python, Data Analysis, SQL')।",
        "faq_3_q": "क्या मैं व्यक्तिगत सिफारिशों के लिए अपना बायोडाटा अपलोड कर सकता हूँ?",
        "faq_3_a": "वर्तमान में, यह सुविधा उपलब्ध नहीं है। हम और भी बेहतर सिफारिशें प्रदान करने के लिए भविष्य के अपडेट में बायोडाटा अपलोड सुविधा जोड़ने पर काम कर रहे हैं।",
        "faq_4_q": "'ऑनलाइन' और 'ऑफ़लाइन' कार्य मोड में क्या अंतर है?",
        "faq_4_a": "**'ऑनलाइन'** चुनने पर आपको **'रिमोट'** स्थान वाली इंटर्नशिप दिखाई देगी। **'ऑफ़लाइन'** चुनने पर आपको **'ऑन-साइट'** या **'हाइब्रिड'** स्थान वाली इंटर्नशिप दिखाई देगी।",
        "faq_5_q": "वजीफा की गणना कैसे की जाती है?",
        "faq_5_a": "प्रत्येक इंटर्नशिप के लिए प्रदर्शित वजीफा डेटासेट में निर्दिष्ट एक निश्चित राशि है। साइडबार में **'न्यूनतम वजीफा'** स्लाइडर केवल उन इंटर्नशिप को फ़िल्टर करता है जो आपके द्वारा चुनी गई राशि के बराबर या उससे अधिक हैं।",
        "footer": "InternMate - Girkar Namira Siddique द्वारा बनाया गया"
    },
    "mr": {
        "page_title": "इंटरनमैट",
        "header_title": "इंटरनमैट",
        "header_tagline": "InternMate सह तुमची पुढील मोठी संधी शोधा—एक वैयक्तिक शोध अनुभव.",
        "sidebar_title": "शोधा आणि फिल्टर करा",
        "sidebar_tagline": "वैयक्तिक शिफारसी मिळवण्यासाठी खालील फिल्टर वापरा.",
        "select_language": "भाषा निवडा",
        "work_mode": "कार्य मोड निवडा",
        "online": "ऑनलाइन",
        "offline": "ऑफलाइन",
        "any": "कोणतेही",
        "enter_skills": "तुमची कौशल्ये टाका (उदा. Python, AI, Web Development)",
        "min_stipend": "किमान स्टायपेंड (₹)",
        "show_recommendations": "शिफारसी दाखवा",
        "recommendations_title": "तुमच्या इंटर्नशिप शिफारसी",
        "stipend_label": "स्टायपेंड",
        "no_results": "तुमच्या निकषांशी जुळणारी कोणतीही इंटर्नशिप सापडली नाही.",
        "help_title": "मदत आणि समर्थन केंद्र",
        "faq_1_q": "शिफारस इंजिन कसे कार्य करते?",
        "faq_1_a": "आमचे शिफारस इंजिन नैसर्गिक भाषा प्रक्रिया (NLP) आणि मशीन लर्निंग अल्गोरिदमचे संयोजन वापरते. ते तुम्ही दिलेल्या कौशल्यांचे आणि आवडीनिवडींचे विश्लेषण करते आणि त्यांची इंटर्नशिपच्या वर्णनाशी आणि आवश्यक कौशल्यांशी तुलना करते. त्यानंतर सर्वात संबंधित जुळणी शोधण्यासाठी ते **Cosine Similarity** नावाचे तंत्र वापरते।",
        "faq_2_q": "मी इंटर्नशिप कशी फिल्टर करू?",
        "faq_2_a": "तुम्ही **कार्य मोड**, **कौशल्ये**, आणि **किमान स्टायपेंड** नुसार इंटर्नशिप फिल्टर करण्यासाठी साइडबार वापरू शकता। कौशल्यांसाठी, तुम्ही स्वल्पविरामाने (,) वेगळे करून अनेक कीवर्ड टाकू शकता (उदा. 'Python, Data Analysis, SQL').",
        "faq_3_q": "मी वैयक्तिक शिफारसींसाठी माझा बायोडाटा अपलोड करू शकतो का?",
        "faq_3_a": "सध्या, हे वैशिष्ट्य उपलब्ध नाही। आम्ही आणखी चांगल्या शिफारसी देण्यासाठी भविष्यातील अपडेट्समध्ये बायोडाटा अपलोड करण्याचे वैशिष्ट्य जोडण्याचे काम करत आहोत।",
        "faq_4_q": "'ऑनलाइन' आणि 'ऑफलाइन' कार्य मोडमध्ये काय फरक आहे?",
        "faq_4_a": "**'ऑनलाइन'** निवडल्यास तुम्हाला **'रिमोट'** स्थान असलेल्या इंटर्नशिप दिसतील. **'ऑफलाइन'** निवडल्यास तुम्हाला **'ऑन-साइट'** किंवा **'हाइब्रिड'** स्थान असलेल्या इंटर्नशिप दिसतील.",
        "faq_5_q": "स्टायपेंडची गणना कशी केली जाते?",
        "faq_5_a": "प्रत्येक इंटर्नशिपसाठी दर्शविलेला स्टायपेंड डेटासेटमध्ये निर्दिष्ट केलेली एक निश्चित रक्कम आहे. साइडबारमधील **'किमान स्टायपेंड'** स्लाइडर तुम्ही निवडलेल्या रकमेच्या बरोबरीच्या किंवा त्याहून अधिक इंटर्नशिप फिल्टर करतो.",
        "footer": "InternMate - Girkar Namira Siddique यांनी तयार केले आहे"
    },
    "ta": {
        "page_title": "இன்டர்ன்மேட்",
        "header_title": "இன்டர்ன்மேட்",
        "header_tagline": "இன்டர்ன்மேட் உடன் உங்கள் அடுத்த பெரிய வாய்ப்பைக் கண்டறியவும் - தனிப்பயனாக்கப்பட்ட தேடல் அனுபவம்.",
        "sidebar_title": "தேடு மற்றும் வடிகட்டு",
        "sidebar_tagline": "தனிப்பயனாக்கப்பட்ட பரிந்துரைகளைப் பெற கீழே உள்ள வடிப்பான்களைப் பயன்படுத்தவும்.",
        "select_language": "மொழியைத் தேர்ந்தெடுக்கவும்",
        "work_mode": "வேலை முறையைத் தேர்ந்தெடுக்கவும்",
        "online": "ஆன்லைன்",
        "offline": "ஆஃப்லைன்",
        "any": "ஏதேனும்",
        "enter_skills": "உங்கள் திறன்களை உள்ளிடவும் (எ.கா., Python, AI, Web Development)",
        "min_stipend": "குறைந்தபட்ச உதவித்தொகை (₹)",
        "show_recommendations": "பரிந்துரைகளைக் காட்டு",
        "recommendations_title": "உங்கள் இன்டர்ன்ஷிப் பரிந்துரைகள்",
        "stipend_label": "உதவித்தொகை",
        "no_results": "உங்கள் அளவுகோல்களுடன் பொருந்தக்கூடிய இன்டர்ன்ஷிப்கள் எதுவும் கண்டறியப்படவில்லை.",
        "help_title": "உதவி மற்றும் ஆதரவு மையம்",
        "faq_1_q": "பரிந்துரை இயந்திரம் எவ்வாறு செயல்படுகிறது?",
        "faq_1_a": "எங்கள் பரிந்துரை இயந்திரம் இயற்கை மொழி செயலாக்கம் (NLP) மற்றும் இயந்திர கற்றல் அல்காரிதம்களின் கலவையைப் பயன்படுத்துகிறது. நீங்கள் வழங்கும் திறன்கள் மற்றும் விருப்பங்களை இது பகுப்பாய்வு செய்து, இன்டர்ன்ஷிப் விளக்கங்கள் மற்றும் தேவையான திறன்களுடன் ஒப்பிடுகிறது. பின்னர் இது மிகவும் பொருத்தமான பொருத்தங்களைக் கண்டறிய **Cosine Similarity** எனப்படும் ஒரு நுட்பத்தைப் பயன்படுத்துகிறது.",
        "faq_2_q": "இன்டர்ன்ஷிப்களை நான் எப்படி வடிகட்டுவது?",
        "faq_2_a": "இன்டர்ன்ஷிப்களை **வேலை முறை**, **திறன்கள்**, மற்றும் **குறைந்தபட்ச உதவித்தொகை** ஆகியவற்றின் அடிப்படையில் வடிகட்ட நீங்கள் பக்கப்பட்டியைப் பயன்படுத்தலாம். திறன்களுக்கு, நீங்கள் கமாவால் (,) பிரிக்கப்பட்ட பல முக்கிய வார்த்தைகளை உள்ளிடலாம் (எ.கா., 'Python, Data Analysis, SQL').",
        "faq_3_q": "தனிப்பயனாக்கப்பட்ட பரிந்துரைகளுக்கு எனது பயோடேட்டாவை நான் பதிவேற்ற முடியுமா?",
        "faq_3_a": "தற்போது, இந்த அம்சம் கிடைக்கவில்லை. இன்னும் சிறந்த பரிந்துரைகளை வழங்குவதற்காக எதிர்கால புதுப்பிப்புகளில் பயோடேட்டா பதிவேற்ற அம்சத்தைச் சேர்க்க நாங்கள் பணியாற்றி வருகிறோம்.",
        "faq_4_q": "'ஆன்லைன்' மற்றும் 'ஆஃப்லைன்' வேலை முறைகளுக்கு என்ன வித்தியாசம்?",
        "faq_4_a": "**'ஆன்லைன்'** என்பதைத் தேர்ந்தெடுப்பது, **'ரிமோட்'** இருப்பிடத்துடன் இன்டர்ன்ஷிப்களை உங்களுக்குக் காண்பிக்கும். **'ஆஃப்லைன்'** என்பதைத் தேர்ந்தெடுப்பது, **'ஆன்-சைட்'** அல்லது **'ஹைப்ரிட்'** இருப்பிடத்துடன் இன்டர்ன்ஷிப்களை உங்களுக்குக் காண்பிக்கும்.",
        "faq_5_q": "உதவித்தொகை எவ்வாறு கணக்கிடப்படுகிறது?",
        "faq_5_a": "ஒவ்வொரு இன்டர்ன்ஷிப்பிற்கும் காட்டப்படும் உதவித்தொகை, தரவுத்தொகுப்பில் குறிப்பிடப்பட்ட ஒரு நிலையான தொகையாகும். பக்கப்பட்டியில் உள்ள **'குறைந்தபட்ச உதவித்தொகை'** ஸ்லைடர், நீங்கள் தேர்ந்தெடுத்த தொகைக்கு சமமான அல்லது அதற்கு அதிகமாக உள்ள இன்டர்ன்ஷிப்களை மட்டுமே வடிகட்டுகிறது।",
        "footer": "InternMate - Girkar Namira Siddique அவர்களால் உருவாக்கப்பட்டது"
    }
}

# Session State for Language
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

# --- Page Config ---
st.set_page_config(
    page_title=text_strings[st.session_state.lang]["page_title"],
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.header {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center;
}
.header h1 {
    color: #4a4a4a;
    font-size: 2.5rem;
    font-weight: bold;
}
.header p {
    color: #888888;
    margin-top: 10px;
}
.sidebar-section {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-top: 20px;
}
.internship-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s;
    border-left: 5px solid #4CAF50;
}
.internship-card:hover {
    transform: translateY(-5px);
}
.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.card-title {
    font-size: 1.25rem;
    font-weight: bold;
    color: #333333;
}
.card-location {
    font-size: 0.9rem;
    color: #777777;
}
.card-stipend {
    font-size: 1rem;
    color: #555555;
    font-weight: bold;
    margin-top: 5px;
}
.card-description {
    font-size: 0.9rem;
    color: #666666;
    margin-top: 10px;
}
.card-skills {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 10px;
}
.skill-tag {
    background-color: #e6f3ff;
    color: #007bff;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 0.8rem;
}
.empty-results {
    text-align: center;
    color: #999999;
    margin-top: 50px;
}
footer {
    text-align: center;
    margin-top: 50px;
    font-size: 0.8rem;
    color: #aaaaaa;
}
.help-section {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-top: 30px;
}
.help-section h3 {
    color: #333333;
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_internships():
    data = {
        'company': [
            'Tech Innovators Inc.', 'Data Wizards Ltd.', 'Creative Solutions Co.',
            'Global Marketing Agency', 'AI Driven Insights', 'Quantum Tech',
            'Financial Futures', 'Health Tech Solutions', 'Sustainable Energy Co.',
            'GameDev Studio', 'Product Pulse Co.', 'PM Solutions Hub'
        ],
        'role': [
            'Software Engineer Intern', 'Data Analyst Intern', 'UI/UX Design Intern',
            'Digital Marketing Intern', 'Machine Learning Intern', 'Cloud Computing Intern',
            'Financial Analyst Intern', 'Healthcare Data Intern', 'Environmental Analyst Intern',
            'Game Developer Intern', 'Product Management Intern', 'PM Intern'
        ],
        'location': [
            'Remote', 'On-site (Mumbai)', 'Hybrid (Delhi)', 'Remote',
            'On-site (Bangalore)', 'Remote', 'Hybrid (Chennai)',
            'On-site (Pune)', 'Remote', 'Hybrid (Hyderabad)', 'On-site (Mumbai)', 'On-site (Delhi)'
        ],
        'stipend': [
            '₹15,000', '₹20,000', '₹12,000', '₹10,000',
            '₹25,000', '₹18,000', '₹16,000', '₹22,000',
            '₹14,000', '₹17,000', '₹21,000', '₹20,000'
        ],
        'description': [
            'Developing web applications using Python and React.',
            'Analyzing large datasets and creating data visualizations.',
            'Designing user interfaces and creating wireframes for mobile apps.',
            'Managing social media campaigns and creating content.',
            'Building and training machine learning models.',
            'Working on cloud infrastructure and deployment pipelines.',
            'Assisting with financial modeling and market analysis.',
            'Processing and analyzing patient data for insights.',
            'Researching and analyzing data for renewable energy projects.',
            'Developing game mechanics and level design using Unity.',
            'Assisting with product roadmaps, market research, and feature ideation.',
            'A hands-on role in a PM team, assisting with product strategy and launch.'
        ],
        'skills': [
            'Python, React, JavaScript, Git', 'Python, Pandas, SQL, Visualization',
            'Figma, Sketch, UI/UX, Prototyping', 'SEO, SEM, Social Media, Content Creation',
            'Python, TensorFlow, Scikit-learn, NLP', 'AWS, Azure, Docker, Kubernetes',
            'Excel, Financial Modeling, Data Analysis', 'SQL, Python, Data Cleaning, Statistics',
            'Python, GIS, R, Data Analysis', 'Unity, C#, Game Design, 3D Modeling',
            'Product Management, Market Research, Agile, JIRA', 'Product Strategy, Agile, Scrum, Market Analysis'
        ]
    }
    return pd.DataFrame(data)

df = load_internships()

# --- Main App Content ---
st.markdown(f"<div class='header'><h1>{text_strings[st.session_state.lang]['header_title']}</h1><p>{text_strings[st.session_state.lang]['header_tagline']}</p></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.write("---")

# --- Sidebar ---
st.sidebar.title(text_strings[st.session_state.lang]['sidebar_title'])
st.sidebar.markdown(text_strings[st.session_state.lang]['sidebar_tagline'])

# Language selection
language_options = {'English': 'en', 'हिंदी': 'hi', 'मराठी': 'mr', 'தமிழ்': 'ta'}
selected_language_name = st.sidebar.selectbox(text_strings[st.session_state.lang]['select_language'], options=list(language_options.keys()))
st.session_state.lang = language_options[selected_language_name]

# Work Mode selection
work_mode_options = {
    text_strings[st.session_state.lang]['online']: 'Online',
    text_strings[st.session_state.lang]['offline']: 'Offline',
    text_strings[st.session_state.lang]['any']: 'Any'
}
selected_work_mode_name = st.sidebar.selectbox(text_strings[st.session_state.lang]['work_mode'], options=list(work_mode_options.keys()))
work_mode = work_mode_options[selected_work_mode_name]

search_query = st.sidebar.text_input(text_strings[st.session_state.lang]['enter_skills'])
st.sidebar.write(text_strings[st.session_state.lang]['min_stipend'])
min_stipend = st.sidebar.slider("", 0, 50000, 0, step=1000)

# Add a button to trigger recommendations
if st.sidebar.button(text_strings[st.session_state.lang]['show_recommendations']):
    # --- Filtering logic ---
    filtered_df = df.copy()

    # Filter by Work Mode
    if work_mode == 'Online':
        filtered_df = filtered_df[filtered_df['location'].str.contains('Remote', case=False, na=False)]
    elif work_mode == 'Offline':
        filtered_df = filtered_df[filtered_df['location'].str.contains('On-site|Hybrid', case=False, na=False)]

    # Filter by stipend
    filtered_df['stipend_numeric'] = filtered_df['stipend'].str.replace('₹', '').str.replace(',', '').astype(int)
    filtered_df = filtered_df[filtered_df['stipend_numeric'] >= min_stipend]

    # --- Recommendation logic ---
    if search_query:
        # Combine skills and description for better matching
        filtered_df['combined_text'] = filtered_df['skills'] + " " + filtered_df['description']

        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer().fit(filtered_df['combined_text'])
        internship_vectors = vectorizer.transform(filtered_df['combined_text'])

        # Create a vector for the user's query
        query_vector = vectorizer.transform([search_query])

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(query_vector, internship_vectors).flatten()

        # Get top recommendations
        filtered_df['similarity_score'] = cosine_similarities
        filtered_df = filtered_df.sort_values(by='similarity_score', ascending=False)
        filtered_df = filtered_df[filtered_df['similarity_score'] > 0]

    # --- Display results ---
    st.write(f"### {text_strings[st.session_state.lang]['recommendations_title']}")

    if not filtered_df.empty:
        st.markdown("<div class='card-grid'>", unsafe_allow_html=True)
        for index, row in filtered_df.iterrows():
            st.markdown(f"""
            <div class='internship-card'>
                <div class='card-header'>
                    <div class='card-title'>{row['role']}</div>
                    <div class='card-location'>{row['location']}</div>
                </div>
                <div class='card-stipend'>{text_strings[st.session_state.lang]['stipend_label']}: {row['stipend']}</div>
                <div class='card-description'>{row['description']}</div>
                <div class='card-skills'>
                    {"".join([f"<span class='skill-tag'>{skill.strip()}</span>" for skill in row['skills'].split(',')])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='empty-results'>{text_strings[st.session_state.lang]['no_results']}</div>", unsafe_allow_html=True)

st.write("---")

# --- Help and Support Center ---
st.markdown(f"""
<div class='help-section'>
    <h3>{text_strings[st.session_state.lang]['help_title']}</h3>
</div>
""", unsafe_allow_html=True)

with st.expander(text_strings[st.session_state.lang]['faq_1_q']):
    st.write(text_strings[st.session_state.lang]['faq_1_a'])

with st.expander(text_strings[st.session_state.lang]['faq_2_q']):
    st.write(text_strings[st.session_state.lang]['faq_2_a'])

with st.expander(text_strings[st.session_state.lang]['faq_3_q']):
    st.write(text_strings[st.session_state.lang]['faq_3_a'])

with st.expander(text_strings[st.session_state.lang]['faq_4_q']):
    st.write(text_strings[st.session_state.lang]['faq_4_a'])

with st.expander(text_strings[st.session_state.lang]['faq_5_q']):
    st.write(text_strings[st.session_state.lang]['faq_5_a'])

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(f"<footer>{text_strings[st.session_state.lang]['footer']}</footer>", unsafe_allow_html=True)
