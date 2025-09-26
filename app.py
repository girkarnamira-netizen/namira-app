import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Language Strings (Includes all multilingual text) ---
text_strings = {
    "en": {
        "page_title": "InternMate", "header_title": "InternMate 💼",
        "header_tagline": "Find your next big opportunity with InternMate—a personalized search experience.",
        "sidebar_title": "🔍 Search and Filter", "sidebar_tagline": "Use the filters below to get personalized recommendations.",
        "select_language": "🌐 Select Language", "select_location": "📍 Preferred Location",
        "work_mode": "💻 Select Work Mode", "online": "Online", "offline": "Offline", "any": "Any",
        "enter_skills": "✏️ Enter your skills (e.g., Python, AI, Web Development)",
        "min_stipend": "💰 Minimum stipend (₹)", "show_recommendations": "✨ Show Recommendations",
        "recommendations_title": "🎯 Your Internship Recommendations", "stipend_label": "Stipend",
        "upload_cv": "📄 Upload CV/Resume (PDF or TXT)",
        "no_results": "😔 No internships found matching your criteria. Try broadening your search!", "help_title": "📚 Help and Support Centre",
        "faq_1_q": "How does the recommendation engine work?", "faq_1_a": "Our recommendation engine uses a combination of natural language processing (NLP) and machine learning algorithms. It uses **Cosine Similarity** to find the most relevant matches based on your skills.",
        "faq_2_q": "How do I filter for internships?", "faq_2_a": "You can use the sidebar to filter internships by **Work Mode**, **Location**, **Skills**, and **Minimum Stipend**.",
        "faq_3_q": "Can I upload my resume for personalized recommendations?", "faq_3_a": "Yes, you can now upload your resume (PDF or TXT) directly in the sidebar. While the system currently uses manual skill entry for matching, this upload feature is in place for future integration with automated CV parsing.",
        "faq_4_q": "What is the difference between 'Online' and 'Offline' work modes?", "faq_4_a": "Selecting **'Online'** shows Remote internships. **'Offline'** shows On-site or Hybrid internships.",
        "faq_5_q": "How is the stipend calculated?", "faq_5_a": "The stipend displayed for each internship is a fixed amount specified in the dataset. The **'Minimum stipend'** slider filters for internships that meet or exceed this amount.",
        "footer": "InternMate - Made by Girkar Namira Siddique"
    },
    "hi": {
        "page_title": "इंटरनमैट", "header_title": "इंटरनमैट 💼",
        "header_tagline": "इंटरनमैट के साथ अपना अगला बड़ा अवसर खोजें—एक व्यक्तिगत खोज अनुभव।",
        "sidebar_title": "🔍 खोजें और फ़िल्टर करें", "sidebar_tagline": "व्यक्तिगत सिफारिशों के लिए नीचे दिए गए फ़िल्टर का उपयोग करें।",
        "select_language": "🌐 भाषा चुनें", "select_location": "📍 पसंदीदा स्थान",
        "work_mode": "💻 कार्य मोड चुनें", "online": "ऑनलाइन", "offline": "ऑफ़लाइन", "any": "कोई भी",
        "enter_skills": "✏️ अपने कौशल दर्ज करें (उदा. Python, AI, Web Development)",
        "min_stipend": "💰 न्यूनतम वजीफा (₹)", "show_recommendations": "✨ सिफारिशें दिखाएं",
        "recommendations_title": "🎯 आपकी इंटर्नशिप सिफारिशें", "stipend_label": "वजीफा",
        "upload_cv": "📄 बायोडाटा / रिज्यूमे अपलोड करें (PDF या TXT)",
        "no_results": "😔 आपके मानदंडों से मेल खाने वाली कोई इंटर्नशिप नहीं मिली। अपनी खोज का विस्तार करें!", "help_title": "📚 सहायता और समर्थन केंद्र",
        "faq_1_q": "सिफारिश इंजन कैसे काम करता है?", "faq_1_a": "हमारा सिफारिश इंजन प्राकृतिक भाषा प्रसंस्करण (NLP) और मशीन लर्निंग एल्गोरिदम के संयोजन का उपयोग करता है। यह सबसे प्रासंगिक मिलान खोजने के लिए **Cosine Similarity** नामक तकनीक का उपयोग करता है।",
        "faq_2_q": "मैं इंटर्नशिप कैसे फ़िल्टर करूँ?", "faq_2_a": "आप **कार्य मोड**, **स्थान**, **कौशल**, और **न्यूनतम वजीफा** द्वारा इंटर्नशिप को फ़िल्टर करने के लिए साइडबार का उपयोग कर सकते हैं।",
        "faq_3_q": "क्या मैं व्यक्तिगत सिफारिशों के लिए अपना बायोडाटा अपलोड कर सकता हूँ?", "faq_3_a": "हाँ, अब आप सीधे साइडबार में अपना बायोडाटा (PDF या TXT) अपलोड कर सकते हैं। हालांकि सिस्टम वर्तमान में मिलान के लिए मैन्युअल स्किल एंट्री का उपयोग करता है, यह अपलोड सुविधा भविष्य में ऑटोमेटेड CV पार्सिंग के साथ एकीकरण के लिए मौजूद है।",
        "faq_4_q": "'ऑनलाइन' और 'ऑफ़लाइन' कार्य मोड में क्या अंतर है?", "faq_4_a": "**'ऑनलाइन'** चुनने पर रिमोट इंटर्नशिप दिखाई देगी। **'ऑफ़लाइन'** चुनने पर ऑन-साइट या हाइब्रिड इंटर्नशिप दिखाई देगी।",
        "faq_5_q": "वजीफा की गणना कैसे की जाती है?", "faq_5_a": "प्रत्येक इंटर्नशिप के लिए प्रदर्शित वजीफा डेटासेट में निर्दिष्ट एक निश्चित राशि है। **'न्यूनतम वजीफा'** स्लाइडर केवल उन इंटर्नशिप को फ़िल्टर करता है जो आपके द्वारा चुनी गई राशि के बराबर या उससे अधिक हैं।",
        "footer": "InternMate - Girkar Namira Siddique द्वारा बनाया गया"
    },
    "mr": {
        "page_title": "इंटरनमैट", "header_title": "इंटरनमैट 💼",
        "header_tagline": "InternMate सह तुमची पुढील मोठी संधी शोधा—एक वैयक्तिक शोध अनुभव.",
        "sidebar_title": "🔍 शोधा आणि फिल्टर करा", "sidebar_tagline": "वैयक्तिक शिफारसी मिळवण्यासाठी खालील फिल्टर वापरा.",
        "select_language": "🌐 भाषा निवडा", "select_location": "📍 पसंतीचे स्थान",
        "work_mode": "💻 कार्य मोड निवडा", "online": "ऑनलाइन", "offline": "ऑफलाइन", "any": "कोणतेही",
        "enter_skills": "✏️ तुमची कौशल्ये टाका (उदा. Python, AI, Web Development)",
        "min_stipend": "💰 किमान स्टायपेंड (₹)", "show_recommendations": "✨ शिफारसी दाखवा",
        "recommendations_title": "🎯 तुमच्या इंटर्नशिप शिफारसी", "stipend_label": "स्टायपेंड",
        "upload_cv": "📄 बायोडाटा / रिझ्युमे अपलोड करा (PDF किंवा TXT)",
        "no_results": "😔 तुमच्या निकषांशी जुळणारी कोणतीही इंटर्नशिप सापडली नाही। कृपया तुमचा शोध विस्तृत करा!", "help_title": "📚 मदत आणि समर्थन केंद्र",
        "faq_1_q": "शिफारस इंजिन कसे कार्य करते?", "faq_1_a": "आमचे शिफारस इंजिन नैसर्गिक भाषा प्रक्रिया (NLP) आणि मशीन लर्निंग अल्गोरिदमचे संयोजन वापरते। सर्वात संबंधित जुळणी शोधण्यासाठी ते **Cosine Similarity** नावाचे तंत्र वापरते।",
        "faq_2_q": "मी इंटर्नशिप कशी फिल्टर करू?", "faq_2_a": "तुम्ही **कार्य मोड**, **स्थान**, **कौशल्ये**, आणि **किमान स्टायपेंड** नुसार इंटर्नशिप फिल्टर करण्यासाठी साइडबार वापरू शकता।",
        "faq_3_q": "मी वैयक्तिक शिफारसींसाठी माझा बायोडाटा अपलोड करू शकतो का?", "faq_3_a": "होय, आता तुम्ही थेट साइडबारमध्ये तुमचा बायोडाटा (PDF किंवा TXT) अपलोड करू शकता. सिस्टीम सध्या जुळणीसाठी मॅन्युअल कौशल्य एंट्री वापरत असली तरी, हे अपलोड वैशिष्ट्य भविष्यात ऑटोमेटेड CV पार्सिंगसह एकत्रीकरणासाठी योग्य आहे।",
        "faq_4_q": "'ऑनलाइन' आणि 'ऑफलाइन' कार्य मोडमध्ये काय फरक आहे?", "faq_4_a": "**'ऑनलाइन'** निवडल्यास रिमोट इंटर्नशिप दिसतील. **'ऑफलाइन'** निवडल्यास ऑन-साइट किंवा हाइब्रिड इंटर्नशिप दिसतील।",
        "faq_5_q": "स्टायपेंडची गणना कशी केली जाते?", "faq_5_a": "प्रत्येक इंटर्नशिपसाठी दर्शविलेला स्टायपेंड डेटासेटमध्ये निर्दिष्ट केलेली एक निश्चित रक्कम आहे। **'किमान स्टायपेंड'** स्लाइडर तुम्ही निवडलेल्या रकमेच्या बरोबरीच्या किंवा त्याहून अधिक इंटर्नशिप फिल्टर करतो।",
        "footer": "InternMate - Girkar Namira Siddique यांनी तयार केले आहे"
    },
    "ta": {
        "page_title": "இன்டர்ன்மேட்", "header_title": "இன்டர்ன்மேட் 💼",
        "header_tagline": "இன்டர்ன்மேட் உடன் உங்கள் அடுத்த பெரிய வாய்ப்பைக் கண்டறியவும் - தனிப்பயனாக்கப்பட்ட தேடல் அனுபவம்.",
        "sidebar_title": "🔍 தேடு மற்றும் வடிகட்டு", "sidebar_tagline": "தனிப்பயனாக்கப்பட்ட பரிந்துரைகளைப் பெற கீழே உள்ள வடிப்பான்களைப் பயன்படுத்தவும்.",
        "select_language": "🌐 மொழியைத் தேர்ந்தெடுக்கவும்", "select_location": "📍 விருப்பமான இடம்",
        "work_mode": "💻 வேலை முறையைத் தேர்ந்தெடுக்கவும்", "online": "ஆன்லைன்", "offline": "ஆஃப்லைன்", "any": "ஏதேனும்",
        "enter_skills": "✏️ உங்கள் திறன்களை உள்ளிடவும் (எ.கா., Python, AI, Web Development)",
        "min_stipend": "💰 குறைந்தபட்ச உதவித்தொகை (₹)", "show_recommendations": "✨ பரிந்துரைகளைக் காட்டு",
        "recommendations_title": "🎯 உங்கள் இன்டர்ன்ஷிப் பரிந்துரைகள்", "stipend_label": "உதவித்தொகை",
        "upload_cv": "📄 பயோடேட்டா / ரெஸ்யூம் பதிவேற்றவும் (PDF அல்லது TXT)",
        "no_results": "😔 உங்கள் அளவுகோல்களுடன் பொருந்தக்கூடிய இன்டர்ன்ஷிப்கள் எதுவும் கண்டறியப்படவில்லை. உங்கள் தேடலை விரிவாக்குங்கள்!", "help_title": "📚 உதவி மற்றும் ஆதரவு மையம்",
        "faq_1_q": "பரிந்துரை இயந்திரம் எவ்வாறு செயல்படுகிறது?", "faq_1_a": "எங்கள் பரிந்துரை இயந்திரம் இயற்கை மொழி செயலாக்கம் (NLP) மற்றும் இயந்திர கற்றல் அல்காரிதம்களின் கலவையைப் பயன்படுத்துகிறது. இது மிகவும் பொருத்தமான பொருத்தங்களைக் கண்டறிய **Cosine Similarity** எனப்படும் ஒரு நுட்பத்தைப் பயன்படுத்துகிறது.",
        "faq_2_q": "இன்டர்ன்ஷிப்களை நான் எப்படி வடிகட்டுவது?", "faq_2_a": "இன்டர்ன்ஷிப்களை **வேலை முறை**, **இடம்**, **திறன்கள்**, மற்றும் **குறைந்தபட்ச உதவித்தொகை** ஆகியவற்றின் அடிப்படையில் வடிகட்ட நீங்கள் பக்கப்பட்டியைப் பயன்படுத்தலாம்.",
        "faq_3_q": "தனிப்பயனாக்கப்பட்ட பரிந்துரைகளுக்கு எனது பயோடேட்டாவை நான் பதிவேற்ற முடியுமா?", "faq_3_a": "ஆம், இப்போது நீங்கள் நேரடியாக பக்கப்பட்டியில் உங்கள் பயோடேட்டாவை (PDF அல்லது TXT) பதிவேற்றலாம். கணினி தற்போது கையேடு திறன் உள்ளீட்டைப் பயன்படுத்தினாலும், எதிர்காலத்தில் தானியங்கு CV பகுப்பாய்வுடன் ஒருங்கிணைக்க இந்த பதிவேற்ற அம்சம் உள்ளது.",
        "faq_4_q": "'ஆன்லைன்' மற்றும் 'ஆஃப்லைன்' வேலை முறைகளுக்கு என்ன வித்தியாசம்?", "faq_4_a": "**'ஆன்லைன்'** என்பதைத் தேர்ந்தெடுப்பது, **'ரிமோட்'** இன்டர்ன்ஷிப்களை உங்களுக்குக் காண்பிக்கும். **'ஆஃப்லைன்'** என்பதைத் தேர்ந்தெடுப்பது, **'ஆன்-சைட்'** அல்லது **'ஹைப்ரிட்'** இன்டர்ன்ஷிப்களை உங்களுக்குக் காண்பிக்கும்.",
        "faq_5_q": "உதவித்தொகை எவ்வாறு கணக்கிடப்படுகிறது?", "faq_5_a": "ஒவ்வொரு இன்டர்ன்ஷிப்பிற்கும் காட்டப்படும் உதவித்தொகை, தரவுத்தொகுப்பில் குறிப்பிடப்பட்ட ஒரு நிலையான தொகையாகும். **'குறைந்தபட்ச உதவித்தொகை'** ஸ்லைடர், நீங்கள் தேர்ந்தெடுத்த தொகைக்கு சமமான அல்லது அதற்கு அதிகமாக உள்ள இன்டர்ன்ஷிப்களை மட்டுமே வடிகட்டுகிறது।",
        "footer": "InternMate - Girkar Namira Siddique அவர்களால் உருவாக்கப்பட்டது"
    }
}

# --- Callback Function for Language Change ---
def update_language():
    """Updates the session state language and forces a rerun."""
    # The new language name is stored in the key 'language_select_key'
    selected_name = st.session_state['language_select_key']
    language_options_map = {'English': 'en', 'हिंदी': 'hi', 'मराठी': 'mr', 'தமிழ்': 'ta'}
    st.session_state.lang = language_options_map.get(selected_name, 'en')
    # Use st.rerun instead of st.experimental_rerun for modern Streamlit versions
    st.rerun()

# --- Data Loading ---
@st.cache_data
def load_internships():
    # Standardize location names for filtering
    data = {
        'company': [
            'Tech Innovators Inc.', 'Data Wizards Ltd.', 'Creative Solutions Co.',
            'Global Marketing Agency', 'AI Driven Insights', 'Quantum Tech',
            'Financial Futures', 'Health Tech Solutions', 'Sustainable Energy Co.',
            'GameDev Studio', 'Product Pulse Co.', 'PM Solutions Hub',
            'E-Commerce Giants', 'FinTech Forward'
        ],
        'role': [
            'Software Engineer Intern', 'Data Analyst Intern', 'UI/UX Design Intern',
            'Digital Marketing Intern', 'Machine Learning Intern', 'Cloud Computing Intern',
            'Financial Analyst Intern', 'Healthcare Data Intern', 'Environmental Analyst Intern',
            'Game Developer Intern', 'Product Management Intern', 'PM Intern',
            'Frontend Developer Intern', 'Risk Analyst Intern'
        ],
        'location': [
            'Remote', 'Mumbai (On-site)', 'Delhi (Hybrid)', 'Remote',
            'Bangalore (On-site)', 'Remote', 'Chennai (Hybrid)',
            'Pune (On-site)', 'Remote', 'Hyderabad (Hybrid)', 'Mumbai (On-site)', 'Delhi (On-site)',
            'Remote', 'Mumbai (Hybrid)'
        ],
        'stipend': [
            '₹15,000', '₹20,000', '₹12,000', '₹10,000',
            '₹25,000', '₹18,000', '₹16,000', '₹22,000',
            '₹14,000', '₹17,000', '₹21,000', '₹20,000',
            '₹18,000', '₹23,000'
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
            'A hands-on role in a PM team, assisting with product strategy and launch.',
            'Building responsive user interfaces with React and Tailwind CSS.',
            'Analyzing financial risks and building predictive models.'
        ],
        'skills': [
            'Python, React, JavaScript, Git', 'Python, Pandas, SQL, Visualization',
            'Figma, Sketch, UI/UX, Prototyping', 'SEO, SEM, Social Media, Content Creation',
            'Python, TensorFlow, Scikit-learn, NLP', 'AWS, Azure, Docker, Kubernetes',
            'Excel, Financial Modeling, Data Analysis', 'SQL, Python, Data Cleaning, Statistics',
            'Python, GIS, R, Data Analysis', 'Unity, C#, Game Design, 3D Modeling',
            'Product Management, Market Research, Agile, JIRA', 'Product Strategy, Agile, Scrum, Market Analysis',
            'React, JavaScript, HTML, CSS, Tailwind', 'Python, R, Risk Analysis, Statistics'
        ],
        'image_url': [
            'https://placehold.co/400x150/007bff/ffffff?text=Software', 
            'https://placehold.co/400x150/4CAF50/ffffff?text=Data',
            'https://placehold.co/400x150/FFC107/ffffff?text=Design',
            'https://placehold.co/400x150/FF5722/ffffff?text=Marketing',
            'https://placehold.co/400x150/9C27B0/ffffff?text=ML/AI',
            'https://placehold.co/400x150/00BCD4/ffffff?text=Cloud',
            'https://placehold.co/400x150/F44336/ffffff?text=Finance',
            'https://placehold.co/400x150/795548/ffffff?text=Health',
            'https://placehold.co/400x150/607D8B/ffffff?text=Energy',
            'https://placehold.co/400x150/E91E63/ffffff?text=Gaming',
            'https://placehold.co/400x150/3F51B5/ffffff?text=Product',
            'https://placehold.co/400x150/03A9F4/ffffff?text=PM',
            'https://placehold.co/400x150/8BC34A/ffffff?text=Frontend',
            'https://placehold.co/400x150/673AB7/ffffff?text=Risk'
        ]
    }
    return pd.DataFrame(data)

df = load_internships()

# --- Session State for Language & Initial Config ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

# Must be called first to ensure UI updates correctly
st.set_page_config(
    page_title=text_strings[st.session_state.lang]["page_title"],
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---
st.markdown("""
<style>
/* Base App Styling */
.stApp { background-color: #f0f2f6; font-family: 'Inter', sans-serif; }

/* Header Styling */
.header { 
    background: linear-gradient(135deg, #e0f7fa, #ffffff);
    padding: 30px; 
    border-radius: 12px; 
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1); 
    text-align: center;
    margin-bottom: 20px;
}
.header h1 { 
    color: #007bff; 
    font-size: 3rem; 
    font-weight: 800; 
    letter-spacing: 1px;
}
.header p { 
    color: #555555; 
    margin-top: 10px;
    font-size: 1.1rem;
}

/* Sidebar button style for aesthetics */
div[data-testid="stSidebar"] button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 10px 15px;
    font-weight: bold;
    margin-top: 15px;
    transition: background-color 0.3s;
}
div[data-testid="stSidebar"] button:hover {
    background-color: #45a049;
}

/* Card Grid Layout */
.card-grid { 
    display: grid; 
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); /* Slightly wider cards */
    gap: 20px; 
    margin-top: 20px;
}
.internship-card { 
    background-color: #ffffff; 
    padding: 0; /* Remove padding from main card to fit image */
    overflow: hidden;
    border-radius: 12px; 
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); 
    transition: transform 0.3s, box-shadow 0.3s; 
    /* Dynamic border-left color set in Python */
    border-left: 5px solid #007bff; 
}
.internship-card:hover { 
    transform: translateY(-5px); 
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
}

.card-image-container {
    height: 150px;
    overflow: hidden;
}
.card-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.card-content {
    padding: 15px;
}
.card-header { 
    display: flex; 
    justify-content: space-between; 
    align-items: flex-start; 
    margin-bottom: 10px; 
}
.card-title { 
    font-size: 1.4rem; 
    font-weight: 700; 
    color: #333333; 
    flex-grow: 1;
}
.card-location { 
    font-size: 0.9rem; 
    color: #007bff; 
    background-color: #e6f3ff;
    padding: 4px 8px;
    border-radius: 6px;
    font-weight: 500;
}
.card-stipend { 
    font-size: 1.1rem; 
    color: #388e3c; /* Dark Green */
    font-weight: bold; 
    margin-top: 5px; 
}
.card-description { 
    font-size: 0.95rem; 
    color: #666666; 
    margin-top: 10px;
    line-height: 1.4;
}
.card-skills { 
    display: flex; 
    flex-wrap: wrap; 
    gap: 8px; 
    margin-top: 15px;
}
.skill-tag { 
    background-color: #f0f0f0; 
    color: #777777; 
    padding: 6px 12px; 
    border-radius: 20px; 
    font-size: 0.8rem;
    font-weight: 500;
}

/* Utility/Footer */
.empty-results { text-align: center; color: #999999; margin-top: 50px; font-size: 1.2rem; }
footer { text-align: center; margin-top: 50px; padding: 10px; font-size: 0.85rem; color: #aaaaaa; border-top: 1px solid #e0e0e0; }
.help-section { margin-top: 30px; }
.help-section h3 { color: #333333; font-size: 1.8rem; font-weight: bold; margin-bottom: 20px; border-bottom: 2px solid #007bff; padding-bottom: 5px;}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Controls (Must be run before Main Content) ---
st.sidebar.title(text_strings[st.session_state.lang]['sidebar_title'])
st.sidebar.markdown(text_strings[st.session_state.lang]['sidebar_tagline'])

# 1. Language selection (Updates st.session_state.lang using callback)
language_options = {'English': 'en', 'हिंदी': 'hi', 'मराठी': 'mr', 'தமிழ்': 'ta'}
current_index = list(language_options.keys()).index(
    list(language_options.keys())[list(language_options.values()).index(st.session_state.lang)]
)

st.sidebar.selectbox(
    text_strings[st.session_state.lang]['select_language'], 
    options=list(language_options.keys()), 
    index=current_index,
    key='language_select_key', # Key used to store the selection
    on_change=update_language # Callback function to handle update and rerun
)

# 2. Location Filter
all_locations = sorted(list(set([loc.split('(')[0].strip() for loc in df['location'] if loc != 'Remote'])))
location_options = [text_strings[st.session_state.lang]['any']] + all_locations
selected_location = st.sidebar.selectbox(text_strings[st.session_state.lang]['select_location'], options=location_options)

# 3. Work Mode selection
work_mode_options = {
    text_strings[st.session_state.lang]['online']: 'Online',
    text_strings[st.session_state.lang]['offline']: 'Offline',
    text_strings[st.session_state.lang]['any']: 'Any'
}
selected_work_mode_name = st.sidebar.selectbox(text_strings[st.session_state.lang]['work_mode'], options=list(work_mode_options.keys()))
work_mode = work_mode_options[selected_work_mode_name]

# 4. CV Upload is now available (Added the file uploader)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**📄 {text_strings[st.session_state.lang]['upload_cv']}**")
uploaded_file = st.sidebar.file_uploader("", type=["pdf", "txt"], accept_multiple_files=False, key="cv_uploader")

# 5. Search Query & Stipend
st.sidebar.markdown("---")
search_query = st.sidebar.text_input(text_strings[st.session_state.lang]['enter_skills'])
st.sidebar.write(text_strings[st.session_state.lang]['min_stipend'])
min_stipend = st.sidebar.slider("", 0, 50000, 0, step=1000)

# --- Main App Content ---
st.markdown(f"<div class='header'><h1>{text_strings[st.session_state.lang]['header_title']}</h1><p>{text_strings[st.session_state.lang]['header_tagline']}</p></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.write("---")

# --- Recommendation Button Logic ---
if st.sidebar.button(text_strings[st.session_state.lang]['show_recommendations']) or 'initial_run' not in st.session_state:
    st.session_state.initial_run = True

    # --- Filtering logic ---
    filtered_df = df.copy()

    # Filter by Work Mode
    if work_mode == 'Online':
        filtered_df = filtered_df[filtered_df['location'].str.contains('Remote', case=False, na=False)]
    elif work_mode == 'Offline':
        # Exclude 'Remote' locations for Offline mode
        filtered_df = filtered_df[~filtered_df['location'].str.contains('Remote', case=False, na=False)]

    # Filter by Specific Location
    if selected_location != text_strings[st.session_state.lang]['any']:
        filtered_df = filtered_df[filtered_df['location'].str.contains(selected_location, case=False, na=False)]
    
    # Filter by stipend
    filtered_df['stipend_numeric'] = filtered_df['stipend'].str.replace('₹', '').str.replace(',', '').astype(int)
    filtered_df = filtered_df[filtered_df['stipend_numeric'] >= min_stipend]

    # --- Recommendation logic ---
    if search_query and not filtered_df.empty:
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
        # Only show results with a match score > 0
        filtered_df = filtered_df[filtered_df['similarity_score'] > 0] 
    elif search_query and filtered_df.empty:
        st.info("Please broaden your filters to enable skill-based matching.")
    elif not search_query:
        # If no search query, just display all filtered results sorted by stipend
        filtered_df = filtered_df.sort_values(by='stipend_numeric', ascending=False)
        
    # --- Display results ---
    st.write(f"### {text_strings[st.session_state.lang]['recommendations_title']}")

    if not filtered_df.empty:
        st.markdown("<div class='card-grid'>", unsafe_allow_html=True)
        for index, row in filtered_df.iterrows():
            # Logic for dynamic card border color based on match score
            border_color = "#D3D3D3" # Default low match
            if search_query:
                score = row.get('similarity_score', 0)
                if score > 0.5:
                    border_color = "#4CAF50" # Green for high match
                elif score > 0.2:
                    border_color = "#FFA500" # Orange for medium match
                else:
                    border_color = "#007bff" # Blue if no match but matches filter

            st.markdown(f"""
            <div class='internship-card' style='border-left: 5px solid {border_color};'>
                <div class='card-image-container'>
                    <img class='card-image' src="{row['image_url']}" alt="{row['role']} Image">
                </div>
                <div class='card-content'>
                    <div class='card-header'>
                        <div class='card-title'>{row['role']}</div>
                        <div class='card-location'>📍 {row['location'].split('(')[0].strip()}</div>
                    </div>
                    <div class='card-stipend'>💵 {text_strings[st.session_state.lang]['stipend_label']}: {row['stipend']}</div>
                    <div class='card-description'>{row['description']}</div>
                    <div class='card-skills'>
                        {"".join([f"<span class='skill-tag'>💡 {skill.strip()}</span>" for skill in row['skills'].split(',')])}
                    </div>
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

# Using current language strings for expander titles
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
