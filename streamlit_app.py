import os
import sys
import streamlit as st
from PIL import Image
import io
from tempfile import NamedTemporaryFile

# Add the repository root to the Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from garage_service_agent import GarageServiceAgent

# Configure Streamlit page
st.set_page_config(
    page_title="UK Smart Garage Assistant",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    div.stMarkdown h1 {
        text-align: center;
        color: #FF4B4B;
        padding-bottom: 1rem;
    }
    div.stMarkdown h2 {
        color: #FF4B4B;
    }
    .garage-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Function to check required secrets
def check_api_secrets():
    """Verify all required API keys are present"""
    required_keys = ["OPENAI_KEY", "GEMINI_KEY", "PERPLEXITY_KEY"]
    missing_keys = []
    
    if "api_keys" not in st.secrets:
        st.error("API keys configuration is missing!")
        st.info("Please configure API keys in your Streamlit secrets.")
        st.stop()
        
    for key in required_keys:
        if key not in st.secrets["api_keys"]:
            missing_keys.append(key)
    
    if missing_keys:
        st.error(f"Missing required API keys: {', '.join(missing_keys)}")
        st.info("Please add the missing keys to your Streamlit secrets.")
        st.stop()

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

# Main title
st.title("🚗 UK Smart Garage Assistant")

# Sidebar configuration
with st.sidebar:
    st.header("Setup & Configuration")
    
    # CSV file upload
    uploaded_file = st.file_uploader("Upload Garage Database (CSV)", type=['csv'])
    
    if uploaded_file is not None and st.session_state.agent is None:
        try:
            # Check API secrets before initializing agent
            check_api_secrets()
            
            # Initialize the agent
            with st.spinner("Initializing garage service agent..."):
                st.session_state.agent = GarageServiceAgent(uploaded_file)
            st.success("✅ Garage database loaded successfully!")
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            st.session_state.agent = None
    
    st.divider()
    
    # Location settings
    location = st.text_input("📍 Your Location", 
                           placeholder="Enter city or postcode",
                           help="Enter your location to find nearby garages")
    
    max_distance = st.slider("🎯 Maximum Distance (km)", 
                           min_value=5, 
                           max_value=100, 
                           value=30,
                           help="Maximum distance to search for garages")
    
    num_results = st.slider("📋 Number of Results", 
                          min_value=1, 
                          max_value=10, 
                          value=5,
                          help="Number of garage recommendations to show")

# Main content area
if st.session_state.agent is None:
    st.info("👈 Please upload your garage database (CSV) file in the sidebar to begin.")
else:
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["💬 Describe Problem", "📸 Upload Media"])
    
    with tab1:
        st.header("Describe Your Car Problem")
        problem_description = st.text_area(
            "What issues are you experiencing with your vehicle?",
            height=150,
            placeholder="Example: My car is making a strange noise from the front wheels when braking..."
        )
        
        if st.button("🔍 Analyze Problem", key="text_analysis"):
            if location:
                with st.spinner("Analyzing your request..."):
                    try:
                        result = st.session_state.agent.handle_request(
                            query=problem_description,
                            location=location,
                            num_results=num_results,
                            max_distance=max_distance
                        )
                        st.session_state.analysis_complete = True
                        st.session_state.current_result = result
                    except Exception as e:
                        st.error(f"Error analyzing request: {str(e)}")
            else:
                st.warning("Please enter your location in the sidebar!")

    with tab2:
        st.header("Upload Media")
        uploaded_media = st.file_uploader(
            "Upload an image/video/audio of the problem",
            type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'mp3', 'wav']
        )
        
        if uploaded_media is not None:
            media_type = uploaded_media.type.split('/')[0]
            
            # Display uploaded media
            if media_type == 'image':
                st.image(uploaded_media, caption="Uploaded Image", use_column_width=True)
            elif media_type == 'video':
                st.video(uploaded_media)
            elif media_type == 'audio':
                st.audio(uploaded_media)
            
            # Create temporary file for processing
            with NamedTemporaryFile(delete=False, suffix=f".{uploaded_media.type.split('/')[-1]}") as tmp_file:
                tmp_file.write(uploaded_media.getvalue())
                media_path = tmp_file.name
            
            if st.button("🔍 Analyze Media", key="media_analysis"):
                if location:
                    with st.spinner(f"Analyzing your {media_type}..."):
                        try:
                            result = st.session_state.agent.handle_request(
                                media_path=media_path,
                                media_type=media_type,
                                location=location,
                                num_results=num_results,
                                max_distance=max_distance
                            )
                            st.session_state.analysis_complete = True
                            st.session_state.current_result = result
                            
                            # Clean up temporary file
                            os.unlink(media_path)
                        except Exception as e:
                            st.error(f"Error analyzing media: {str(e)}")
                            if os.path.exists(media_path):
                                os.unlink(media_path)
                else:
                    st.warning("Please enter your location in the sidebar!")

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.current_result:
        st.divider()
        result = st.session_state.current_result
        
        # Handle errors
        if 'error' in result:
            st.error(f"Analysis Error: {result['error']}")
        else:
            # Analysis Results
            st.header("🔎 Problem Analysis")
            analysis = result["analysis"]
            
            # Create metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Problem Type", analysis.get('problem_type', 'N/A'))
            with col2:
                st.metric("Urgency Level", analysis.get('urgency_level', 'N/A'))
            with col3:
                st.metric("Est. Duration", analysis.get('estimated_duration', 'N/A'))
            
            # Service Details
            st.subheader("📋 Service Requirements")
            scol1, scol2 = st.columns(2)
            with scol1:
                st.write("**Required Services:**")
                for service in analysis.get('required_services', []):
                    st.write(f"• {service}")
            with scol2:
                st.write("**Special Considerations:**")
                for consideration in analysis.get('special_considerations', []):
                    st.write(f"• {consideration}")
            
            # Recommended Garages
            st.header("🏪 Recommended Garages")
            for i, garage in enumerate(result["garage_results"], 1):
                with st.expander(f"#{i} - {garage['garage_name']}", expanded=i==1):
                    gcol1, gcol2 = st.columns(2)
                    
                    with gcol1:
                        st.write("**📍 Location Details**")
                        st.write(f"Address: {garage['address']}")
                        if 'distance' in garage:
                            st.write(f"Distance: {garage['distance']:.2f} km")
                        
                        st.write("\n**📞 Contact Information**")
                        st.write(f"Phone: {garage['phone']}")
                        if garage['email'] != 'No Email':
                            st.write(f"Email: {garage['email']}")
                        if garage['website'] != 'No Website':
                            st.write(f"Website: {garage['website']}")
                    
                    with gcol2:
                        if 'distance_analysis' in garage:
                            st.write("**🚗 Travel Analysis**")
                            st.write(garage['distance_analysis'])
                        
                        if 'service_info' in garage:
                            st.write("\n**🔧 Available Services**")
                            service_info = garage['service_info']
                            st.write(", ".join(service_info.get('available_services', [])))
                            st.write("**Notes:**", service_info.get('service_notes', 'N/A'))
            
            # Expert Recommendation
            st.header("💡 Expert Recommendation")
            st.write(result["recommendation"])

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>UK Smart Garage Assistant - Your trusted partner in vehicle maintenance</p>
        <p>🚗 🔧 🛠️</p>
    </div>
""", unsafe_allow_html=True)
