import os
import sys
import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
from tempfile import NamedTemporaryFile

# Add the repository root to the Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from garage_service_agent import GarageServiceAgent  # This imports your existing code

# Set page configuration
st.set_page_config(
    page_title="UK Smart Multi Garage Agent",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS to enhance the appearance
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Main title with emoji decoration
st.title("🚗 UK Smart Multi Garage Agent 🔧")

# Sidebar for file upload and location input
with st.sidebar:
    st.header("Setup & Configuration")
    
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload UK Garages Database (CSV)", type=['csv'])
    
    if uploaded_file is not None and st.session_state.agent is None:
        # Initialize the agent with the uploaded CSV
        st.session_state.agent = GarageServiceAgent(uploaded_file)
        st.success("✅ Database loaded successfully!")
    
    st.divider()
    
    # Location input
    user_location = st.text_input("📍 Your Location", 
                                 placeholder="Enter your city or postcode")
    
    max_distance = st.slider("🎯 Maximum Distance (km)", 
                           min_value=5, 
                           max_value=100, 
                           value=30)
    
    num_results = st.slider("📋 Number of Results", 
                          min_value=1, 
                          max_value=10, 
                          value=5)

# Main content area
if st.session_state.agent is None:
    st.info("👈 Please upload the UK Garages database (CSV) file in the sidebar to begin.")
else:
    # Create two columns for input methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📝 Describe Your Problem")
        problem_description = st.text_area(
            "What's wrong with your vehicle?",
            height=150,
            placeholder="Describe the issue you're experiencing..."
        )
        if st.button("🔍 Find Garages (Text)"):
            if user_location:
                with st.spinner("Analyzing your request..."):
                    result = st.session_state.agent.handle_request(
                        query=problem_description,
                        location=user_location,
                        num_results=num_results,
                        max_distance=max_distance
                    )
                    st.session_state.analysis_complete = True
                    st.session_state.result = result
            else:
                st.error("Please enter your location in the sidebar!")

    with col2:
        st.header("📸 Upload Media")
        uploaded_media = st.file_uploader(
            "Upload an image/video/audio of the problem",
            type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'mp3', 'wav']
        )
        
        if uploaded_media is not None:
            if uploaded_media.type.startswith('image'):
                st.image(uploaded_media, caption="Uploaded Image", use_column_width=True)
            elif uploaded_media.type.startswith('video'):
                st.video(uploaded_media)
            elif uploaded_media.type.startswith('audio'):
                st.audio(uploaded_media)
                
            media_type = uploaded_media.type.split('/')[0]
            
            # Save uploaded file temporarily
            with NamedTemporaryFile(delete=False, suffix=f".{uploaded_media.type.split('/')[-1]}") as tmp_file:
                tmp_file.write(uploaded_media.getvalue())
                media_path = tmp_file.name
            
            if st.button("🔍 Find Garages (Media)"):
                if user_location:
                    with st.spinner("Analyzing your media..."):
                        result = st.session_state.agent.handle_request(
                            media_path=media_path,
                            media_type=media_type,
                            location=user_location,
                            num_results=num_results,
                            max_distance=max_distance
                        )
                        st.session_state.analysis_complete = True
                        st.session_state.result = result
                else:
                    st.error("Please enter your location in the sidebar!")

    # Display results if analysis is complete
    if st.session_state.analysis_complete and hasattr(st.session_state, 'result'):
        st.divider()
        
        # Error handling
        if 'error' in st.session_state.result:
            st.error(f"An error occurred: {st.session_state.result['error']}")
        else:
            # Analysis results
            st.header("🔎 Analysis Results")
            analysis = st.session_state.result["analysis"]
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("Problem Type", analysis.get('problem_type', 'N/A'))
            with cols[1]:
                st.metric("Urgency Level", analysis.get('urgency_level', 'N/A'))
            with cols[2]:
                st.metric("Estimated Duration", analysis.get('estimated_duration', 'N/A'))
            
            # Required Services and Special Considerations
            st.subheader("📋 Service Details")
            service_col1, service_col2 = st.columns(2)
            with service_col1:
                st.write("**Required Services:**")
                for service in analysis.get('required_services', []):
                    st.write(f"- {service}")
            with service_col2:
                st.write("**Special Considerations:**")
                for consideration in analysis.get('special_considerations', []):
                    st.write(f"- {consideration}")
            
            # Garage Results
            st.header("🏪 Recommended Garages")
            for i, garage in enumerate(st.session_state.result["garage_results"], 1):
                with st.expander(f"{i}. {garage['garage_name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Contact Information:**")
                        st.write(f"📞 Phone: {garage['phone']}")
                        if garage['email'] != 'No Email':
                            st.write(f"📧 Email: {garage['email']}")
                        if garage['website'] != 'No Website':
                            st.write(f"🌐 Website: {garage['website']}")
                    
                    with col2:
                        st.write("**Location Details:**")
                        st.write(f"📍 Address: {garage['address']}")
                        if 'distance' in garage:
                            st.write(f"🚗 Distance: {garage['distance']:.2f} km")
                    
                    if 'distance_analysis' in garage:
                        st.write("**Travel Analysis:**")
                        st.write(garage['distance_analysis'])
                    
                    if 'service_info' in garage:
                        service_info = garage['service_info']
                        st.write("**Available Services:**")
                        st.write(", ".join(service_info.get('available_services', [])))
                        st.write("**Service Notes:**")
                        st.write(service_info.get('service_notes', 'N/A'))
            
            # Expert Recommendation
            st.header("💡 Expert Recommendation")
            st.markdown(st.session_state.result["recommendation"])

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>UK Smart Multi Garage Agent - Your trusted partner in vehicle maintenance</p>
        <p>🚗 🔧 🛠️</p>
    </div>
""", unsafe_allow_html=True)
