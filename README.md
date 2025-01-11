# 🚗 UK Garage Service Agent

An intelligent multi-model system that helps users find and connect with automotive service providers in the UK. The system uses GPT, Gemini, and Perplexity AI to analyze service requests, process media inputs, and provide tailored garage recommendations.

## ✨ Features

- **Multi-Modal Input Processing**
  - Text-based problem descriptions
  - Image analysis of car issues
  - Video processing support
  - Audio input analysis

- **Intelligent Service Analysis**
  - Problem type classification
  - Urgency level assessment
  - Service requirement identification
  - Duration estimation
  - Special considerations detection

- **Location-Based Garage Finding**
  - Text similarity matching
  - Geolocation-based distance calculation
  - Travel time and convenience analysis
  - Multi-factor garage ranking

- **Service Verification**
  - Automated service availability checking
  - Website content analysis
  - Real-time service matching

- **User Interface Options**
  - Command-line interface
  - Streamlit web application
  - Interactive response system

## 🛠️ Technical Stack

- **AI Models**
  - OpenAI GPT (Primary Analysis)
  - Google Gemini (Media Processing)
  - Perplexity AI (Service Verification)

- **Core Libraries**
  - pandas: Data management
  - scikit-learn: Text similarity
  - geopy: Location services
  - streamlit: Web interface

## 📋 Prerequisites

- Python 3.8+
- API keys for:
  - OpenAI
  - Google Gemini
  - Perplexity AI
- Garage database CSV file

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/uk-garage-agent.git
cd uk-garage-agent
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
PERPLEXITY_API_KEY=your_perplexity_key
```

## 💻 Usage

### Command Line Interface

Run the main script:
```bash
python garage_service_agent.py
```

Follow the interactive prompts to:
1. Describe your car problem or upload media
2. Enter your location
3. Review analysis and recommendations

### Web Interface

Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The web interface provides:
- Easy API key configuration
- CSV database upload
- Interactive service request form
- Media upload capability
- Structured results display

## 📊 Data Structure

The garage database CSV should include the following columns:
- Garage Name
- Location
- City
- Postcode
- Phone
- Email
- Website

## 🤖 System Architecture

The system operates in three main stages:

1. **Input Processing**
   - Text analysis via GPT
   - Media processing via Gemini
   - Location validation

2. **Garage Matching**
   - Text similarity computation
   - Geolocation processing
   - Distance calculation
   - Travel analysis

3. **Service Verification**
   - Website content analysis
   - Service matching
   - Availability confirmation

## 🔒 Security Notes

- API keys are stored locally in `.env`
- Temporary files are used for media processing
- User location data is not permanently stored
- All API communications use secure endpoints

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT API
- Google for Gemini API
- Perplexity AI for service verification
- Contributors and testers

## ⚠️ Disclaimer

This system provides recommendations based on available data and AI analysis. Always verify critical information directly with garages and follow proper vehicle maintenance procedures.
