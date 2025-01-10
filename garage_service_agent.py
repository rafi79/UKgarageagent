import os
import base64
from typing import Optional, Union, List
import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time
import json

class GarageServiceAgent:
    def __init__(self, csv_file):
        """Initialize the multi-model garage service agent"""
        print("Initializing Garage Service Agent...")
        try:
            self.garages_df = pd.read_csv(csv_file)
            self.geolocator = Nominatim(user_agent="garage_service_agent")
            self.geocoded_locations = {}
            
            # Initialize models and prepare TF-IDF
            self.setup_models()
            self.prepare_text_similarity()
            print("Initialization complete!")
            
        except Exception as e:
            error_msg = f"Failed to initialize Garage Service Agent: {str(e)}"
            print(error_msg)
            if st.runtime.exists():
                st.error(error_msg)
            raise
    
    def setup_models(self):
        """Initialize all AI models with secure key management"""
        try:
            # Validate that required secrets exist
            if "api_keys" not in st.secrets:
                raise ValueError("API keys section not found in Streamlit secrets")
                
            required_keys = ["OPENAI_KEY", "GEMINI_KEY", "PERPLEXITY_KEY"]
            missing_keys = [key for key in required_keys if key not in st.secrets["api_keys"]]
            
            if missing_keys:
                raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
            
            # Setup OpenAI (GPT) as main agent
            self.gpt_client = OpenAI(api_key=st.secrets["api_keys"]["OPENAI_KEY"])
            
            # Setup Gemini for media analysis
            genai.configure(api_key=st.secrets["api_keys"]["GEMINI_KEY"])
            self.gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "top_p": 0.95,
                    "top_k": 40,
                }
            )
            
            # Setup Perplexity for service verification
            self.perplexity_api_key = st.secrets["api_keys"]["PERPLEXITY_KEY"]
            self.perplexity_url = "https://api.perplexity.ai/chat/completions"
            
        except Exception as e:
            error_msg = f"Error setting up API clients: {str(e)}"
            print(error_msg)
            if st.runtime.exists():
                st.error(error_msg)
                st.info("Please ensure all API keys are properly configured in Streamlit secrets.")
            raise

    def prepare_text_similarity(self):
        """Prepare TF-IDF vectors for text similarity matching"""
        self.garages_df['search_text'] = self.garages_df.apply(
            lambda x: f"{x['Garage Name']} {x['Location']} {x['City']} {x['Postcode']}",
            axis=1
        ).fillna('')
        
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.garages_df['search_text'])

    def analyze_query_with_gpt(self, query: str, location: str) -> dict:
        """Main agent: GPT analyzes query and coordinates other models"""
        try:
            prompt = f"""
            Analyze this car service request and coordinate the response.
            Query: {query}
            Location: {location}
            Respond with a JSON object containing these fields:
            {{
                "problem_type": "service type description",
                "urgency_level": "High/Medium/Low",
                "required_services": ["needed", "services"],
                "special_considerations": ["considerations"],
                "estimated_duration": "time estimate",
                "next_steps": ["steps"]
            }}
            """
            
            completion = self.gpt_client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            try:
                return json.loads(completion.choices[0].message.content)
            except json.JSONDecodeError:
                return {
                    "problem_type": "Service Request",
                    "urgency_level": "Medium",
                    "required_services": [query],
                    "special_considerations": ["Verify service requirements"],
                    "estimated_duration": "To be determined",
                    "next_steps": ["Contact nearest garage", "Get service quote"]
                }
                
        except Exception as e:
            error_msg = f"GPT analysis error: {str(e)}"
            print(error_msg)
            if st.runtime.exists():
                st.warning("Error during analysis, using fallback response")
            return {
                "problem_type": "Service Request",
                "urgency_level": "Medium",
                "required_services": [query],
                "special_considerations": ["Verify service requirements"],
                "estimated_duration": "To be determined",
                "next_steps": ["Contact nearest garage", "Get service quote"]
            }

    def analyze_media_with_gemini(self, media_path: str, media_type: str) -> str:
        """Analyze media using Gemini"""
        try:
            with open(media_path, "rb") as file:
                if media_type in ["image", "jpg", "jpeg", "png"]:
                    image_data = file.read()
                    image_parts = [{
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(image_data).decode()
                    }]
                    
                    generation_config = {
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                    }
                    
                    safety_settings = {
                        "HARM_CATEGORY_HARASSMENT": "block_none",
                        "HARM_CATEGORY_HATE_SPEECH": "block_none",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
                    }
                    
                    prompt = """
                    Analyze this car problem in detail. Include:
                    1. Visual problem identification
                    2. Potential causes
                    3. Required services
                    4. Urgency level
                    5. Safety considerations
                    Provide a structured response.
                    """
                    
                    response = self.gemini_model.generate_content(
                        contents=[prompt, image_parts[0]],
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    return response.text
                    
                elif media_type in ["video", "mp4", "mov"]:
                    video_data = file.read()
                    response = self.gemini_model.generate_content(
                        contents=[{
                            "mime_type": "video/mp4",
                            "data": base64.b64encode(video_data).decode()
                        }],
                        prompt="Analyze this video of a car problem thoroughly..."
                    )
                    return response.text
                    
                elif media_type in ["audio", "mp3", "wav"]:
                    audio_data = file.read()
                    response = self.gemini_model.generate_content(
                        contents=[{
                            "mime_type": "audio/wav",
                            "data": base64.b64encode(audio_data).decode()
                        }],
                        prompt="Analyze this audio description of a car problem..."
                    )
                    return response.text
                    
        except Exception as e:
            error_msg = f"Media analysis error: {str(e)}"
            print(error_msg)
            if st.runtime.exists():
                st.error(error_msg)
            return f"Could not analyze {media_type}: {str(e)}"

    def analyze_distance_with_gemini(self, user_location: str, garage_location: str, distance: float) -> str:
        """Analyze distance and travel considerations using Gemini"""
        try:
            prompt = f"""
            Analyze the travel distance between customer location and garage:
            Customer Location: {user_location}
            Garage Location: {garage_location}
            Distance: {distance:.2f} km

            Provide a brief analysis of:
            1. Estimated travel time by car
            2. Travel convenience
            3. Alternative transport options if available
            4. Whether the distance is reasonable for:
               - Emergency repairs
               - Regular maintenance
               - Long-term repairs
            
            Keep the response concise and practical.
            """
            
            response = self.gemini_model.generate_content(
                contents=[prompt],
                generation_config={
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            
            return response.text
            
        except Exception as e:
            error_msg = f"Distance analysis error: {str(e)}"
            print(error_msg)
            if st.runtime.exists():
                st.warning(error_msg)
            return "Distance analysis not available"

    def verify_services_with_perplexity(self, garage: dict, required_services: List[str]) -> dict:
        """Verify service availability using Perplexity"""
        try:
            services_list = ", ".join(required_services)
            prompt = f"""
            Given this garage's website ({garage['website']}), verify if {garage['garage_name']} at {garage['address']} 
            offers these services: {services_list}

            Return a JSON object:
            {{
                "has_required_services": true/false,
                "available_services": ["confirmed", "services"],
                "service_notes": "Additional details"
            }}
            """
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.perplexity_url, json=payload, headers=headers)
            result = response.json()
            
            try:
                return json.loads(result['choices'][0]['message']['content'])
            except (json.JSONDecodeError, KeyError, IndexError):
                return {
                    "has_required_services": True,
                    "available_services": required_services,
                    "service_notes": "Service availability needs direct confirmation"
                }
                
        except Exception as e:
            error_msg = f"Service verification error: {str(e)}"
            print(error_msg)
            if st.runtime.exists():
                st.warning(error_msg)
            return {
                "has_required_services": True,
                "available_services": required_services,
                "service_notes": "Error verifying services, please contact garage directly"
            }

    def find_similar_locations(self, query: str, n: int = 5) -> List[dict]:
        """Find similar locations using TFIDF"""
        query_vector = self.tfidf.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)
        
        top_indices = similarities[0].argsort()[-n:][::-1]
        return [
            {
                'garage_name': self.garages_df.iloc[i]['Garage Name'],
                'address': f"{self.garages_df.iloc[i]['Location']} {self.garages_df.iloc[i]['City']} {self.garages_df.iloc[i]['Postcode']}",
                'similarity_score': float(similarities[0][i]),
                'phone': self.garages_df.iloc[i]['Phone'],
                'email': self.garages_df.iloc[i]['Email'],
                'website': self.garages_df.iloc[i]['Website']
            }
            for i in top_indices
        ]

    def find_garages(self, location: str, num_results: int = 5, max_distance: Optional[float] = None) -> List[dict]:
        """Find nearby garages using TFIDF, geolocation, and Gemini analysis"""
        results = []
        print(f"Searching for garages near {location}...")
        
        try:
            similar_garages = self.find_similar_locations(location, n=num_results*2)
            user_coords = self.geocode_location(location)
            
            if user_coords:
                print("Calculating distances and analyzing travel considerations...")
                for garage in similar_garages:
                    garage_coords = self.geocode_location(garage['address'])
                    if garage_coords:
                        distance = geodesic(user_coords, garage_coords).kilometers
                        if max_distance is None or distance <= max_distance:
                            garage['distance'] = distance
                            garage['distance_analysis'] = self.analyze_distance_with_gemini(
                                location, 
                                garage['address'], 
                                distance
                            )
                            results.append(garage)
                
                results = sorted(results, key=lambda x: x['distance'])[:num_results]
            else:
                print("Using similarity-based results...")
                results = similar_garages[:num_results]
                
        except Exception as e:
            error_msg = f"Error finding garages: {str(e)}"
            print(error_msg)
            if st.runtime.exists():
                st.warning(error_msg)
            results = similar_garages[:num_results]
            
        return results

    def geocode_location(self, address: str) -> Optional[tuple]:
        """Geocode location with caching and retries"""
        if address in self.geocoded_locations:
            return self.geocoded_locations[address]
            
        try:
            time.sleep(1)  # Rate limiting
            location = self.geolocator.geocode(
                address,
                timeout=10
            )
            if location:
                coords = (location.latitude, location.longitude)
                self.geocoded_locations[address] = coords
                return coords
                
        except Exception as e:
            print(f"Geocoding error: {e}")
            try:
                time.sleep(2)  # Longer wait for retry
                location = self.geolocator.geocode(
                    address,
                    timeout=15
                )
                if location:
                    coords = (location.latitude, location.longitude)
                    self.geocoded_locations[address] = coords
                    return coords
            except Exception as retry_e:
                print(f"Geocoding retry error: {retry_e}")
                def handle_request(self, 
                      query: Optional[str] = None,
                      media_path: Optional[str] = None,
                      media_type: Optional[str] = None,
                      location: str = None,
                      num_results: int = 5,
                      max_distance: Optional[float] = None) -> dict:
        """Main request handler with GPT as coordinator"""
        try:
            if st.runtime.exists():
                st.write("Analyzing request...")
            else:
                print("Analyzing request...")

            if query:
                analysis = self.analyze_query_with_gpt(query, location)
            elif media_path:
                if st.runtime.exists():
                    st.write(f"Analyzing {media_type} with Gemini...")
                else:
                    print(f"Analyzing {media_type} with Gemini...")
                    
                media_analysis = self.analyze_media_with_gemini(media_path, media_type)
                
                if st.runtime.exists():
                    st.write("Processing Gemini analysis with GPT...")
                else:
                    print("Processing Gemini analysis with GPT...")
                    
                analysis = self.analyze_query_with_gpt(media_analysis, location)
            else:
                raise ValueError("Query or media input required")

            if not analysis:
                raise ValueError("Analysis failed")

            if st.runtime.exists():
                st.write("Finding suitable garages...")
            else:
                print("Finding suitable garages...")
                
            garages = self.find_garages(location, num_results * 2, max_distance)

            if st.runtime.exists():
                st.write("Verifying services...")
            else:
                print("Verifying services...")
                
            verified_garages = []
            for garage in garages:
                if garage.get('website') and garage['website'] != 'No Website':
                    service_info = self.verify_services_with_perplexity(
                        garage, 
                        analysis.get('required_services', [])
                    )
                    if service_info and service_info.get('has_required_services'):
                        garage.update({'service_info': service_info})
                        verified_garages.append(garage)

                if len(verified_garages) >= num_results:
                    break

            if st.runtime.exists():
                st.write("Generating final recommendation...")
            else:
                print("Generating final recommendation...")
                
            recommendation_prompt = f"""
            Based on this analysis, provide a helpful recommendation:
            Problem Analysis: {json.dumps(analysis, indent=2)}
            Available Garages: {json.dumps(verified_garages, indent=2)}
            
            Include:
            1. Problem summary
            2. Recommended actions
            3. Best garage options with reasoning
            4. Additional advice including travel considerations
            """
            
            final_response = self.gpt_client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "user", "content": recommendation_prompt}
                ]
            )

            return {
                "analysis": analysis,
                "garage_results": verified_garages,
                "recommendation": final_response.choices[0].message.content
            }

        except Exception as e:
            error_msg = f"Request handling error: {str(e)}"
            print(error_msg)
            if st.runtime.exists():
                st.error(error_msg)
            return {"error": str(e)}

# Optional main function for testing
def main():
    """Interactive garage service agent testing"""
    if st.runtime.exists():
        st.error("This script should be imported, not run directly in Streamlit")
        return
        
    print("\nWelcome to the Garage Service Agent")
    print("=================================")
    print("I can help you find automotive services in your area.")
    
    try:
        # For testing, you should provide a path to your test CSV file
        agent = GarageServiceAgent('path/to/your/test/garage_data.csv')
        
        while True:
            print("\nHow can I help you today?")
            print("1. Describe your problem")
            print("2. Upload an image/video/audio")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "3":
                print("\nThank you for using the Garage Service Agent. Goodbye!")
                break
                
            location = input("\nEnter your location: ").strip()
            
            try:
                if choice == "1":
                    query = input("\nPlease describe your car problem: ").strip()
                    print("\nProcessing your request...")
                    result = agent.handle_request(
                        query=query,
                        location=location
                    )
                elif choice == "2":
                    media_path = input("\nEnter path to your media file: ").strip()
                    media_type = input("Media type (image/video/audio): ").strip().lower()
                    print("\nAnalyzing your media...")
                    result = agent.handle_request(
                        media_path=media_path,
                        media_type=media_type,
                        location=location
                    )
                
                if result.get("error"):
                    print(f"\nError: {result['error']}")
                else:
                    print("\nAnalysis of Your Request:")
                    print("========================")
                    analysis = result["analysis"]
                    print(f"Problem Type: {analysis.get('problem_type', 'N/A')}")
                    print(f"Urgency Level: {analysis.get('urgency_level', 'N/A')}")
                    print(f"Required Services: {', '.join(analysis.get('required_services', []))}")
                    print(f"Special Considerations: {', '.join(analysis.get('special_considerations', []))}")
                    print(f"Estimated Duration: {analysis.get('estimated_duration', 'N/A')}")
                    
                    print("\nRecommended Garages:")
                    print("===================")
                    for i, garage in enumerate(result["garage_results"], 1):
                        print(f"\n{i}. {garage['garage_name']}")
                        print(f"   Address: {garage['address']}")
                        if 'distance' in garage:
                            print(f"   Distance: {garage['distance']:.2f} km")
                            if 'distance_analysis' in garage:
                                print("\n   Travel Analysis:")
                                print("   " + "\n   ".join(garage['distance_analysis'].split('\n')))
                        if 'service_info' in garage:
                            service_info = garage['service_info']
                            print(f"\n   Services Available: {', '.join(service_info.get('available_services', []))}")
                            print(f"   Service Notes: {service_info.get('service_notes', 'N/A')}")
                        print(f"\n   Contact Information:")
                        print(f"   Phone: {garage['phone']}")
                        if garage['email'] != 'No Email':
                            print(f"   Email: {garage['email']}")
                        if garage['website'] != 'No Website':
                            print(f"   Website: {garage['website']}")
                    
                    print("\nExpert Recommendation:")
                    print("=====================")
                    print(result["recommendation"])
                    
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                print("Please try again with different input.")
            
            print("\n" + "="*60)

    except Exception as e:
        print(f"\nCritical Error: {e}")
        print("Please ensure the CSV file is present and properly formatted.")

if __name__ == "__main__":
    main()
