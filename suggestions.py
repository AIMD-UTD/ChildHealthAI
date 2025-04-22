import google.generativeai as genai

# Configure your API key
genai.configure(api_key="YOUR_API_KEY")

# Load the model (use "gemini-pro" for text-based tasks)
model = genai.GenerativeModel("gemini-pro")

# Function to get suggestion from Gemini
def get_gemini_suggestion(user_prompt):
    response = model.generate_content(user_prompt)
    return response.text

# Example usage
user_input = """You are a helpful assistant designed to support parents and healthcare providers in preventing mental health issues in children.  
Based on the following information, give personalized and actionable suggestions to help reduce mental health risks and support the child's well-being.  

Prediction: {insert_model_prediction_here}
User Inputs:

Age: {age}

Environment: {brief_description_of_environment}

Behavioral Patterns: {user_input_behaviors}

Family History: {family_history_if_any}

Recent Life Events: {events_if_any}

Suggest preventive actions, lifestyle changes, or resources that could help. Keep your tone warm and encouraging, and avoid medical diagnosis."""
suggestion = get_gemini_suggestion(user_input)
print("Gemini Suggestion:", suggestion)
