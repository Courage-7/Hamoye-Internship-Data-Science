import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import google.generativeai as genai

# Load data
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Preprocess the data
def preprocess_data(df):
    region_encoder = LabelEncoder()
    event_encoder = LabelEncoder()
    df['region_encoded'] = region_encoder.fit_transform(df['Region'])
    df['event_encoded'] = event_encoder.fit_transform(df['event_type'])
    df['log_fatalities'] = np.log1p(df['fatalities'])
    df['region_event_interaction'] = df['region_encoded'] * df['event_encoded']
    X = df[['fatalities', 'log_fatalities', 'event_encoded', 'region_event_interaction']]
    y = df['region_encoded']
    return df, X, y, region_encoder, event_encoder

# Train Random Forest model
def train_rf_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        predictions = rf_model.predict(X_test_scaled)
        return rf_model, X_test_scaled, y_test, predictions
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, None, None

# Plot feature importance using Streamlit's native chart
def plot_feature_importance(rf_model, X):
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)
    st.bar_chart(feature_importance.set_index('feature')['importance'])
    st.write(feature_importance)

# Generate conflict mitigation strategy
def conflict_mitigation_prompt(conflict_type, region):
    return f"""
    Generate a detailed and actionable mitigation strategy for addressing {conflict_type} in the {region}. The strategy should include the following components:

    1. Immediate Response Actions:
       - Specific emergency measures to be taken to ensure immediate safety and stabilization.
       - Roles and responsibilities of key responders (e.g., security forces, medical personnel, humanitarian organizations).

    2. Medium-Term Interventions:
       - Programs and initiatives to address underlying issues and reduce the recurrence of conflict.
       - Steps to rebuild trust and promote reconciliation among affected communities.

    3. Long-Term Prevention Measures:
       - Structural changes and policies to prevent future conflicts.
       - Education and awareness campaigns to foster a culture of peace and non-violence.

    4. Key Stakeholders to Involve:
       - Identification of essential stakeholders (e.g., government agencies, NGOs, community leaders, international organizations).
       - Their roles and contributions to the mitigation efforts.

    5. Potential Challenges in Implementation:
       - Anticipated obstacles and resistance.
       - Strategies to overcome these challenges and ensure the effectiveness of the mitigation plan.

    Note: Ensure the strategies are well-structured, clearly numbered, and provide practical, actionable steps. Avoid using bold or special characters.
    """

def generate_conflict_mitigation_strategy(api_key, conflict_type, region):
    genai.configure(api_key=api_key)
    # Corrected model method and prompt generation
    model = genai.Model(name='gemini-1.5-flash')
    prompt_text = conflict_mitigation_prompt(conflict_type, region)
    response = model.generate(
        prompt=prompt_text,
        max_tokens=800,
        temperature=0.3,
        top_p=1
    )
    return response.choices[0].text

# Streamlit app layout
st.title('Conflict Data Analysis and Mitigation Strategies')
st.write("This app analyzes conflict data using a Random Forest model and generates conflict mitigation strategies using a Generative AI model.")

# API Key input
api_key = st.text_input("Enter your API key for Generative AI:", type="password")

# File upload
uploaded_file = st.file_uploader("Upload your conflict data CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)

    # Preprocess data
    processed_df, X, y, region_encoder, event_encoder = preprocess_data(df)

    # Train Random Forest model
    rf_model, X_test_scaled, y_test, predictions = train_rf_model(X, y)

    if rf_model:
        # Evaluate model
        class_labels = [
            "Africa",
            "Asia",
            "Middle East",
            "Latin America",
            "Europe",
            "USA/Canada"
        ]
        st.subheader('Model Evaluation')
        st.text(classification_report(y_test, predictions, target_names=class_labels))

        # Plot feature importance
        st.subheader('Feature Importance')
        plot_feature_importance(rf_model, X)

        # Generate conflict mitigation strategy
        st.subheader('Conflict Mitigation Strategy')
        selected_region = st.selectbox('Select Region:', class_labels)
        generate_button = st.button('Generate Mitigation Strategy')
        if generate_button:
            if api_key:
                predicted_conflict_type = event_encoder.inverse_transform([np.argmax(np.bincount(y_test[predictions == region_encoder.transform([selected_region])[0]]))])[0]
                strategy = generate_conflict_mitigation_strategy(api_key, predicted_conflict_type, selected_region)
                st.text_area('Generated Strategy:', value=strategy, height=300)
            else:
                st.error("Please enter your API key.")

        # Display processed data
        st.subheader('Processed Data')
        st.write(processed_df)

        # Download link for the processed data
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(processed_df)

        st.download_button(
            label="Download Processed Data as CSV",
            data=csv,
            file_name='processed_data.csv',
            mime='text/csv',
        )
    else:
        st.error("Model training failed. Please check the data and try again.")
else:
    st.warning("Please upload a CSV file.")
