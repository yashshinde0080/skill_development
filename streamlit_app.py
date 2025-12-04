"""
Streamlit UI for Car Price Prediction

A web-based interface for predicting used car prices using trained ML models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from loguru import logger

from src.models.predict import ModelPredictor
from src.config.config_loader import get_config

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #080808;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #080808;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1565C0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the model predictor (cached for performance)."""
    try:
        model_path = project_root / "models" / "best_model.pkl"
        preprocessor_path = project_root / "models" / "preprocessor.pkl"
        
        predictor = ModelPredictor(
            model_path=model_path,
            preprocessor_path=preprocessor_path
        )
        return predictor, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def get_fuel_types():
    """Get available fuel types."""
    return ["Petrol", "Diesel", "CNG", "LPG", "Electric"]


def get_seller_types():
    """Get available seller types."""
    return ["Individual", "Dealer", "Trustmark Dealer"]


def get_transmission_types():
    """Get available transmission types."""
    return ["Manual", "Automatic"]


def get_owner_types():
    """Get available owner types."""
    return ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]


def single_prediction_tab(predictor):
    """Render the single prediction tab."""
    st.subheader("🚗 Enter Car Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input(
            "Year of Manufacture",
            min_value=1990,
            max_value=2024,
            value=2018,
            step=1,
            help="The year the car was manufactured"
        )
        
        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=1000,
            help="Total kilometers driven"
        )
        
        fuel = st.selectbox(
            "Fuel Type",
            options=get_fuel_types(),
            help="Type of fuel the car uses"
        )
        
        seller_type = st.selectbox(
            "Seller Type",
            options=get_seller_types(),
            help="Type of seller"
        )
        
        transmission = st.selectbox(
            "Transmission",
            options=get_transmission_types(),
            help="Type of transmission"
        )
    
    with col2:
        owner = st.selectbox(
            "Owner Type",
            options=get_owner_types(),
            help="Number of previous owners"
        )
        
        mileage = st.number_input(
            "Mileage (km/l)",
            min_value=0.0,
            max_value=50.0,
            value=18.0,
            step=0.1,
            help="Fuel efficiency in km per liter"
        )
        
        engine = st.number_input(
            "Engine Capacity (CC)",
            min_value=500.0,
            max_value=6000.0,
            value=1500.0,
            step=50.0,
            help="Engine capacity in cubic centimeters"
        )
        
        max_power = st.number_input(
            "Max Power (bhp)",
            min_value=30.0,
            max_value=500.0,
            value=100.0,
            step=5.0,
            help="Maximum power output in brake horsepower"
        )
        
        seats = st.number_input(
            "Number of Seats",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="Number of seats in the car"
        )
    
    st.markdown("---")
    
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        with st.spinner("Calculating price..."):
            try:
                prediction = predictor.predict_single(
                    year=year,
                    km_driven=km_driven,
                    fuel=fuel,
                    seller_type=seller_type,
                    transmission=transmission,
                    owner=owner,
                    mileage=mileage,
                    engine=engine,
                    max_power=max_power,
                    seats=seats
                )
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted Selling Price</h3>
                    <p class="prediction-value">₹{prediction:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Try to get confidence interval
                try:
                    car_data = {
                        'year': year,
                        'km_driven': km_driven,
                        'fuel': fuel,
                        'seller_type': seller_type,
                        'transmission': transmission,
                        'owner': owner,
                        'mileage': mileage,
                        'engine': engine,
                        'max_power': max_power,
                        'seats': seats
                    }
                    confidence = predictor.predict_with_confidence(car_data)
                    if confidence['std'][0] > 0:
                        st.info(f"95% Confidence Interval: ₹{confidence['lower_95'][0]:,.2f} - ₹{confidence['upper_95'][0]:,.2f}")
                except Exception:
                    pass
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")


def batch_prediction_tab(predictor):
    """Render the batch prediction tab."""
    st.subheader("📁 Upload CSV for Batch Predictions")
    
    st.markdown("""
    Upload a CSV file with the following columns:
    - `year`: Year of manufacture
    - `km_driven`: Kilometers driven
    - `fuel`: Fuel type (Petrol/Diesel/CNG/LPG/Electric)
    - `seller_type`: Seller type (Individual/Dealer/Trustmark Dealer)
    - `transmission`: Transmission type (Manual/Automatic)
    - `owner`: Owner type
    - `mileage`: Mileage in km/l
    - `engine`: Engine capacity in CC
    - `max_power`: Max power in bhp
    - `seats`: Number of seats
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview of uploaded data:")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🔮 Predict All", type="primary", use_container_width=True):
                with st.spinner("Making predictions..."):
                    predictions = predictor.predict(df)
                    df['predicted_price'] = predictions
                    
                    st.success(f"✅ Predictions completed for {len(df)} records!")
                    
                    st.write("### Results:")
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Price", f"₹{df['predicted_price'].mean():,.2f}")
                    with col2:
                        st.metric("Min Price", f"₹{df['predicted_price'].min():,.2f}")
                    with col3:
                        st.metric("Max Price", f"₹{df['predicted_price'].max():,.2f}")
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="car_price_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Predict used car prices using Machine Learning</p>", unsafe_allow_html=True)
    
    # Load predictor
    predictor, error = load_predictor()
    
    if error:
        st.error(f"""
        ⚠️ **Model not found!**
        
        Please train the model first by running:
        ```bash
        python scripts/run_training.py --sample-data
        ```
        
        Error: {error}
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts used car selling prices based on various features like:
        - Year of manufacture
        - Kilometers driven
        - Fuel type
        - Transmission type
        - And more...
        
        **Model Performance:**
        - R² Score: ~92%
        - Uses XGBoost/Random Forest ensemble
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.markdown("- Models: Linear Regression, Random Forest, XGBoost")
        st.markdown("- Best Model: XGBoost")
    
    # Main tabs
    tab1, tab2 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction"])
    
    with tab1:
        single_prediction_tab(predictor)
    
    with tab2:
        batch_prediction_tab(predictor)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit | Car Price Prediction v1.0.0</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

