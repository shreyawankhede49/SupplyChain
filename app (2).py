
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('best_classification_model.pkl')
scaler = joblib.load('scaler.pkl') # Load the saved scaler

# Define the feature columns - ensure these match the training data
feature_cols = ['vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate',
                'eta_variation_hours', 'traffic_congestion_level', 'warehouse_inventory_level',
                'loading_unloading_time', 'handling_equipment_availability', 'order_fulfillment_status',
                'weather_condition_severity', 'port_congestion_level', 'shipping_costs',
                'supplier_reliability_score', 'lead_time_days', 'historical_demand',
                'iot_temperature', 'cargo_condition_status', 'route_risk_level',
                'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score',
                'disruption_likelihood_score', 'delay_probability']

st.title('Supply Chain Risk Classification App')
st.write('Enter the feature values below to predict the risk classification.')

# Create input fields for each feature
input_data = {}
for col in feature_cols:
    input_data[col] = st.number_input(f'Enter value for {col}', value=0.0)

if st.button('Predict Risk'):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    st.success(f'Predicted Risk Classification: {prediction[0]}')
