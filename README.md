# RideGuard - Ride Cancellation Predictior

## Overview
RideGuard is an intelligent data mining project that leverages machine learning to predict ride cancellations in ride-sharing services. By analyzing historical Uber ride data and various booking parameters, the system provides real-time predictions to help identify rides that are at high risk of cancellation, particularly those initiated by drivers.

## Features
- **Predictive Analytics**: Uses advanced machine learning algorithms to predict ride cancellation likelihood
- **Real-time Predictions**: Instant analysis based on booking details, driver ratings, customer ratings, and ride characteristics
- **Interactive Web Interface**: User-friendly Streamlit application for easy prediction access
- **Comprehensive Data Analysis**: Processes multiple factors including:
  - Booking value and ride distance
  - Driver and customer ratings
  - Vehicle type and payment method
  - Time-based features (hour of day, peak hours, weekends)
  - Location-based data (pickup and drop locations)
  - Average VTAT (Vehicle Turnaround Time) and CTAT (Customer Turnaround Time)

## Technology Stack
- **Machine Learning**: Decision Tree model (best accuracy) with complete preprocessing pipeline
- **Frontend**: Streamlit for interactive web application
- **Data Processing**: Pandas for data manipulation and analysis
- **Model Persistence**: Joblib for model serialization
- **Python Libraries**: NumPy, Scikit-learn
