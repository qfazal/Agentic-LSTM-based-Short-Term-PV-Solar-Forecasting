import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# --- 1. UI Setup ---
st.set_page_config(page_title="GHI Forecasting", page_icon="☀️", layout="wide")
st.title("☀️ Short-Term Solar Irradiance Forecast")
st.write("Upload meteorological sensor data to evaluate the LSTM model's 15-minute ahead forecasting performance.")

# --- 2. Sequence Generator Function ---
def create_daily_sequences(features_data, target_data, window_size=60, horizon=15):
    X_windows, y_targets, target_dates = [], [], []
    expected_delta = pd.Timedelta(minutes=(window_size + horizon - 1))

    for date, daily_df in features_data.groupby(features_data.index.date):
        if len(daily_df) > (window_size + horizon):
            X_data = daily_df.values
            y_data = target_data.loc[daily_df.index].values
            timestamps = daily_df.index
            safe_limit = len(daily_df) - window_size - horizon + 1

            for i in range(safe_limit):
                start_time = timestamps[i]
                target_time = timestamps[i + window_size + horizon - 1]

                if (target_time - start_time) == expected_delta:
                    X_windows.append(X_data[i : i + window_size])
                    y_targets.append(y_data[i + window_size + horizon - 1])
                    target_dates.append(target_time)

    return np.array(X_windows), np.array(y_targets).flatten(), np.array(target_dates)

# --- 3. Load Artifacts (Cached) ---
@st.cache_resource
def load_deployment_artifacts():
    lstm = load_model('solar_lstm_model.keras')
    f_scaler = joblib.load('feature_scaler.pkl')
    t_scaler = joblib.load('target_scaler.pkl')
    return lstm, f_scaler, t_scaler

try:
    model, feature_scaler, target_scaler = load_deployment_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# --- 4. Data Ingestion & Validation ---
uploaded_file = st.file_uploader("Upload Sensor Data (.tab format)", type=["tab", "txt", "csv"])

if uploaded_file is not None:
    # Read the file and dynamically skip the metadata header
    text_io = io.TextIOWrapper(uploaded_file, encoding='utf-8')
    lines = text_io.readlines()

    start_line = 0
    for i, line in enumerate(lines):
        if line.startswith('*/'):
            start_line = i + 1
            break

    uploaded_file.seek(0)
    live_data = pd.read_csv(uploaded_file, sep='\t', skiprows=start_line)

    # Standardize the index
    if 'Date/Time' in live_data.columns:
        live_data['Date/Time'] = pd.to_datetime(live_data['Date/Time'])
        live_data.set_index('Date/Time', inplace=True)
        live_data.sort_index(inplace=True)

    if len(live_data) >= 75:
        st.divider()
        st.subheader("Forecast Configuration")

        # --- 5. Date Filtering (Strictly Unseen Test Days) ---
        test_mask = live_data.index.day > 26
        test_data = live_data[test_mask]

        if len(test_data) == 0:
            st.error("⚠️ No test-set days (days 27–31) found in the uploaded file.")
        else:
            # Extract unique test dates for the dropdown
            unique_test_dates = np.unique(test_data.index.date)

            selected_date = st.selectbox(
                "Select an unseen test day to evaluate (Days 27–31):",
                unique_test_dates
            )

            # --- 6. Execution & Inference ---
            if st.button(f"Generate Forecast for {selected_date}"):
                with st.spinner(f"Processing physical states for {selected_date}..."):

                    # Isolate the exact day chosen by the user
                    day_mask = live_data.index.date == selected_date
                    day_data = live_data[day_mask].copy()

                    if len(day_data) < 75:
                        st.error(f"⚠️ {selected_date} only contains {len(day_data)} valid rows. At least 75 continuous rows are required.")
                    else:
                        # Reconstruct the pipeline exactly as done during training
                        time_of_day = day_data.index.hour + (day_data.index.minute / 60.0)
                        day_data['Time_Sin'] = np.sin(time_of_day * (2. * np.pi / 24))
                        day_data['Time_Cos'] = np.cos(time_of_day * (2. * np.pi / 24))

                        target_col = 'SWD [W/m**2]'
                        features_to_drop = [target_col, 'Height [m]']

                        X_raw = day_data.drop(columns=features_to_drop, errors='ignore').values
                        y_raw = day_data[[target_col]].values

                        X_scaled = feature_scaler.transform(X_raw)
                        y_scaled = target_scaler.transform(y_raw).flatten()

                        X_scaled_df = pd.DataFrame(X_scaled, index=day_data.index)
                        y_scaled_df = pd.Series(y_scaled, index=day_data.index)

                        # Generate sequence windows
                        X_seq, y_seq, target_dates = create_daily_sequences(
                            X_scaled_df, y_scaled_df, window_size=60, horizon=15
                        )

                        if len(X_seq) == 0:
                            st.error("No valid continuous 75-minute sequences found for this specific date.")
                        else:
                            # Run batch inference
                            scaled_preds = model.predict(X_seq)
                            real_preds = target_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
                            real_actuals = target_scaler.inverse_transform(y_seq.reshape(-1, 1)).flatten()

                            # Calculate Daily Metrics
                            daily_mae = mean_absolute_error(real_actuals, real_preds)
                            daily_rmse = root_mean_squared_error(real_actuals, real_preds)

                            # --- 7. Display Dashboard ---
                            st.success(f"Forecast Generated Successfully for {selected_date}!")

                            col1, col2, col3 = st.columns(3)
                            col1.metric(label="Predicted Peak GHI", value=f"{real_preds.max():.2f} W/m²")
                            col2.metric(label="Daily MAE", value=f"{daily_mae:.2f} W/m²")
                            col3.metric(label="Daily RMSE", value=f"{daily_rmse:.2f} W/m²")

                            st.divider()
                            st.subheader(f"Historical Tracking: {selected_date}")

                            # Generate Matplotlib figure
                            fig, ax = plt.subplots(figsize=(12, 5))
                            ax.plot(target_dates, real_actuals, label='Actual GHI (Ground Truth)', color='blue')
                            ax.plot(target_dates, real_preds, label='Predicted GHI (15m Forecast)', color='orange', linestyle='dashed')

                            # Format plot
                            ax.set_ylabel('Global Horizontal Irradiance (W/m²)')
                            ax.set_xlabel('Time of Day')
                            ax.legend()
                            ax.grid(True)

                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            fig.autofmt_xdate()

                            # Render plot
                            st.pyplot(fig)

                             # LLM

                            st.divider()
                            st.subheader("🤖 AI Architectural Diagnostic")
                            st.write("Analyze phase shifts and peak mismatches caused by the lack of spatial CNN features.")

                            # 1. Isolate the window of maximum error
                            error_array = np.abs(real_actuals - real_preds)
                            max_error_idx = np.argmax(error_array)

                            # Define a 2-hour window (8 steps backward, 8 steps forward for 15-min intervals)
                            start_idx = max(0, max_error_idx - 8)
                            end_idx = min(len(real_actuals), max_error_idx + 8)

                            window_dates = target_dates[start_idx:end_idx]
                            window_actuals = real_actuals[start_idx:end_idx]
                            window_preds = real_preds[start_idx:end_idx]

                            # 2. Calculate Diagnostic Metrics
                            actual_peak = np.max(window_actuals)
                            predicted_peak = np.max(window_preds)
                            peak_mismatch = actual_peak - predicted_peak

                            actual_peak_time = window_dates[np.argmax(window_actuals)]
                            predicted_peak_time = window_dates[np.argmax(window_preds)]

                            # Calculate time lag in minutes
                            time_lag_minutes = (predicted_peak_time - actual_peak_time).total_seconds() / 60

                            # 3. Construct the Data Table for the LLM
                            window_df = pd.DataFrame({
                                'Time': window_dates,
                                'Actual_GHI': window_actuals,
                                'Predicted_GHI': window_preds
                            })
                           
                            if st.button("Generate Diagnostic Report"):
    
                              with st.spinner("Analyzing architectural limitations at high speed..."):
                                  try:
                                      from langchain_groq import ChatGroq
                                      from langchain_core.prompts import ChatPromptTemplate
                                      from langchain_core.output_parsers import StrOutputParser
                                      
                                      # 1. Initialize Groq LLM (Using Llama 3.3 70B for fast, technical reasoning)
                                      llm = ChatGroq(
                                          model="llama-3.3-70b-versatile", 
                                          temperature=0.2,
                                          api_key="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" # <--- Hardcoded string
                                      )
                                      
                                      # 2. Construct the LangChain Prompt Template
                                      prompt = ChatPromptTemplate.from_messages([
                                          ("system", """
                                          You are an AI Systems Design Engineer diagnosing time-series forecasting errors. 
                                          Focus strictly on explaining the errors through architectural limitations: specifically, 
                                          why a purely temporal LSTM (trained only on point-sensor data) suffers from phase lag 
                                          and peak mismatch compared to a hybrid CNN-LSTM that utilizes All-Sky Imager spatial data.
                                          """),
                                          ("user", """
                                          Diagnostic Metrics:
                                          - Time of Maximum Error: {max_error_time}
                                          - Peak Mismatch (Actual - Predicted): {mismatch:.2f} W/m²
                                          - Phase Lag: {lag} minutes
                                          
                                          Time-Series Window (2 Hours):
                                          {data_table}
                                          
                                          Provide a concise, 2-paragraph technical explanation of why the LSTM model missed the peak 
                                          and delayed the ramp prediction, explicitly referencing its lack of a spatial CNN module.
                                          """)
                                      ])
                                      
                                      # 3. Build the LCEL Chain
                                      chain = prompt | llm | StrOutputParser()
                                      
                                      # 4. Invoke the chain with variables mapped to the prompt
                                      explanation = chain.invoke({
                                          "max_error_time": target_dates[max_error_idx],
                                          "mismatch": peak_mismatch,
                                          "lag": time_lag_minutes,
                                          "data_table": window_df.to_markdown(index=False)
                                      })
                                      
                                      st.info(explanation)
                                      
                                  except Exception as e:
                                      st.error(f"LangChain/Groq API Error: {e}")
    else:
        st.warning(f"⚠️ The uploaded file only contains {len(live_data)} rows. At least 75 are required.")
