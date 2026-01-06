import streamlit as st
import requests
import pandas as pd

# Set page configuration
st.set_page_config(page_title="SliceWise 5G Slice Predictor", layout="wide")
st.image("graph_visualization/networkslicing.jpg", width='stretch')
st.title("📡 5G Network Slice Resource Allocation")
st.markdown("Adjust the network metrics below to predict the required Slice Category (**eMBB**, **MTC**, or **URLLC**).")

# Sidebar for Network Load
st.sidebar.header("Global Context")
st.sidebar.image("graph_visualization/sidebar.jpg", width='stretch')
network_load = st.sidebar.selectbox("Network Load", ["peak", "off-peak", "night"])

# Main Layout: Three columns for parameters
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Signal Quality")
    dl_cqi = st.slider("DL CQI (Channel Quality)", 0.0, 15.0, 8.0)
    ul_sinr = st.slider("UL SINR (Signal/Noise dB)", -10.0, 40.0, 34.0)
    dl_mcs = st.slider("DL MCS (Modulation)", 0.0, 28.0, 4.0)

with col2:
    st.subheader("Throughput & Traffic")
    tx_brate = st.number_input("DL Bitrate (Mbps)", min_value=0.0, value=0.009, format="%.4f")
    dl_buffer = st.number_input("DL Buffer (Bytes)", min_value=0, value=0)
    ul_samples = st.number_input("UL Samples", min_value=0, value=42)

with col3:
    st.subheader("Efficiency & Priority")
    grant_ratio = st.slider("Grant Ratio", 0.0, 2.0, 1.76)
    prb_efficiency = st.number_input("PRB Efficiency", min_value=0.0, value=0.0003, format="%.5f")
    latency_proxy = st.number_input("Latency Proxy", min_value=0.0, value=0.0)
    # Computed fields
    mcs_sinr_ratio = dl_mcs / (ul_sinr if ul_sinr != 0 else 1)
    ul_turbo_iters = 1.0 # Fixed default

# Construct the payload
payload = {
    "dl_mcs": dl_mcs,
    "ul_sinr": ul_sinr,
    "tx_brate_downlink_mbps": tx_brate,
    "dl_buffer_bytes": int(dl_buffer),
    "ul_turbo_iters": ul_turbo_iters,
    "dl_cqi": dl_cqi,
    "ul_n_samples": int(ul_samples),
    "network_load": network_load,
    "mcs_sinr_ratio": mcs_sinr_ratio,
    "grant_ratio": grant_ratio,
    "prb_efficiency": prb_efficiency,
    "latency_proxy": latency_proxy
}

if st.button("Predict Slice Category", width='stretch'):
    try:
        # Connect to your FastAPI backend
        response = requests.post("http://localhost:9696/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            category = result["slice_category"]
            probs = result["probabilities"]

            st.divider()
            
            # Display Prediction
            st.header(f"Prediction: :blue[{category}]")
            
            # Display Probabilities as Progress Bars
            cols = st.columns(3)
            for i, (name, val) in enumerate(probs.items()):
                with cols[i % 3]:
                    st.metric(label=name.split()[0], value=f"{val}%")
                    st.progress(val / 100)
                    
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Could not connect to FastAPI server. Is it running on port 8080? Error: {e}")