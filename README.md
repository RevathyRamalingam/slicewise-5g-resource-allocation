## 📡⚡slicewise-5g-resource-allocation
SliceWise: ML-Powered Resource Allocation for 5G Network Slicing

## 🎯 Problem Statement

Cellular communications with the advent of 5G mobile networks, demand high-reliable communication, ultra-low latency, increased capacity, enhanced security, and high-speed user connectivity. To meet these user demands with rising population and increased network usage, mobile operators require a programmable solution capable of supporting multiple independent tenants on a single physical infrastructure. The advent of 5G networks facilitates end-to-end resource allocation through Network Slicing (NS), which allows for the division of the network into distinct virtual slices.  
Network slicing in 5G stands as a pivotal feature for next-generation wireless networks, delivering substantial benefits to both mobile operators and businesses. This **slicewise-5g-resource-allocation** project aims to develop a Machine Learning (ML) model that can predict the optimal network slice based on key network and device parameters. This project helps to manage network load balancing and addresses network slice failures.

### The Real-World Problem:

When a device connects to a 5G network, the network needs to decide: "Which slice should handle this connection?"

A smartphone streaming Netflix needs high bandwidth (eMBB)
A self-driving car needs ultra-low latency (URLLC)
An IoT temperature sensor needs minimal resources but massive connectivity (mMTC)

### How slicewise-5g-resource-allocation Helps?

Given a set of network features (network_load, , ul_sinr, etc), the goal is to predict one of the following network slice id:

| Class | Category             |
|-------|----------------------|
| 0     | eMBB                 |
| 1     | mMTC                 |
| 2     | URLLC                |

This is a **multiclass classification** problem where the model must predict one of the three classes.

### 📊 Dataset Characteristics and Target Classes 

Input: Device characteristics ('timestamp','num_ues', 'imsi', 'rnti', 'slicing_enabled', 'slice_id',
       'slice_prb', 'power_multiplier', 'scheduling_policy', 'dl_mcs',
       'dl_n_samples', 'dl_buffer_bytes', 'tx_brate_downlink_mbps',
       'tx_pkts_downlink', 'tx_errors_downlink_pct', 'dl_cqi', 'ul_mcs',
       'ul_n_samples', 'ul_buffer_bytes', 'rx_brate_uplink_mbps',
       'rx_pkts_uplink', 'rx_errors_uplink_pct', 'ul_rssi', 'ul_sinr', 'phr',
       'sum_requested_prbs', 'sum_granted_prbs', 'dl_pmi', 'dl_ri', 'ul_n',
       'ul_turbo_iters','hour','network_load','mcs_sinr_ratio', 'grant_ratio',
       'prb_efficiency', 'latency_proxy')

#### network slicing input parameters:

timestamp: Time when the measurement was recorded. It is in Unix/Epoch milliseconds format. 

num_ues: Number of user equipment (devices) connected to the network. 

imsi: International Mobile Subscriber Identity, unique identifier for the subscriber. 

rnti: Radio Network Temporary Identifier, temporary identifier assigned to UE during connection. 

slicing_enabled: Boolean flag indicating whether network slicing is active. 

slice_id: Unique identifier for the specific network slice(target variable to be predicted)

slice_prb: Physical Resource Blocks allocated to this slice. 

power_multiplier: Scaling factor for transmission power allocation. 

scheduling_policy: Algorithm used for resource scheduling (e.g., round-robin, proportional fair) 

dl_mcs: Downlink Modulation and Coding Scheme index indicating data rate. 

dl_n_samples: Number of downlink samples collected during measurement period. 

dl_buffer_bytes: Amount of data waiting in downlink buffer. 

tx_brate_downlink_mbps: Transmission bit rate for downlink in megabits per second. 

tx_pkts_downlink: Number of packets transmitted in downlink. 

tx_errors_downlink_pct: Percentage of transmission errors in downlink. 

dl_cqi: Downlink Channel Quality Indicator reported by UE. 

ul_mcs: Uplink Modulation and Coding Scheme index. 

ul_n_samples: Number of uplink samples collected during measurement period. 

ul_buffer_bytes: Amount of data waiting in uplink buffer. 

rx_brate_uplink_mbps: Reception bit rate for uplink in megabits per second. 

rx_pkts_uplink: Number of packets received in uplink. 

rx_errors_uplink_pct: Percentage of reception errors in uplink. 

ul_rssi: Uplink Received Signal Strength Indicator. 

ul_sinr: Uplink Signal-to-Interference-plus-Noise Ratio. 

phr: Power Headroom Report indicating remaining transmission power capacity. 

sum_requested_prbs: Total Physical Resource Blocks requested by UE. 

sum_granted_prbs: Total Physical Resource Blocks actually allocated to UE. 

dl_pmi: Downlink Precoding Matrix Indicator for MIMO transmission. 

dl_ri: Downlink Rank Indicator specifying number of spatial layers. 

ul_n: Uplink parameter (likely related to HARQ or turbo decoding). 

ul_turbo_iters: Number of turbo decoder iterations used in uplink. 

hour: Hour of day when measurement was taken(engineered/derived feature from timestamp feature)

network_load: Overall network utilization or congestion level(engineered/derived feature from timestamp). 

mcs_sinr_ratio: Derived ratio between MCS and SINR indicating link adaptation efficiency(engineered/derived feature from dl_mcs and ul_sinr). 

grant_ratio: Ratio of granted PRBs to requested PRBs(engineered/derived feature from sum_granted_prbs and sum_requested_prbs ). 

prb_efficiency: Measure of how efficiently allocated PRBs are being utilized(engineered/derived feature from tx_brate_downlink_mbps and sum_granted_prbs) 

latency_proxy: Estimated or derived latency metric for the connection(engineered/derived feature from dl_buffer_bytes and tx_brate_downlink_mbps ). 

Output: Predicted network slice id (eMBB / mMTC / URLLC) 
The target output **slice id**  comprises three distinct classes: 

**Enhanced Mobile Broadband (eMBB)**: 

Focuses on high-bandwidth and high-speed data transmission. 
Facilitates activities such as high-definition video streaming, online gaming, and immersive media experiences.

**Ultra-Reliable Low Latency Communication (URLLC)**: 

Emphasizes extremely reliable and low-latency connections. 
Supports critical applications like autonomous vehicles, industrial automation, satellite launch and communication, remote surgeries, etc 

**Massive Machine Type Communication (mMTC)**: 

Aims to support a massive number of connected devices. 
Enables efficient communication between Internet of Things (IoT) devices, smart cities, and sensor networks. 

Benefit: 

Automated slice selection - No manual configuration needed  
Optimal resource allocation - Each connection gets appropriate network resources  
Load balancing - Distributes traffic across slices efficiently  
Quality of Service (QoS) - Ensures each application gets what it needs  
Prevents slice failures - Avoids overloading any single slice  

## 📁 Project Structure

The project folder consists of the following files and directories:

### Root-level Files:
- `.python_version` — Specifies the Python version for reproducibility.
- `Dockerfile` — Used for deploying the FASTAPI network slice Prediction service in Docker.
- `requirements.txt` — Lists all the required Python libraries (e.g., pandas, numpy, scikit-learn, fastapi, pydantic) for cloud deployment on Render.
- `notebook.ipynb` — Jupyter notebook with the following steps:
  - Fetch dataset, preprocess, and clean data.
  - Perform correlation and mutual information analysis.
  - Check target distribution.
  - Hyperparameter tuning with cross-validation.
  - Evaluate models like Logistic Regression, Decision Tree, and Random Forest.
- `uv.lock` — Locks the specific versions of dependencies to ensure reproducibility.
- `pyproject.toml` — Contains the dependencies for the Uvicorn project. 

### Files Inside Folders:
-**src/**: Contains 
       `predict.py` the FASTAPI app to make predictions 
       `train.py`  Saves the final model as `xgboost_model.bin` in the `model` directory after dataset cleaning, hyperparameter tuning, and model training (done in the Jupyter notebook `notebook.ipynb`). 
       `notebook.py` - run this script to see how hyperparameter tuning and model is evaluated against performance like recall, precision and F1 scores.
       `combined_slice_dataset.csv` combines 3 csv files inside data folder and merges into one.
- **tests/**: contains pytest files test_train.py
       `__init__.py`  added for running pytest
- **model/**: Contains the pickle-saved model named `xgboost_model.bin`.
- **data/**: Contains 3 different dataset `.csv` from different Base station generated by Colosseum wireless network emulator 
- **graph_visualization/**:contains graphs to evaluate and compare the metric such as precision, recall, f1, correlation, mi scores, accuracy and roc_auc scores
contains distribution of target graph, correlation matrix, Confusion matrix, DecisionTree Hyperparameter tuning graphs and random Forest  Hyperparameter tuning heatmap and Model comparision.
- **output_screenshots/** : contains the screenshots of output
---
🎥 Demo Videos on Cloud Deployment

Part 1: Cloud Deployment Overview and testing[https://www.loom.com/share/6632698ca88f424c886b87c38bfc676a]

Part 2: Pydantic validation on API[https://www.loom.com/share/4cefb723b71d4d3088d9f00128ad1358]

---
TRAINING THE MODEL

You can run the notebook.py script to see how the model is trained and hyperparameter tuning is done for several algorithms such as LogisticRegression, DecisionTree and randomForestRegressor.
go to project root folder and run below command
> python src/notebook.py
---

## Observation

1. Initially started with kaggle dataset on 5g network slice classification. But the dataset was not balanced and had only 7 features most of them were static features. The model simply memorized the values and gave accurate predictions(ROC 1.0). Such models can fail in realtime when the features are not constant. So took a dataset from colosseum dataset and it is balanced(imbalance ratio = 1.007) and it had more than 30 features. 
2. Static features such as ['timestamp','num_ues', 'imsi', 'rnti', 'slicing_enabled', 'slice_id',
       'slice_prb', 'power_multiplier', 'scheduling_policy', 
         'dl_n_samples', 
        'tx_errors_downlink_pct', 'ul_mcs', 'ul_buffer_bytes', 'rx_brate_uplink_mbps',
       'rx_pkts_uplink', 'rx_errors_uplink_pct', 'ul_rssi',  'phr',
       'sum_requested_prbs', 'sum_granted_prbs', 'dl_pmi', 'dl_ri', 'ul_n','tx_pkts_downlink'] are removed from the dataset as they are constant values for each slice type and model simply memorizes the values and gives accurate predictions(ROC 1.0). Such models can fail in realtime when the features are not constant and hence they are removed. 
Before removing static features, the model was giving precision, recall, F1score at 1.0 and roc_auc of 1.0. There was label leakage as most of the network features were constant for each slice type. 
3. network_load feature is engineered based on timestamp feature and is used to classify the network load as peak, off-peak and night. 
4. mcs_sinr_ratio', 'grant_ratio','prb_efficiency', 'latency_proxy' features were also derived from dl_mcs, ul_sinr and other input features. 
5. Xgboost(accuracy =0.96, roc_auc = 0.997, F1score = 0.965) outperformed all the other models- logistic regression, random forest and decisiontree. 
6. The model was 10% not accurate for critical mission applications(Recall =0.90), after engineering features such as
'mcs_sinr_ratio', 'grant_ratio', 'prb_efficiency', 'latency_proxy'
mcs_sinr_ratio - This feature relates the data rate capability (MCS) to the actual signal quality (SINR)
grant_ratio - Captures how efficiently the network allocates resources to user requests
prb_efficiency - Measures how effectively allocated spectrum resources are actually used
Helped distinguish between simple capacity constraints and actual network anomalies
Reduced false negatives where raw PRB usage appeared normal but efficiency was poor
latency_proxy - Provides an indirect measure of end-to-end performance degradation
Helped catch cascade failures and multi-factor degradation scenarios
7. pytests were written to cover unit testing scenarios
8. Solution is docker containerized and deployed in Render cloud ,the streamlit UI interacts with backendd FASTAPI server to predict the network slice as per the user network pattern.

## Future Scope 
1. Real-Time Dynamic Slicing 

Implement automatic slice reconfiguration as network conditions change, allowing slices to adapt on-the-fly to traffic surges or failures without manual intervention

2. Anomaly Detection & Security

Add ML models to detect unusual slice behavior, potential threats/attacks, or performance degradation, with automated alerts and suggest corrective actions to operator to maintain network reliability and strong customer base.

3. include SHAP features to explain why AI predicted slice id and the reason behind that. This will be useful for network operator to understand the model and support customer better.

## 🛠 Steps to Run the Project Locally

### Option 1: Using Docker

1. Clone the repository:
    ```bash
    git clone https://github.com/RevathyRamalingam/obesity-class-prediction.git
    ```

2. Build the Docker image:
    ```bash
    docker build -t obesity-prediction .
    ```

3. Run the Docker container:
    ```bash
    docker run -it -p 9696:9696 obesity-prediction:latest
    ```

4. Open your browser and navigate to:
    ```
    http://127.0.0.1:9696/docs
    ```

5. Provide the following JSON input to test the prediction:
    ```json
    {
      "age": 39,
      "gender": "female",
      "height": 1.51,
      "weight": 72,
      "calc": "no",
      "favc": "yes",
      "fcvc": 2.396265,
      "ncp": 1.073421,
      "scc": "no",
      "smoke": "no",
      "ch2o": 1.5,
      "family_history_with_overweight": "yes",
      "faf": 0.022598,
      "tue": 0.061282,
      "caec": "sometimes",
      "mtrans": "automobile"
    }
    ```

6. The output will look like this:
    ```json
    {
      "Health_category": "overweight_level_ii",
      "probabilities": {
        "insufficient_weight": 0,
        "normal_weight": 0.1317,
        "obesity_type_i": 31.7571,
        "obesity_type_ii": 0,
        "obesity_type_iii": 0,
        "overweight_level_i": 4.5125,
        "overweight_level_ii": 63.5986
      }
    }
    ```

Alternatively, you can also use a **curl** command to see the output in the CLI:
```bash
curl -X 'POST' \
  'http://localhost:9696/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "age":39,
    "gender": "female",
    "height": 1.51,
    "weight": 72,
    "calc": "no",
    "favc": "yes",
    "fcvc": 2.396265,
    "ncp": 1.073421,
    "scc": "no",
    "smoke": "no",
    "ch2o": 1.5,
    "family_history_with_overweight": "yes",
    "faf": 0.022598,
    "tue": 0.061282,
    "caec": "sometimes",
    "mtrans": "automobile"
  }'
  ```
Option 2: Using Uvicorn

Clone the repository:

git clone https://github.com/RevathyRamalingam/obesity-class-prediction.git

cd obesity-class-prediction

Install Uvicorn and FastAPI:

pip install uvicorn fastapi

Run the Uvicorn server:

uvicorn main:app --host 0.0.0.0 --port 9696

Open your browser and go to below url,provide the same JSON input as before to get obesity prediction

http://127.0.0.1:9696/docs

---

☁️ Steps to Deploy the Project in the Cloud (Render)
There are two ways by which you can run the project in cloud:
1. pulling the docker image from Dockerhub 
Select image from dockerhub with the name "docker pull revathy1/obesity-predict:1" and choose deploy.
It will automatically run the image from docker hub.

2. Deploying it manually 
The settings for Render are also available as screenshots in output_screenshots folder.Below is the detailed steps for the deployment.

Go to Render.

Sign in to your account or create a new one.

Once logged in, go to the Dashboard on Render.

Click the New button in the top-right corner and select Web Service (or another appropriate service type).

You'll be prompted to Connect GitHub to Render if you haven't done so already. Follow the steps to authorize Render to access your GitHub repositories.

After connecting GitHub, select the repository:

https://github.com/RevathyRamalingam/obesity-class-prediction

In Settings, fill in the following:

General:

Name: Choose a name for your project.

Region: Select the region closest to your geographical location.

Deploy & Build:

Repository: https://github.com/RevathyRamalingam/obesity-class-prediction

Branch: master

Build Command: echo "Hello"

Start Command: pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 9696 --reload

Choose the default settings for the rest of the fields.

After deployment, you will see the URL where the service is running in the cloud.

Click on the link to access the Prediction Service and test it with the sample JSON input provided earlier.

Error Handling (Example)
If you provide an extra field in the input JSON, you'll receive a 422 Unprocessable Entity error:

{
  "detail": [
    {
      "type": "extra_forbidden",
      "loc": ["body", "new"],
      "msg": "Extra inputs are not permitted",
      "input": 9
    }
  ]
}

---

## Dataset Citation 

This project uses the Colosseum O-RAN ColO-RAN Dataset:

M. Polese, L. Bonati, S. D'Oro, S. Basagni, T. Melodia, 
"ColO-RAN: Developing Machine Learning-based xApps for Open RAN 
Closed-loop Control on Programmable Experimental Platforms," 
IEEE Transactions on Mobile Computing, pp. 1-14, July 2022.

Dataset: https://github.com/wineslab/colosseum-oran-coloran-dataset

5g dataset is generated using  Colosseum wireless network emulator from 7 Base stations in dense urban area in Rome,Italy across 42 users.

## Acknowledgement

I would like to thank Alexey Grigorev for conducting this mlzoomcamp course in DataTalksClub enabling me to do such an interesting capstone project in network side and Michael Polese for providing awesome 5g dataset.