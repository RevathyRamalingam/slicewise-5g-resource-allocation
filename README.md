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
Input: Device characteristics (use case, LTE category, time of day, packet delay, GBR requirements, etc.)
Output: Predicted slice type (eMBB / URLLC / mMTC)

Benefit:

Automated slice selection - No manual configuration needed  
Optimal resource allocation - Each connection gets appropriate network resources  
Load balancing - Distributes traffic across slices efficiently  
Quality of Service (QoS) - Ensures each application gets what it needs  
Prevents slice failures - Avoids overloading any single slice  

## 📊 Dataset Characteristics and Target Classes

The dataset is structured to support the development of an ML model that can classify the optimal network slice based on device parameters such as, 

1. Usecase - where 5g network is used for smart phone, Gaming, IOT devices, transportation, etc.  
2. LTE/5G Category -Indicates the device or service category as per LTE/5G standards.

Lower numbers (e.g., 1) → basic devices / low capability

Higher numbers (e.g., 22) → advanced or specialized devices (often IoT or high-performance)

3. Technology supported e.g. LTE/5G → standard mobile broadband

IoT (LTE-M, NB-IoT) → low-power IoT communication for sensors and smart devices  
LTE-M means Long Term Evolution for Machine type communication used in smart watches, connected vehicles(non-critical)  
NB-IoT means Narrowband Internet Of Things used in sensors deployed in smart meters, smart housing, smart city trackers.  

4. Day - The day of the week when the data was recorded.

Examples:
sunday
saturday
Used to analyze weekly traffic patterns.

5. Time - Represents the hour of the day (usually 0–23 or 1–24 format).

Example:
1 → 1 AM
14 → 2 PM
Helps in identifying peak and off-peak hours.

6. GBR (Guaranteed Bit Rate)

Specifies whether the network slice provides guaranteed data speed.
GBR → guaranteed network speed (used for critical services emergency calls, autonomous vehicles, remote surgeries, industrial monitoring)
Non-GBR → no guaranteed network speed (normal internet usage, browsing, video streaming)

7. Packet Loss Rate -The fraction of packets lost during transmission.

Examples:

0.01 → 1% packet loss (acceptable for browsing)

0.000001 → extremely low loss (required for critical IoT)
Lower is better network reliability.

8. Packet Delay

The latency (delay) in milliseconds.
Examples:
100 ms → normal mobile internet

10 ms → ultra-low latency (needed for real-time systems)

9. The target output **slice type**  comprises three distinct classes:

**Enhanced Mobile Broadband (eMBB)**:

Focuses on high-bandwidth and high-speed data transmission.
Facilitates activities such as high-definition video streaming, online gaming, and immersive media experiences.

**Ultra-Reliable Low Latency Communication (URLLC)**:

Emphasizes extremely reliable and low-latency connections.
Supports critical applications like autonomous vehicles, industrial automation, satellite launch and communication, remote surgeries, etc

**Massive Machine Type Communication (mMTC)**:

Aims to support a massive number of connected devices.
Enables efficient communication between Internet of Things (IoT) devices, smart cities, and sensor networks.

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
4. Xgboost(accuracy =0.92, roc_auc = 0.99) outperformed all the other models- logistic regression, random forest and decisiontree. 



## 📋 Example Predictions
| Use Case             | Packet Delay | GBR Prediction | Slice Type | Why? |
|----------------------|--------------|----------------|------------|------|
| Gaming               | 10 ms        | Yes            | eMBB       | Needs high speed and moderate latency |
| Autonomous Vehicle   | 5 ms         | Yes            | URLLC      | Critical use case, requires ultra-low latency |
| IoT Sensor           | 100 ms       | No             | mMTC       | Low bandwidth, supports many devices |
| 4K Streaming         | 50 ms        | Yes            | eMBB       | High bandwidth requirement |
| Remote Surgery       | 1 ms         | Yes            | URLLC      |Mission-critical, zero tolerance for delay |


## Dataset Citation

This project uses the Colosseum O-RAN ColO-RAN Dataset:

M. Polese, L. Bonati, S. D'Oro, S. Basagni, T. Melodia, 
"ColO-RAN: Developing Machine Learning-based xApps for Open RAN 
Closed-loop Control on Programmable Experimental Platforms," 
IEEE Transactions on Mobile Computing, pp. 1-14, July 2022.

Dataset: https://github.com/wineslab/colosseum-oran-coloran-dataset

5g dataset is generated using  Colosseum wireless network emulator from 7 Base stations in dense urban area in Rome,Italy across 42 users.