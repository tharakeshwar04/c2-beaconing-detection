# 🚨 C2 Beaconing Discovery via Periodicity and Behavioral Features in Network Flow Logs

## 📌 Overview

This project presents a Ai/Machine Learning-based system for detecting Command-and-Control (C2) beaconing behavior using network traffic flow data.

The system is built using the CTU-13 botnet dataset and integrates a trained Random Forest model into a SOC-style Security Operations Center dashboard using Gradio.

C2 beaconing is commonly seen when malware or botnets repeatedly communicate with an external command server at regular intervals. This project focuses on identifying those repeated and periodic communication patterns using pair-level traffic analysis.

---

## 🖥️ Dashboard Preview
<img width="1600" height="760" alt="WhatsApp Image 2026-05-01 at 2 46 14 PM" src="https://github.com/user-attachments/assets/2ab44173-c9d1-46df-8bae-581e1c147ec5" />

## 🎯 Objectives

Detect suspicious C2 beaconing behavior using flow-based network data

Analyze repeated communication between source and destination pairs

Extract timing-based and traffic-based features from network flows

Use machine learning to identify suspicious beaconing patterns

Generate severity levels for flagged traffic

Provide human-readable reason codes for SOC analysts

Simulate a SOC-style dashboard for security investigation

Export flagged results as downloadable CSV reports

## 📊 Dataset

CTU-13 Botnet Dataset

Network traffic flow data

Scenario used: Scenario 1

Includes traffic types such as:

Botnet traffic

Normal background traffic

Legitimate communication

The dataset is useful for studying botnet behavior because it contains realistic infected-host communication patterns. In this project, the dataset is used to identify repeated outbound communication that may indicate C2 beaconing.

## Data Preprocessing

The preprocessing pipeline converts raw flow data into a standard format that can be used by the model.

Supported input formats include both standard CSV columns and CTU-style columns.

Standardized columns:
timestamp
src_ip
dst_ip
dst_port
protocol
bytes
packets

CTU-style column mapping:

| Standard Column | CTU-Style Column |
| --------------- | ---------------- |
| timestamp       | StartTime        |
| src_ip          | SrcAddr          |
| dst_ip          | DstAddr          |
| dst_port        | Dport            |
| protocol        | Proto            |
| bytes           | TotBytes         |
| packets         | TotPkts          |

Preprocessing steps include:

Converted timestamp values into datetime format

Converted destination ports, bytes, and packets into numeric values

Handled missing and invalid values

Removed incomplete records

Sorted traffic by timestamp

Grouped flows into communication pairs

Filtered pairs with too few flows for meaningful beaconing analysis

## 🔗 Pair-Level Feature Engineering

The system groups network flows using the following communication pair: src_ip, dst_ip, dst_port, protocol
For each communication pair, the system extracts behavioral features such as:

Flow count

Duration in seconds

Mean inter-arrival time

Standard deviation of inter-arrival time

Coefficient of variation of inter-arrival time

Median inter-arrival time

Near-median interval consistency

Average bytes

Byte variation

Average packets

Packet variation

These features help identify whether a host is contacting another system repeatedly at regular intervals, which is a common sign of C2 beaconing.

## 🧠 Model Used
## 🔹 Random Forest Classifier

The main machine learning model used in this project is Random Forest.

Random Forest was selected because:

It performs well on structured tabular data

It handles multiple numerical features effectively

It is stable and reliable for classification tasks

It can identify patterns across timing and traffic-based features

The model outputs a probability score for each communication pair.
rf_prob = probability of suspicious beaconing behavior

## 📈 Detection Logic

The dashboard uses the model probability score to determine whether a communication pair should be flagged.

Example:
If rf_prob >= detection threshold → Flag as suspicious
If rf_prob < detection threshold → Treat as normal

Default threshold: 0.50

The threshold can be adjusted in the dashboard. A lower threshold increases sensitivity, while a higher threshold makes detection stricter.

## 🚦 Severity Levels

The dashboard assigns severity based on the Random Forest probability score.

| RF Probability | Severity |
| -------------- | -------- |
| 0.90 and above | Critical |
| 0.70 – 0.89    | High     |
| 0.50 – 0.69    | Medium   |
| Below 0.50     | Low      |

## 🔍 Reason Codes

The dashboard generates reason codes to explain why a communication pair may be suspicious.

Example reason codes include:

Very regular timing

Regular timing

Intervals repeat very consistently

Intervals are fairly consistent

Many repeated connections

Typical beacon interval range

Known DNS server traffic suppressed

Flagged by model-based multi-feature pattern

These reason codes make the model output easier to understand and more useful for SOC-style analysis.

## 🚨 Key Insight

C2 beaconing is difficult to detect by looking at only individual packets or single flows.

The key idea in this project is:

Repeated, periodic, and consistent outbound communication can indicate possible C2 beaconing activity.

Instead of analyzing each flow separately, this system groups traffic into communication pairs and studies behavior over time.

This makes the dashboard more useful for SOC-style investigation because it focuses on patterns, not just individual events.

## 🚀 Future Improvements

Add real-time traffic ingestion

Support Zeek/Bro logs directly

Add visual charts for severity distribution

Add beacon interval timeline charts

Improve false positive reduction

Add SHAP-based model explanations

Support multiple CTU-13 scenarios

Add automatic model retraining

Deploy the dashboard using Hugging Face Spaces or Streamlit Cloud

Add authentication for SOC-style deployment

## 👤 Author

Tharakeshwar Navarathinam
Penn State University
Cybersecurity Analytics and Operations

## 🧠 Final Takeaway

This project demonstrates how machine learning can be used to detect possible C2 beaconing behavior in network traffic.

The system focuses on repeated communication patterns, timing consistency, and pair-level behavior instead of analyzing flows in isolation.

By combining machine learning, severity scoring, reason codes, and a SOC-style dashboard, this project provides a practical approach for identifying suspicious command-and-control communication.






