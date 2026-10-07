# FedMed — Cross-Silo Federated Learning Engine

## Overview

FedMed is a privacy-preserving federated learning system for healthcare applications. The project demonstrates how multiple hospitals can collaboratively train a medical imaging model without sharing their raw patient data with a central server.

The project uses a simulated multi-hospital environment in which each hospital performs local model training on its own private MRI dataset. Only model updates are communicated to the central server for federated aggregation.

## Problem Statement

Medical AI models require large amounts of patient data for effective training. However, healthcare institutions cannot freely share raw patient data because of privacy and regulatory requirements.

FedMed addresses this problem using **Cross-Silo Federated Learning**, allowing multiple simulated hospitals to collaboratively train a brain-tumor segmentation model while keeping their local MRI data private.

## Project Objective

The main objectives of FedMed are:

- Establish a centralized 3D U-Net baseline.
- Set up three simulated hospital nodes.
- Partition MRI data across the hospital nodes.
- Train the model locally at each hospital.
- Aggregate local model updates using Federated Averaging (FedAvg).
- Implement secure communication between the server and hospital nodes.
- Integrate Homomorphic Encryption using TenSEAL.
- Add Differential Privacy to model updates.
- Develop a dashboard for monitoring federated training.
- Visualize MRI tumor segmentation results.

## System Architecture

```text
                    Central Server
                         |
                  Flower Aggregator
                       FedAvg
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
     Hospital 1      Hospital 2      Hospital 3
     Local Data      Local Data      Local Data
          |              |              |
          v              v              v
     Local Training  Local Training  Local Training
          |              |              |
          +--------------+--------------+
                         |
                  Model Updates
                         |
                  Secure Aggregation
                         |
                         v
                   Global Model
                         |
                         v
                Monitoring Dashboard
                Technology Stack
Machine Learning
- Python
- PyTorch
- MONAI
- 3D U-Net
Federated Learning
- Flower
- Federated Averaging (FedAvg)
Privacy & Security
- TenSEAL
- Homomorphic Encryption
- Differential Privacy
- gRPC
- TLS
Dashboard
- React
- Recharts
- WebSocket
Development Plan
Week 1 — Centralized Baseline & Node Scaffolding
- Prepare the public MRI dataset.
- Implement MRI data loading and preprocessing.
- Normalize the MRI data.
- Create training and validation splits.
- Establish a centralized 3D U-Net baseline.
- Set up the Flower framework.
- Configure three simulated hospital nodes.
Week 2 — Federated Training
- Partition the MRI dataset across three hospital nodes.
- Implement local training at each hospital.
- Implement communication between clients and the central server.
- Broadcast the global model to the hospital nodes.
- Collect local model updates.
- Aggregate updates using FedAvg.
- Implement secure communication using gRPC/TLS.
Week 3 — Privacy & Live Metrics
- Integrate TenSEAL Homomorphic Encryption.
- Encrypt model updates on the client side.
- Perform aggregation on encrypted updates.
- Stream training loss and accuracy metrics.
- Integrate live metrics with the monitoring interface.
Week 4 — Differential Privacy & Dashboard
- Add Differential Privacy to model updates.
- Evaluate the effect of privacy mechanisms on model performance.
- Build the React monitoring dashboard.
- Visualize training loss and accuracy.
- Display final MRI tumor segmentation masks.
- Perform final integration and testing.

FedMed/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── partitions/
│       ├── hospital_1/
│       ├── hospital_2/
│       └── hospital_3/
│
├── preprocessing/
│
├── models/
│
├── server/
│
├── clients/
│   ├── hospital_1/
│   ├── hospital_2/
│   └── hospital_3/
│
├── privacy/
│
├── dashboard/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore