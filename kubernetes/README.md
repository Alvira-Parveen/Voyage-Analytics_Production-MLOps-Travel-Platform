# Kubernetes Deployment Guide for Voyage Analytics

This directory contains production-grade Kubernetes (k8s) manifests to orchestrate, deploy, and scale the Voyage Analytics platform.

---

## 🏛️ Kubernetes Architecture

The stack is split into three decoupled services:
1. **`voyage-api` (FastAPI backend):** Replicated across 2 pods behind a ClusterIP service with readiness/liveness health probes.
2. **`voyage-dashboard` (Streamlit frontend):** Deployed as a single pod mapped to a LoadBalancer service, connecting to the API service dynamically.
3. **`voyage-mlflow` (Experiment registry):** Configured with a PersistentVolumeClaim (PVC) to persist the SQLite backend database across pod restarts.

---

## 🚀 How to Run the App on Kubernetes

### Step 1: Install & Start Minikube (Local Cluster)
If testing locally, install `kubectl` and `minikube`:
```bash
# Start the local cluster
minikube start
```

### Step 2: Apply the Deployment Manifests
Apply the configuration files to deploy the storage, containers, and services:
```bash
# Navigate to the kubernetes directory
cd kubernetes

# Deploy MLflow Storage and Service
kubectl apply -f mlflow-deployment.yaml

# Deploy the FastAPI Backend
kubectl apply -f api-deployment.yaml

# Deploy the Streamlit Dashboard
kubectl apply -f dashboard-deployment.yaml
```

### Step 3: Monitor the Cluster Status
Check that all pods are up and running:
```bash
# List all pods
kubectl get pods

# List services to get exposed ports
kubectl get services
```

### Step 4: Access the Dashboard Locally
If running on Minikube, tunnel the LoadBalancer service to expose the Streamlit Dashboard UI:
```bash
# Expose services
minikube service voyage-dashboard-service
```
This will automatically open the Streamlit App (`http://localhost:8501`) in your browser connected to the replicated FastAPI cluster backend!
