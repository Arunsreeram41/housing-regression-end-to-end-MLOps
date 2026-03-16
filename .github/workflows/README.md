# Housing ML End-to-End Project

## Project Overview
Housing Regression MLE is an end-to-end machine learning pipeline for predicting housing prices using XGBoost. This project demonstrates a production-grade MLOps setup featuring modular pipelines, experiment tracking via MLflow, serverless containerization with **AWS Fargate**, and automated CI/CD via GitHub Actions.

---

## 🏗 Architecture
The system follows a microservices architecture where the frontend and backend are decoupled but communicate seamlessly via a private network.

### Cloud Infrastructure & Deployment
- **AWS S3 Integration**: Centralized storage for data and models in the `housing-regression-mlops` bucket.
- **Amazon ECR**: Private container registry for versioned Docker images.
- **Amazon ECS (Fargate)**: Serverless container orchestration, providing isolated compute environments.
- **ECS Service Connect**: Implements a service mesh within the `mlops-network` namespace, allowing Streamlit to resolve the API via `http://housing-api:8000` without the cost of a Load Balancer.
- **IAM Task Roles**: The API container utilizes the `housing-mlops-task-role` to securely download models from S3 at runtime.
- **CI/CD Pipeline**: Automated deployment via GitHub Actions (`ci.yml`) using `--force-new-deployment` to trigger immediate image updates.

### ECS Services & Resource Allocation
| Service Name | Task Role | Port | CPU | Memory |
| :--- | :--- | :--- | :--- | :--- |
| **housing-api-task-service-hk820xh3** | `housing-mlops-task-role` | 8000 | 256 (.25 vCPU) | 512 MB |
| **housing-streamlit-service** | None (Client-only) | 8501 | 256 (.25 vCPU) | 512 MB |

---

## 🚀 Web Applications

### FastAPI Backend
- **Function**: Serves predictions via a REST API.
- **Internal DNS**: `http://housing-api:8000` (Used by Streamlit).
- **Public Testing**: Accessible via `http://[API_PUBLIC_IP]:8000`.

### Streamlit Dashboard
- **Function**: Interactive UI for housing price visualization and prediction.
- **Public Access**: Accessible via `http://[STREAMLIT_PUBLIC_IP]:8501`. This url will be printed in the logs of ci.yml also.
- **Integration**: Communicates with the backend using the `API_URL` environment variable.

---

## 🛠 Common Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync
Docker (Local & Production)
Bash
# Build API container
docker build -t housing-api .

# Build Streamlit container  
docker build -t housing-streamlit -f Dockerfile.streamlit .

# Run API container locally
docker run -p 8000:8000 housing-api
Deployment
Deployments are fully automated on every push to the main branch:

GitHub Actions builds the Docker images.

Images are pushed to Amazon ECR with the :latest tag.

ECS services are updated to trigger a fresh pull of the new images.

📈 Key Design Patterns
Service-to-Service Networking
This project utilizes AWS ECS Service Connect for internal communication. This provides a DNS-based discovery mechanism that keeps traffic within the AWS VPC, increasing security and reducing operational costs by avoiding an Application Load Balancer.

Role-Based Access Control (RBAC)
We strictly separate the Execution Role (used by ECS to pull images) and the Task Role (used by the FastAPI code). This ensures that only the API container has the identity required to read from the S3 model bucket.

Data Leakage Prevention
The project implements strict data engineering practices:

Time-based splits: Prevents looking into the future during training.

Encoder Persistence: Frequency and target encoders are fitted only on training data and saved as pickles for inference consistency.

📂 File Structure Notes
src/api/: FastAPI implementation with S3 model loading.

app.py: Streamlit application logic.

.github/workflows/ci.yml: GitHub Actions deployment pipeline.