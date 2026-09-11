# PhD Research Project Requirements Document (PRD)

## A Lightweight and Green Blockchain–AI Framework for Secure and Energy-Efficient Supply Chain Management

---

## 1. Document Information

| Item | Description |
|---|---|
| Research Level | PhD |
| Research Field | Computer Science / Cybersecurity |
| Research Domain | Supply Chain Management |
| Core Technologies | Blockchain, Artificial Intelligence, Machine Learning |
| Research Themes | Cybersecurity, Lightweight Computing, Green Computing, Multi-Objective Optimization |
| Implementation Language | Python |
| Primary Dataset | DataCo Smart Supply Chain |
| Secondary Dataset | Olist Brazilian E-Commerce |
| Research Duration | 3–4 Years |
| Expected Publications | 3–4 Scopus-indexed Research Papers |

---

# 2. Research Title

### Primary Title

**A Lightweight and Green Blockchain–AI Framework for Secure and Energy-Efficient Supply Chain Management**

### Alternative Academic Title

**A Lightweight and Energy-Efficient Blockchain–AI Framework for Secure and Sustainable Supply Chain Management**

### Recommended Title

**A Lightweight and Green Blockchain–AI Framework for Secure and Energy-Efficient Supply Chain Management**

The recommended title is retained because it explicitly represents the five core dimensions of the research:

- **Blockchain–AI** — technological foundation
- **Secure** — cybersecurity focus
- **Lightweight** — resource efficiency
- **Green** — environmental sustainability
- **Energy-Efficient** — measurable energy objective
- **Supply Chain Management** — application domain

---

# 3. Research Vision

The proposed research aims to develop a **secure, intelligent, lightweight, and green Blockchain–AI framework** for Supply Chain Management.

The framework is designed to simultaneously address:

1. Data integrity
2. Transaction security
3. Anomaly detection
4. Traceability
5. Computational efficiency
6. Storage efficiency
7. Energy efficiency
8. Scalability

The central research principle is:

\[
Security + Intelligence + Trust + Traceability
\]

subject to:

\[
Low\ Energy + Low\ Computational\ Cost + Low\ Storage\ Overhead + Low\ Latency
\]

---

# 4. Research Context

Modern Supply Chain Management increasingly relies on digital platforms to exchange information among suppliers, manufacturers, logistics providers, warehouses, distributors, and customers.

This digital transformation generates large volumes of heterogeneous data and increases the dependency of Supply Chain Management on trustworthy information exchange.

However, distributed supply chain environments are exposed to several cybersecurity threats, including:

- Data Tampering
- Unauthorized Transactions
- Malicious Participants
- Abnormal Transaction Behavior
- Fraudulent Activities
- Data Integrity Violations
- Identity-related attacks
- Loss of Trust among Participants

Blockchain can provide tamper-evident records, transaction traceability, and decentralized trust.

Artificial Intelligence can provide intelligent analysis, anomaly detection, and risk prediction.

However, the integration of Blockchain and AI may introduce additional:

- Computational Overhead
- Storage Overhead
- Communication Overhead
- Transaction Latency
- Energy Consumption

Therefore, a research challenge emerges:

> How can Blockchain and AI be integrated into a secure Supply Chain framework while simultaneously minimizing computational, storage, latency, and energy costs?

---

# 5. Research Problem

Existing Blockchain-based Supply Chain solutions primarily focus on:

- Data Integrity
- Traceability
- Transparency
- Trust

while AI-based solutions primarily focus on:

- Prediction
- Classification
- Anomaly Detection
- Risk Analysis

However, the integrated Blockchain–AI architecture may introduce significant resource and energy overhead.

Furthermore, existing approaches do not necessarily provide a unified optimization framework that simultaneously considers:

\[
Security
\]

\[
AI\ Performance
\]

\[
Energy\ Consumption
\]

\[
Computational\ Cost
\]

\[
Storage\ Overhead
\]

\[
Transaction\ Latency
\]

Therefore, the research problem is defined as:

> **There is a need for a lightweight and green Blockchain–AI framework capable of improving Supply Chain security and trustworthiness while reducing energy consumption, computational cost, storage overhead, and transaction latency without significantly compromising security and Artificial Intelligence performance.**

---

# 6. Research Gap

The research gap is organized into five dimensions.

## 6.1 Blockchain Security Gap

Blockchain improves data integrity and traceability, but additional mechanisms are required to identify abnormal and potentially malicious transaction behavior.

## 6.2 Artificial Intelligence Security Gap

Artificial Intelligence can detect anomalies and suspicious behavior, but its integration into Blockchain-enabled Supply Chain environments may increase computational requirements.

## 6.3 Lightweight Computing Gap

Existing Blockchain–AI solutions may require substantial computational, memory, communication, and storage resources.

## 6.4 Green Computing Gap

Energy efficiency is often treated as an evaluation metric rather than a design objective.

## 6.5 Multi-Objective Optimization Gap

Security, AI performance, energy efficiency, latency, storage, and computational cost are often investigated independently rather than within a unified optimization framework.

---

# 7. Research Aim

The primary aim of this research is:

> **To design, develop, and evaluate a lightweight and green Blockchain–AI framework that enhances the security, integrity, trustworthiness, and intelligence of Supply Chain Management while minimizing energy consumption and computational, storage, communication, and latency overhead.**

---

# 8. Research Objectives

## RO1 — Secure Blockchain Architecture

Design a Blockchain architecture that protects Supply Chain transactions and improves data integrity and traceability.

## RO2 — Transaction Integrity

Develop a cryptographic transaction verification mechanism to detect unauthorized modification of Supply Chain data.

## RO3 — AI-Based Anomaly Detection

Develop an Artificial Intelligence mechanism for detecting abnormal and potentially malicious Supply Chain transactions.

## RO4 — Lightweight Artificial Intelligence

Reduce the computational complexity of Artificial Intelligence models through feature selection and model optimization.

## RO5 — Lightweight Blockchain

Reduce Blockchain-related:

- Computational Overhead
- Storage Overhead
- Communication Overhead
- Transaction Latency

## RO6 — Green Computing

Minimize:

- Energy Consumption
- Energy per Transaction
- Energy per Prediction
- Estimated Carbon Emissions

## RO7 — Multi-Objective Optimization

Develop a Multi-Objective Optimization model that jointly considers:

- Security
- AI Performance
- Energy Consumption
- Computational Cost
- Storage Overhead
- Transaction Latency

## RO8 — Scalability Evaluation

Evaluate system performance under increasing numbers of Supply Chain transactions.

## RO9 — Comparative Evaluation

Compare the proposed framework against conventional and state-of-the-art baseline configurations.

## RO10 — Reproducible Prototype

Develop a reproducible Python-based experimental prototype.

---

# 9. Research Questions

### RQ1

How can Blockchain improve data integrity, traceability, and trust in Supply Chain Management?

### RQ2

How can Artificial Intelligence detect abnormal and potentially malicious Supply Chain transactions?

### RQ3

How can the Blockchain–AI architecture be lightweighted without significantly compromising security and AI performance?

### RQ4

How can energy consumption and estimated carbon emissions be reduced in Blockchain-enabled Supply Chain Management?

### RQ5

What is the optimal trade-off among security, AI performance, energy consumption, computational cost, storage overhead, and transaction latency?

### RQ6

Can the proposed framework outperform conventional Blockchain–AI configurations in terms of security and resource efficiency?

### RQ7

How does the proposed framework perform under increasing transaction volumes?

---

# 10. Research Hypotheses

### H1

The proposed Blockchain architecture improves Supply Chain data integrity compared with a centralized data management approach.

### H2

The proposed AI-based anomaly detection mechanism improves the detection of abnormal and potentially malicious transactions.

### H3

Feature selection reduces Artificial Intelligence computational cost while maintaining acceptable detection performance.

### H4

The proposed Lightweight Blockchain architecture reduces computational, storage, and latency overhead.

### H5

The proposed Green Computing mechanism reduces energy consumption.

### H6

The proposed Multi-Objective Optimization approach achieves a better security–efficiency trade-off than single-objective optimization.

### H7

The proposed framework maintains acceptable security and AI performance as transaction volume increases.

---

# 11. Conceptual Architecture

```text
                 Supply Chain Data Sources
                         │
                         ▼
              Data Acquisition Layer
                         │
                         ▼
          Data Preprocessing and Validation
                         │
                         ▼
                Feature Engineering
                         │
                         ▼
                Feature Selection
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
    Lightweight AI Layer      Blockchain Security Layer
             │                       │
             ▼                       ▼
    Anomaly Detection          Transaction Validation
    Risk Prediction            Cryptographic Hashing
    Classification             Access Control
             │                       │
             └───────────┬───────────┘
                         ▼
                Smart Contract Layer
                         │
                         ▼
              Green Computing Layer
                         │
                         ▼
          Multi-Objective Optimization
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Security     Energy      Performance
             │           │           │
             ▼           ▼           ▼
          Detection      CO₂       Latency
          Accuracy                  Storage
             │           │           │
             └───────────┼───────────┘
                         ▼
          Optimal Secure Supply Chain
```

---

# 12. System Requirements

## SR1 — Data Acquisition

The system shall ingest structured Supply Chain data from publicly available datasets.

## SR2 — Data Preprocessing

The system shall perform:

- Missing Value Handling
- Duplicate Detection
- Data Cleaning
- Encoding
- Normalization where appropriate
- Feature Engineering

## SR3 — Feature Selection

The system shall identify relevant features while reducing dimensionality and computational requirements.

## SR4 — AI-Based Anomaly Detection

The system shall identify abnormal transaction patterns.

## SR5 — Blockchain Transaction Management

The system shall convert relevant Supply Chain records into Blockchain transactions.

## SR6 — Transaction Integrity

The system shall use cryptographic hashing to verify transaction integrity.

## SR7 — Access Control

The system shall enforce role-based or participant-based access policies where applicable.

## SR8 — Smart Contract Logic

The system shall simulate or implement transaction validation rules through Smart Contract logic.

## SR9 — Energy Measurement

The system shall measure or estimate energy consumption associated with major computational components.

## SR10 — Optimization

The system shall identify Pareto-optimal configurations under multiple objectives.

---

# 13. Blockchain Requirements

Each Blockchain transaction should contain, where applicable:

- Transaction ID
- Participant ID
- Product ID
- Order ID
- Timestamp
- Quantity
- Transaction Status
- Data Hash

Each Block should contain:

- Block ID
- Timestamp
- Transaction List
- Previous Block Hash
- Current Block Hash

The Blockchain layer shall support:

- Transaction Creation
- Transaction Validation
- Hash Verification
- Block Creation
- Block Verification
- Access Control
- Tamper Detection

---

# 14. On-Chain and Off-Chain Architecture

To support the Lightweight requirement, the proposed system shall use an **On-Chain/Off-Chain Architecture**.

## On-Chain Data

Only essential verification metadata should be stored on-chain:

- Transaction ID
- Data Hash
- Timestamp
- Status
- Verification Metadata

## Off-Chain Data

Large or detailed records should remain off-chain:

- Complete Order Information
- Product Information
- Detailed Shipment Information
- Analytical Data
- Large Dataset Records

The Blockchain layer will maintain cryptographic references to the off-chain data.

This design reduces:

- Blockchain Storage
- Transaction Size
- Validation Cost
- Storage Overhead

---

# 15. Artificial Intelligence Requirements

The AI layer shall support two primary functions.

## 15.1 Anomaly Detection

Detection of:

- Abnormal Transactions
- Suspicious Transaction Patterns
- Unusual Participant Behavior

## 15.2 Supply Chain Risk Prediction

Prediction of relevant Supply Chain risks, such as:

- Late Delivery Risk

Risk prediction may be treated as a secondary AI task to complement the primary cybersecurity objective.

---

# 16. Candidate AI Algorithms

The research shall evaluate suitable baseline and lightweight algorithms.

### Classification

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost
- LightGBM

### Anomaly Detection

- Isolation Forest
- One-Class SVM
- Autoencoder

The final algorithm shall be selected based on experimental evidence rather than predetermined assumptions.

---

# 17. Feature Selection Requirements

Feature Selection shall be used to reduce model complexity.

Candidate methods include:

- Particle Swarm Optimization (PSO)
- Grey Wolf Optimization (GWO)
- Hybrid PSO–GWO

The feature selection process shall evaluate:

\[
Accuracy
\]

\[
F1\ Score
\]

\[
Training\ Time
\]

\[
Inference\ Time
\]

\[
Memory\ Usage
\]

\[
Energy\ Consumption
\]

---

# 18. Lightweight Computing Requirements

The Lightweight design shall target two major components.

## 18.1 Lightweight AI

Reduction of:

- Number of Features
- Model Complexity
- Training Time
- Inference Time
- Memory Usage

## 18.2 Lightweight Blockchain

Reduction of:

- Transaction Size
- Block Size
- Storage Overhead
- Validation Cost
- Communication Overhead
- Transaction Latency

---

# 19. Green Computing Requirements

Green Computing shall be treated as a **design objective**, not merely a final evaluation metric.

The system shall measure or estimate:

- Energy Consumption
- Energy per Transaction
- Energy per Prediction
- Computational Resource Utilization
- Estimated Carbon Emissions

The basic energy model is:

\[
E = P \times T
\]

where:

- \(E\) = Energy Consumption
- \(P\) = Power Consumption
- \(T\) = Execution Time

Estimated carbon emissions may be represented as:

\[
CO_2 = E \times CI
\]

where:

- \(CO_2\) = Estimated Carbon Emissions
- \(CI\) = Carbon Intensity

The methodology used to estimate energy and carbon emissions shall be explicitly documented and experimentally validated where possible.

---

# 20. Cybersecurity Requirements

The security layer shall address selected threats relevant to Blockchain-enabled Supply Chain Management.

## Threats

- Data Tampering
- Unauthorized Transactions
- Malicious Participants
- Replay Attacks
- Abnormal Transaction Behavior
- Transaction Manipulation
- Smart Contract Misuse

## Security Mechanisms

The framework may incorporate:

- Cryptographic Hashing
- Digital Signatures
- Authentication
- Authorization
- Access Control
- Transaction Validation
- Anomaly Detection
- Tamper Detection

The final threat model shall be restricted to threats that can be experimentally simulated and objectively evaluated.

---

# 21. Multi-Objective Optimization

The optimization layer shall jointly consider security and resource efficiency.

## Objectives to Minimize

\[
Energy\ Consumption
\]

\[
Carbon\ Emissions
\]

\[
Computational\ Cost
\]

\[
Storage\ Overhead
\]

\[
Transaction\ Latency
\]

## Objectives to Maximize

\[
Security
\]

\[
AI\ Performance
\]

\[
Traceability
\]

The optimization problem can be represented as:

\[
\max \{Security, AI\ Performance, Traceability\}
\]

subject to:

\[
\min \{Energy, CO_2, Computation, Storage, Latency\}
\]

---

# 22. Candidate Optimization Algorithms

The following algorithms shall be experimentally evaluated:

### Baseline

- Grid Search

### Metaheuristic Optimization

- Particle Swarm Optimization (PSO)
- Grey Wolf Optimization (GWO)

### Multi-Objective Optimization

- NSGA-II
- Multi-Objective Particle Swarm Optimization (MOPSO)

The final optimization method shall be selected based on:

- Convergence
- Computational Cost
- Solution Quality
- Pareto Front Quality
- Stability
- Reproducibility

---

# 23. Pareto Optimization

The framework shall generate a Pareto Front representing trade-offs between conflicting objectives.

For example:

\[
Security \leftrightarrow Energy
\]

\[
Accuracy \leftrightarrow Computation
\]

\[
Latency \leftrightarrow Security
\]

\[
Storage \leftrightarrow Traceability
\]

This allows the research to identify multiple optimal configurations rather than a single solution based on an arbitrary weighting scheme.

---

# 24. Dataset Requirements

## Primary Dataset

**DataCo Smart Supply Chain Dataset**

The dataset shall be used for:

- Supply Chain Analysis
- Transaction Generation
- Risk Prediction
- Feature Selection
- AI Evaluation
- Blockchain Simulation

## Secondary Dataset

**Olist Brazilian E-Commerce Dataset**

The secondary dataset shall be used for:

- External Validation
- Generalization Testing
- Robustness Evaluation

## Synthetic Transaction Dataset

Synthetic transactions shall be generated from realistic Supply Chain patterns to evaluate:

- Scalability
- Transaction Processing
- Energy Consumption
- Latency
- Storage Growth

Target transaction volumes may include:

- 10,000
- 50,000
- 100,000
- 500,000
- 1,000,000

---

# 25. Data Processing Pipeline

```text
Raw Supply Chain Dataset
          │
          ▼
Data Cleaning
          │
          ▼
Missing Value Handling
          │
          ▼
Duplicate Detection
          │
          ▼
Feature Engineering
          │
          ▼
Feature Selection
          │
          ▼
Train / Validation / Test
          │
          ▼
AI Model Training
          │
          ▼
Blockchain Transaction Generation
          │
          ▼
Security Evaluation
          │
          ▼
Energy and Performance Evaluation
```

Data Leakage shall be explicitly prevented during training and evaluation.

---

# 26. Experimental Design

The research shall use controlled comparative experiments.

## Experiment 1

Centralized Supply Chain Data Management.

## Experiment 2

Traditional Blockchain without AI.

## Experiment 3

Blockchain + Standard AI.

## Experiment 4

Blockchain + Lightweight AI.

## Experiment 5

Lightweight Blockchain + Lightweight AI.

## Experiment 6

Security-Enhanced Blockchain–AI Framework.

## Experiment 7

Green-Optimized Framework.

## Experiment 8

Final Multi-Objective Optimized Framework.

---

# 27. Evaluation Metrics

## 27.1 AI Performance Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- False Positive Rate
- False Negative Rate

## 27.2 Cybersecurity Metrics

- Attack Detection Rate
- Anomaly Detection Rate
- Tamper Detection Rate
- Unauthorized Transaction Detection Rate
- Integrity Verification Rate

## 27.3 Blockchain Metrics

- Transaction Latency
- Throughput
- Block Size
- Storage Overhead
- Validation Time

## 27.4 Computational Metrics

- CPU Utilization
- Memory Usage
- Training Time
- Inference Time

## 27.5 Green Metrics

- Energy Consumption
- Energy per Transaction
- Energy per Prediction
- Estimated Carbon Emissions

---

# 28. Statistical Evaluation

The experimental results shall be supported by appropriate statistical analysis.

Possible methods include:

- Mean
- Standard Deviation
- Confidence Intervals
- Wilcoxon Signed-Rank Test
- Friedman Test
- Post-hoc Analysis
- Effect Size

Statistical testing shall be selected according to the experimental design and data distribution.

---

# 29. Baseline Configurations

The proposed framework shall be compared against:

### Baseline A

Centralized Data Management.

### Baseline B

Traditional Blockchain.

### Baseline C

Blockchain + Standard AI.

### Baseline D

Blockchain + Lightweight AI.

### Baseline E

Blockchain + AI + Optimization.

### Proposed Configuration

**Lightweight + Green + Secure Blockchain–AI + Multi-Objective Optimization**

---

# 30. Novelty

The novelty of this research shall not be claimed as the simple integration of Blockchain and Artificial Intelligence.

Instead, the primary novelty is:

> **A unified security-aware and energy-aware optimization framework that jointly addresses Blockchain security, AI-based anomaly detection, computational efficiency, storage efficiency, transaction latency, and energy consumption in Supply Chain Management.**

The expected novel contributions are:

### N1 — Lightweight Security Architecture

A Blockchain architecture designed to reduce resource overhead while maintaining security properties.

### N2 — AI-Based Transaction Security

An AI-based anomaly detection layer integrated with Blockchain transaction processing.

### N3 — Green Blockchain–AI Design

Energy consumption and estimated carbon emissions incorporated as explicit design objectives.

### N4 — Multi-Objective Security Optimization

Joint optimization of security and system efficiency.

### N5 — Security–Energy–Performance Trade-off

A systematic analysis of the relationship between:

\[
Security
\leftrightarrow
AI\ Performance
\leftrightarrow
Energy
\leftrightarrow
Latency
\leftrightarrow
Storage
\]

---

# 31. Expected Contributions

The research is expected to contribute:

1. A Lightweight Blockchain architecture for Supply Chain Management.
2. An AI-based anomaly detection mechanism for Supply Chain transaction security.
3. An On-Chain/Off-Chain storage strategy for reducing Blockchain storage overhead.
4. A Green Computing methodology for evaluating energy consumption.
5. A Multi-Objective Optimization model for security and efficiency.
6. A Security–Energy–Performance trade-off model.
7. A reproducible Python-based experimental framework.
8. A comprehensive evaluation methodology for Blockchain–AI Supply Chain security.

---

# 32. Python Implementation Requirements

The entire experimental framework shall be implemented primarily using Python.

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- XGBoost
- LightGBM

## Deep Learning

- PyTorch, if required

## Optimization

- PSO
- GWO
- NSGA-II
- MOPSO

## Cryptography

- hashlib
- cryptography

## Database

- SQLite
- PostgreSQL

## API

- FastAPI

## Visualization

- Matplotlib
- Plotly

## Experimental Environment

- Jupyter Notebook
- VS Code

## Version Control

- Git
- GitHub

---

# 33. Prototype Architecture

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── ai/
│   ├── blockchain/
│   ├── security/
│   ├── energy/
│   ├── optimization/
│   └── evaluation/
│
├── notebooks/
│
├── experiments/
│
├── models/
│
├── results/
│
├── tests/
│
├── docs/
│
├── requirements.txt
│
└── README.md
```

---

# 34. Research Methodology

The research shall follow the following methodological phases:

### Phase 1 — Literature Review

Study the intersection of:

- Blockchain
- Artificial Intelligence
- Supply Chain Security
- Lightweight Computing
- Green Computing
- Multi-Objective Optimization

### Phase 2 — Dataset Preparation

Prepare and analyze the selected Supply Chain datasets.

### Phase 3 — Baseline AI Development

Develop and evaluate baseline AI models.

### Phase 4 — Blockchain Prototype

Develop the Blockchain transaction and validation layer.

### Phase 5 — Cybersecurity Layer

Implement selected security mechanisms and threat scenarios.

### Phase 6 — Lightweight AI

Apply Feature Selection and model optimization.

### Phase 7 — Green Computing

Measure and estimate energy and environmental metrics.

### Phase 8 — Multi-Objective Optimization

Develop and evaluate the optimization layer.

### Phase 9 — Integrated Framework

Integrate Blockchain, AI, Cybersecurity, Lightweight Computing, Green Computing, and Optimization.

### Phase 10 — Evaluation

Conduct comparative, scalability, ablation, and statistical evaluations.

---

# 35. Ablation Study

Ablation studies shall be conducted to determine the contribution of each major component.

The following configurations shall be evaluated:

| Configuration | Blockchain | AI | Security | Lightweight | Green | Optimization |
|---|---:|---:|---:|---:|---:|---:|
| C1 | ✓ | — | — | — | — | — |
| C2 | ✓ | ✓ | — | — | — | — |
| C3 | ✓ | ✓ | ✓ | — | — | — |
| C4 | ✓ | ✓ | ✓ | ✓ | — | — |
| C5 | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| C6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

This analysis will determine whether each component provides measurable research value.

---

# 36. Scalability Evaluation

The framework shall be evaluated under increasing transaction volumes.

Example:

\[
10K \rightarrow 50K \rightarrow 100K \rightarrow 500K \rightarrow 1M
\]

For each volume, the following shall be measured:

- Throughput
- Latency
- CPU Utilization
- Memory Usage
- Storage
- Energy Consumption
- Energy per Transaction
- Detection Performance

---

# 37. Reproducibility Requirements

All experiments shall be reproducible.

The research shall document:

- Dataset versions
- Python version
- Library versions
- Hardware specifications
- Random Seeds
- Model Parameters
- Optimization Parameters
- Experimental Configurations
- Execution Time
- Energy Measurement Method
- Evaluation Scripts

Where ethically and legally permissible, source code and experiment configurations should be made available.

---

# 38. Research Risks and Mitigation

## Risk 1 — Limited Attack Data

**Risk:** Public Supply Chain datasets may not contain labeled cybersecurity attacks.

**Mitigation:** Develop controlled and reproducible attack scenarios based on the defined Threat Model.

---

## Risk 2 — Energy Measurement Accuracy

**Risk:** Software-level energy estimation may not perfectly represent physical energy consumption.

**Mitigation:** Clearly distinguish between measured and estimated energy and document the measurement methodology.

---

## Risk 3 — Insufficient Novelty

**Risk:** Blockchain–AI integration is already an established research direction.

**Mitigation:** Focus novelty on the unified optimization of security, energy, computational cost, latency, storage, and AI performance.

---

## Risk 4 — Excessive Research Scope

**Risk:** Combining Blockchain, AI, Cybersecurity, Green Computing, and Optimization can make the project too broad.

**Mitigation:** Define a focused Threat Model, a limited set of AI algorithms, selected optimization algorithms, and measurable research objectives.

---

# 39. Publication Plan

## Research Paper 1

### **Blockchain and Artificial Intelligence for Secure and Green Supply Chain Management: A Systematic Literature Review**

Focus:

- Blockchain
- AI
- Cybersecurity
- Green Computing
- Lightweight Computing

---

## Research Paper 2

### **A Lightweight Blockchain Framework for Secure and Trustworthy Supply Chain Management**

Focus:

- Blockchain Architecture
- Data Integrity
- Transaction Validation
- On-Chain/Off-Chain Architecture
- Storage and Latency Efficiency

---

## Research Paper 3

### **Lightweight AI-Based Anomaly Detection for Secure Blockchain-Enabled Supply Chain Transactions**

Focus:

- Feature Selection
- Lightweight AI
- Anomaly Detection
- Cybersecurity
- Computational Efficiency

---

## Research Paper 4

### **A Multi-Objective Optimized Lightweight and Green Blockchain–AI Framework for Secure Supply Chain Management**

Focus:

- Blockchain
- AI
- Cybersecurity
- Lightweight Computing
- Green Computing
- Energy Efficiency
- Multi-Objective Optimization

This paper represents the principal integrated contribution of the PhD research.

---

# 40. Research Timeline

## Year 1

### Months 1–3

- Literature Review
- Research Problem Definition
- Research Questions
- Threat Model Definition

### Months 4–6

- Dataset Analysis
- Data Preprocessing
- Baseline AI Models

### Months 7–12

- Systematic Literature Review
- Initial Blockchain Architecture
- Research Paper 1

---

# Year 2

### Months 13–18

- Blockchain Prototype
- Transaction Validation
- On-Chain/Off-Chain Architecture

### Months 19–24

- Security Layer
- Feature Selection
- Lightweight AI
- Research Paper 2

---

# Year 3

### Months 25–30

- AI-Based Anomaly Detection
- Energy Measurement
- Green Computing

### Months 31–36

- Multi-Objective Optimization
- Security–Energy–Performance Analysis
- Research Paper 3

---

# Year 4

### Months 37–42

- Integrated Framework
- Scalability Experiments
- Ablation Studies
- Statistical Evaluation

### Months 43–48

- Research Paper 4
- Thesis Writing
- Final Validation
- Dissertation Defense Preparation

---

# 41. Acceptance Criteria

The proposed framework shall be considered successful if experimental evidence demonstrates measurable improvements in one or more of the following areas without unacceptable degradation in the remaining objectives:

## Security

Improved anomaly, tampering, and unauthorized transaction detection.

## AI Performance

Acceptable or improved Precision, Recall, F1-Score, and ROC-AUC.

## Lightweight Performance

Reduced computational, memory, storage, and latency overhead.

## Green Performance

Reduced energy consumption and energy per transaction or prediction.

## Optimization

Improved Pareto-optimal solutions compared with baseline optimization approaches.

## Scalability

Acceptable system performance under increasing transaction volumes.

---

# 42. Final Research Framework

The proposed research can be formally represented as:

\[
Framework =
Blockchain +
AI +
Cybersecurity +
Lightweight\ Computing +
Green\ Computing +
MultiObjective\ Optimization
\]

with the overall optimization objective:

\[
\max
\left(
Security,
AI\ Performance,
Traceability
\right)
\]

subject to:

\[
\min
\left(
Energy,
CO_2,
Computational\ Cost,
Storage\ Overhead,
Latency
\right)
\]

under scalability constraints.

---

# 43. Core Research Proposition

The central proposition of this PhD research is:

> **A Supply Chain Management system can achieve stronger security and trust through Blockchain and Artificial Intelligence while maintaining practical resource requirements by jointly optimizing security, AI performance, computational cost, storage overhead, latency, and energy consumption.**

The proposed framework therefore treats the problem as a **multi-dimensional security and efficiency optimization problem**, rather than as a simple Blockchain implementation or an isolated Artificial Intelligence application.

---

# 44. Final Research Identity

The research is positioned at the intersection of:

\[
\boxed{
Cybersecurity
+
Blockchain
+
Artificial Intelligence
+
Lightweight Computing
+
Green Computing
+
MultiObjective Optimization
}
\]

with:

\[
\boxed{
Supply\ Chain\ Management
}
\]

as the application environment.

The primary scientific contribution is therefore:

> **Security-aware, lightweight, and green Blockchain–AI optimization for Supply Chain Management.**