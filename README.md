# Hybrid Quantum-Classical Machine Learning Pipeline

This repository demonstrates a full end-to-end Quantum Machine Learning (QML) pipeline running on physical IBM Quantum hardware via the Qiskit Runtime API.

## Architecture
- **Data Preprocessing:** Classical data (Iris dataset subset) normalization via `scikit-learn` to map features to tight quantum phase angles.
- **Quantum Feature Map:** `ZZFeatureMap` translates classical data into quantum superposition states.
- **Quantum Ansatz:** `RealAmplitudes` acts as the trainable parameter layer (the quantum "weights").
- **Hybrid Optimization:** `COBYLA` classical optimizer iteratively reads physical quantum measurements to minimize the loss function.
- **Hardware Integration:** Asynchronous execution on IBM's physical superconducting quantum processors using `QiskitRuntimeService` and modern `SamplerV2` primitives.

## Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Authenticate with IBM Quantum (Save your token locally once):
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN_HERE", set_as_default=True)
   ```
4. Run the pipeline:
   ```bash
   python qml_vqc_pipeline.py
   ```
