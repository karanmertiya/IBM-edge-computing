"""
qml_vqc_pipeline.py
End-to-end Hybrid Quantum Machine Learning (QML) Training & Inference Pipeline.
Executes on IBM Quantum Cloud Hardware via Qiskit Runtime Primitives.
"""
import os
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import time

def run_quantum_ai_pipeline():
    print("1. Loading and Preprocessing Classical Data...")
    # We use a standard dataset (Iris) but filter it for binary classification
    data = load_iris()
    X = data.data[:100]  # First 100 rows
    y = data.target[:100]
    
    # QML REQUIRES scaling. Quantum phase angles must be tightly bounded.
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("2. Authenticating with IBM Quantum Cloud...")
    # Note: Ensure you have saved your account token before running this.
    # You only need to save it once using QiskitRuntimeService.save_account(...)
    try:
        service = QiskitRuntimeService()
    except Exception as e:
        print(f"Failed to authenticate: {e}")
        print("Please ensure your IBM API token is saved.")
        return
    
    # We ask IBM's cloud to find us the least busy, free, operational quantum hardware
    print("   Searching for available Quantum Hardware...")
    backend = service.least_busy(simulator=False, operational=True)
    print(f"   -> Success! Connected to physical backend: {backend.name}")
    
    # Wrap the backend in a modern Sampler primitive for the ML algorithm
    sampler = Sampler(backend=backend)

    print("3. Building the Quantum Neural Network Architecture...")
    num_features = X.shape[1]
    
    # The Feature Map encodes classical data into quantum superposition
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    
    # The Ansatz is our "trainable" quantum layer (similar to hidden layers in PyTorch)
    ansatz = RealAmplitudes(num_qubits=num_features, reps=2)

    print("4. Compiling the Hybrid VQC Optimizer...")
    # COBYLA is a classical optimizer that will update our quantum weights based on cloud measurements
    optimizer = COBYLA(maxiter=30)
    
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )

    print("5. Initiating Cloud Training (This sends jobs to the quantum queue)...")
    start_time = time.time()
    
    # FIT: This triggers the hybrid loop. Classical computer calculates -> Quantum computer measures -> Repeat.
    print("   Submitting training jobs to quantum hardware. This may take some time depending on the cloud queue...")
    vqc.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"   -> Quantum Training Completed in {elapsed:.2f} seconds.")

    print("6. Running Inference on Test Data...")
    quantum_accuracy = vqc.score(X_test, y_test)
    
    print("\n=========================================")
    print("         QML PIPELINE RESULTS            ")
    print("=========================================")
    print(f"Hardware Used     : IBM {backend.name}")
    print(f"Classical Opt     : COBYLA")
    print(f"Quantum Accuracy  : {quantum_accuracy * 100:.2f}%")
    print("=========================================")

if __name__ == "__main__":
    run_quantum_ai_pipeline()
