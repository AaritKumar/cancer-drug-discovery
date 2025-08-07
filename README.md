# AI-Powered Cancer Drug Discovery for EGFR Inhibitors

## 🏆 Project Overview

This project implements a complete computational pipeline for discovering novel drug candidates targeting the Epidermal Growth Factor Receptor (EGFR), a key protein implicated in various cancers. The pipeline leverages a sophisticated **Graph Neural Network (GNN)** to predict the bioactivity of molecules, enabling a massive virtual screen of over 100,000 compounds. The top candidates are then subjected to a simulated molecular docking analysis for validation.

The core workflow is as follows:
1.  **Data Acquisition**: Download and process extensive bioactivity data for EGFR from the ChEMBL database.
2.  **Model Training**: Train a GNN model on this data to learn the relationship between a molecule's structure and its ability to inhibit EGFR. (Note: A pre-trained model, `egfr_model.pth`, is provided).
3.  **Virtual Screening**: Use the trained GNN to predict the EGFR inhibition probability for a large library of drug-like compounds.
4.  **Candidate Prioritization**: Rank the compounds based on their predicted probability and other drug-like properties.
5.  **In-Silico Validation**: Perform a simulated molecular docking analysis on the top candidates to validate their potential binding affinity to the EGFR protein, comparing them against known FDA-approved drugs.

---

## 📂 Project Structure

```
.
├── drug_discovery_dataset/
│   └── chembl_drug_discovery_dataset.csv  # Virtual screening library (~100k compounds)
├── docking_results/
│   ├── docking_validation_plots.png       # Output plots from docking simulation
│   ├── docking_validation_report.txt      # Output text summary from docking
│   └── docking_validation_results.csv     # Output CSV from docking
├── chembl_processor.py             # Script to process a raw ChEMBL dataset into a screening library
├── dataset_download.py             # Script to download and prepare the EGFR training dataset from ChEMBL
├── drug_discovery.py               # Main script to run the virtual screening pipeline
├── egfr_large_dataset_8179_compounds.csv # The training dataset for the GNN model
├── egfr_model.pth                  # The pre-trained GNN model file
├── gnn_model.py                    # Defines the GNN architecture and molecular feature processor
├── molecular_docking.py            # Script for simulated molecular docking validation
├── top_100_discoveries.txt         # Text file listing the top 100 predicted candidates
└── README.md                       # This file
```

---

## 🔬 Core Concepts & Methodology

### 1. Why Graph Neural Networks for Drug Discovery?

Molecules can be naturally represented as graphs, where **atoms are nodes** and **chemical bonds are edges**. This representation allows GNNs to learn directly from the 2D structure of a molecule, capturing complex topological and chemical information that traditional machine learning models, which rely on fixed-size "fingerprints", might miss.

<p align="center">
  <img src="https://i.imgur.com/lFMTG0p.png" width="500">
  <br>
  <em>Figure 1: A molecule represented as a graph for a GNN.</em>
</p>

### 2. From SMILES to Graphs

The process starts with a SMILES string (e.g., `CCO` for ethanol), a 2D representation of a molecule. The `EnhancedMolecularProcessor` in `gnn_model.py` converts this into a graph object suitable for PyTorch Geometric.

-   **Node Features (Atom Features)**: For each atom, we extract a vector of its chemical properties. This project uses 16 features, including:
    -   One-hot encoded atomic symbol (e.g., C, N, O, F)
    -   Number of heavy atom neighbors
    -   Number of attached hydrogens
    -   Formal charge
    -   Whether the atom is in an aromatic ring
    -   One-hot encoded hybridization type (sp, sp², sp³)

-   **Edge Features (Bond Features)**: For each bond, we extract a vector of its properties. This project uses 4 features:
    -   One-hot encoded bond type (Single, Double, Triple, Aromatic)
    -   Whether the bond is conjugated
    -   Whether the bond is in a ring

### 3. The `AdvancedEGFRNet` GNN Architecture

The heart of this project is the `AdvancedEGFRNet` model defined in `gnn_model.py`. It uses Graph Attention (GAT) layers, which allow the model to learn the importance of neighboring atoms when updating the representation of a central atom.

The architecture consists of:
1.  **Input Processing**: Two linear layers process the initial atom and bond features.
2.  **GATv2Conv Layers**: Three Graph Attention v2 layers form the core of the network. These layers update each atom's feature vector based on a weighted average of its neighbors' features. The "attention" mechanism learns the weights, effectively focusing on the most important parts of the molecule for the prediction task.
3.  **Global Pooling**: After the GAT layers, a `global_add_pool` operation sums up all the atom feature vectors to create a single feature vector for the entire molecule (the graph-level representation).
4.  **Prediction Head**: A series of fully connected layers with dropout and ReLU activation functions process the graph-level vector to produce the final output: a probability that the molecule inhibits EGFR.

#### The Math Behind Graph Attention (GAT)

The core idea of GAT is to compute the representation of a node by attending over its neighbors. The attention coefficient \( \alpha_{ij} \) between a central node \(i\) and its neighbor \(j\) is calculated as follows:

1.  **Linear Transformation**: First, the feature vectors (\(\mathbf{h}\)) of the nodes are transformed by a shared weight matrix \(\mathbf{W}\).
    \[ \mathbf{z}_i = \mathbf{W}\mathbf{h}_i \]
2.  **Attention Mechanism**: An attention score \(e_{ij}\) is calculated using a learnable weight vector \(\mathbf{a}\). In GATv2, the mechanism is more expressive than the original GAT.
    \[ e_{ij} = \mathbf{a}^T \text{LeakyReLU}(\mathbf{W}[\mathbf{h}_i \| \mathbf{h}_j]) \]
3.  **Softmax Normalization**: The scores are normalized across all neighbors of node \(i\) using the softmax function to get the final attention weights \(\alpha_{ij}\).
    \[ \alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})} \]
4.  **Feature Aggregation**: The output feature vector for node \(i\) is a weighted sum of its neighbors' transformed features, where the weights are the attention coefficients.
    \[ \mathbf{h}'_i = \sigma\left(\sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij} \mathbf{z}_j\right) \]

This process is repeated for multiple "heads" with different learned weights, and the results are aggregated, allowing the model to capture different types of relationships simultaneously.

### 4. (Simulated) Molecular Docking

After the GNN identifies promising candidates, `molecular_docking.py` provides a final validation step. Since running real molecular docking software is computationally expensive and complex, this script **simulates the process**. It calculates a "binding affinity" score based on an empirical model that combines key molecular descriptors known to be important for protein-ligand binding.

The simulated binding affinity is calculated as:
\[ \text{Affinity} = \text{Base} + w_{mw}S_{mw} + w_{logp}S_{logp} + w_{hb}S_{hb} + w_{arom}S_{arom} + P_{flex} + B_{qed} \]
Where:
-   \(S_{mw}\): Score based on Molecular Weight (optimal ~450 Da).
-   \(S_{logp}\): Score based on LogP (lipophilicity, optimal ~3.0).
-   \(S_{hb}\): Score from Hydrogen Bond Donors/Acceptors.
-   \(S_{arom}\): Score from Aromatic Rings (very important for EGFR's ATP pocket).
-   \(P_{flex}\): A penalty for having too many rotatable bonds (high flexibility is entropically unfavorable for binding).
-   \(B_{qed}\): A bonus based on the Quantitative Estimate of Drug-likeness (QED) score.

The script compares the scores of our novel candidates against known FDA-approved EGFR inhibitors (Erlotinib, Gefitinib) to provide a valuable benchmark.

---

## 🚀 How to Run the Pipeline

### 1. Installation

First, clone the repository and install the required Python packages. It is highly recommended to use a Conda environment.

```bash
# Clone the repository
git clone <repository_url>
cd <repository_directory>

# Create a Conda environment
conda create -n drug-discovery python=3.9
conda activate drug-discovery

# Install core packages
pip install pandas numpy matplotlib seaborn requests

# Install cheminformatics packages
pip install rdkit
pip install torch torchvision torchaudio
pip install torch_geometric
```
**Note:** Installing PyTorch and PyTorch Geometric can be tricky. Please refer to their official websites for the correct installation commands for your specific OS and CUDA version.

### 2. Run the Virtual Screening

The main script, `drug_discovery.py`, runs the entire screening process. It will:
1.  Load the pre-trained GNN model (`egfr_model.pth`).
2.  Load the virtual screening dataset (`drug_discovery_dataset/chembl_drug_discovery_dataset.csv`).
3.  Process each of the ~100,000 compounds, converting them to graphs and predicting their EGFR inhibition probability.
4.  Filter, score, and rank the results.
5.  Save the top 100 discoveries to `top_100_discoveries.txt`.
6.  Save the full, ranked list of promising candidates to `egfr_prediction_results.csv`.

To run the screening, execute:
```bash
python drug_discovery.py
```
This process is computationally intensive and may take a significant amount of time depending on your hardware.

### 3. Run the Docking Validation

After the screening is complete, you can run the simulated docking validation on a few of the top candidates (hardcoded in the script for demonstration).

```bash
python molecular_docking.py
```
This will create a `docking_results` directory containing a CSV file with the results, a plots image, and a detailed text report comparing the novel candidates to FDA-approved drugs.

---

## 📄 Explanation of Key Files

-   `gnn_model.py`: This is the scientific core of the project. It contains the `EnhancedMolecularProcessor` class, which handles the complex task of converting a SMILES string into a graph with rich atom and bond features. It also defines the `AdvancedEGFRNet` PyTorch model, specifying the GNN architecture with its Graph Attention layers.

-   `drug_discovery.py`: This is the main orchestrator. It ties everything together by loading the model and data, iterating through the screening library, calling the model for predictions, calculating a final drug-likeness score, and saving all the results in a user-friendly format.

-   `dataset_download.py`: A utility script for project reproducibility. It shows how the original training data was acquired by targeting EGFR in the ChEMBL database, fetching bioactivity data, filtering for high-quality `IC50` measurements, and cleaning the data for training.

-   `chembl_processor.py`: Another data utility. This script was used to process a very large, raw ChEMBL dataset into the ~100,000 compound screening library used in this project. It filters molecules based on drug-like properties (Lipinski's Rule of Five) to ensure the screening library is of high quality.

-   `molecular_docking.py`: The final validation step. This script takes top candidates and simulates how they would bind to the EGFR protein. It's a self-contained analysis that prepares 3D structures for the molecules, calculates a simulated binding affinity score, and generates a rich report comparing the new candidates to known drugs, complete with plots and statistics.
