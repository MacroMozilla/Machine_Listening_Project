# First-Shot Unsupervised Anomalous Sound Detection
## Machine Listening Project 

Team members:
| Name | NetID | 
|----------|----------|
| Hongyi Zeng | hz3866 |
| Chancellor Rickey | cr3843 |
| Swapnil Sharma | ss19753 |

### step 0
pip install -r req.txt

---
### Step 1
set your path where you have your data downloaded and unzipped in config.json

---
### Step 2
run a_prepare_data/pre_path.py to initialize the info cache

---
### Step 3
run a_prepare_data/b0_prep_dataset.py to initialize the attributes cache


---
### Dataset

The datasets are prepared using the following scripts:

* `b0_prep_dataset.py`: Base dataset preparation script for raw waveform feature generation and also the log-mel feature generation
* ~~`b1_prep_dataset_MelSpec.py`: Dataset preparation using Mel-spectrogram features~~ NOT utilized in this branch
* ~~`b2_prep_dataset_wav2vec.py`: Dataset preparation using Wav2Vec embeddings~~ NOT utilized in this branch


---

### Model

The model components are defined across the following scripts:

* `a_nn_metrics.py`: Contains evaluation metrics used to assess model performance
* `b_nn_loss.py`: Defines custom loss functions used during training
* `c_nn_model.py`: Implements the neural network architecture

---

### Training

* `d_train_raw.py`: Training script
* ~~`d_train_melspec.py`~~ NOT utilized in this branch
* ~~`d_train_wav2vec.py`~~ NOT utilized in this branch

---

