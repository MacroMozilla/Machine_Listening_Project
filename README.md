### step 0
pip install -r req.txt

---
### Step 1
set your path in config.json

---
### Step 2
run a_prepare_data/pre_path.py to initialize the info cache

---
### Step 3
run a_prepare_data/b0_prep_dataset.py to initialize the attributes cache


---
### Dataset

The datasets are prepared using the following scripts:

* `b0_prep_dataset.py`: Base dataset preparation script.
* `b1_prep_dataset_MelSpec.py`: Dataset preparation using Mel-spectrogram features.
* `b2_prep_dataset_wav2vec.py`: Dataset preparation using Wav2Vec embeddings.

Each script handles data preprocessing tailored to the corresponding feature extraction method.



---

### Model

The model components are defined across the following scripts:

* `a_nn_metrics.py`: Contains evaluation metrics used to assess model performance.
* `b_nn_loss.py`: Defines custom loss functions used during training.
* `c_nn_model.py`: Implements the neural network architecture.

---

### Training

* `a_train_with_config_withatt.py`: Training script using attribute-based configurations.
* `b_train_with_config_nonatt.py`: Training script using non-attribute-based configurations.

---


### Visualization

* `plot_reduction.py`: Generates t-SNE plots using class embeddings and wave embeddings, without attribute information.

---
