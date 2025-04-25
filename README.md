# Transfer-Learning-Based-Modular-Defense-Against-Natural-Adversarial-Examples

Our work introduces a novel defense approach against NAEs in the ImageNet dataset by leveraging a transfer learning-based defense module inserted before the final classification layer. Our method differentiates between salient features (SF), which are robust and closely aligned with human perception, and trivial features (TF), which can mislead models and undermine their performance. We evaluate multiple architectures for the defense module, mainly GANs, an autoencoder, and a one-hidden-layer network. We effectively extract and prioritize SFs, thereby enhancing the model’s ability to classify and defend against NAEs accurately. Extensive experiments on ImageNet-A demonstrate that our approach significantly improves the robustness of DNNs. Our defense is specifically useful in settings where high computational power for training or inference is not available, making it suitable for edge and low-power devices.


## Repository Structure
```
./
├── config.py                 # Configuration and data loading utilities
├── evaluate_gan.py          # Evaluation scripts for GAN models
├── evaluate_onelayer.py     # Evaluation scripts for one-layer classifier
├── finetune_autoencoder.py  # Fine-tuning scripts for autoencoder
├── GAN_imagenet.py          # GAN implementation for feature generation
├── generate_and_save_features*.py  # Feature extraction scripts for ResNet/Inception
├── helpers.py               # Utility functions and helper methods
├── onelayer.py             # One-layer classifier implementation
├── prepare_dataset*.py      # Dataset preparation and processing scripts
├── train_gan.py            # GAN training implementation
├── train_onelayer.py       # One-layer classifier training
└── train_unsupervised_autoencoder.py  # Autoencoder training implementation
```


## Defense Module
We have three different architectures for the defense module. GAN, Autoencoder, and a one layer module. 
![Architecture Diagram](https://raw.githubusercontent.com/ohamdash/Transfer-Learning-Based-Modular-Defense-Against-Natural-Adversarial-Examples/main/assets/architecture.png)

### Installation
1. **Clone the repository:**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare datasets:**
```bash
# Create directory structure
mkdir -p original-datasets/imagenet-a
mkdir -p original-datasets/imagenet-a-plus
mkdir -p original-datasets/imagenet

# Place datasets in appropriate directories
# ImageNet -> original-datasets/imagenet/
# ImageNet-A -> original-datasets/imagenet-a/
# ImageNet-A-Plus -> original-datasets/imagenet-a-plus/

# Prepare dataset
python3 prepare_dataset_plus.py
```
4. **Prepare the validation dataset:** Install the official imagenet validation dataset as a `.tar` file, then run the `prepare.sh` script to prepare the dataset. Finally, move the desired classes to the desired datasets folder by running `copy_dirs.py`.

### Quick Start
**Generate and save features:**
```bash
python3 generate_and_save_features.py
```
This will save the generated features to `./data/features`. Make sure to set the dataset directories to fit your setup. This has to be done once, then those features can be used to train and test all the other models. Note that this only generates features and labels for two dataset folders at once (benign training and adversarial training dataset for example). To generate features for all the datasets and train/test splits, please run multiple times while modifying the path of the datasets.
#### GAN
1. **Train the GAN model:**
```bash
python3 train_gan.py --gan_epochs 25001 --name_suffix resnet
```
This will save the trained GAN model to the `./models` directory. The training epochs and the name suffix of the saved GAN model can be controlled through the flags shown in the example above. The base GAN names are `SF_GAN` and `TF_GAN`.

2. **Finetune and evaluate the GAN:**
```bash
python3 evaluate_gan.py --model_name <model-name> --adv_dataset <dataset> --cnn <cnn-type> [--finetune_dataset <dataset>] [--ben_epochs <epochs>] [--adv_epochs <epochs>]
```
| Argument | Description | Choices/Format | Required | Default |
| --- | --- | --- | --- | --- |
| `--model_name` | Name of the GAN model to evaluate (path relative to `./models/`) | String | Yes | None |
| `--adv_dataset` | Name of the adversarial dataset to test on | `in-a`, `in-a-plus` | Yes | None |
| `--finetune_dataset` | Name of the fine-tuning dataset (if fine-tuning is enabled) | `in-a`, `in-a-plus`, or empty | No | Empty (no fine-tuning) |
| `--cnn` | Name of the CNN architecture to use | `inc-v3`, `resnet` | Yes | `inc-v3` |
| `--ben_epochs` | Number of benign fine-tuning epochs | Integer | No | 10 |
| `--adv_epochs` | Number of adversarial fine-tuning epochs | Integer | No | 30 |

The accuracies will be printed after the evaluation is over, make sure to set the correct paths for your datasets and GAN pretrained model. 
#### Autoencoder
1. **Train the unsupervised autoencoder:**
```bash
python3 train_unsupervised_autoencoder.py
```
This will save the trained unsupervised autoencoder model to the `./models` directory.

2. **Finetune and evaluate the autoencoder:**
```bash
python3 finetune_autoencoder.py
```

The accuracies will be printed after the evaluation is over, make sure to set the correct paths for your datasets and autoencoder pretrained model.
#### One Layer
1. **Train the one layer defense module**
```bash
python3 train_onelayer.py
```
This will save the trained one layer defense module to the `./models` directory. 

2. **Evaluate the one layer defense module**
```bash
python3 evaluate_onelayer.py
```
The evaluation accuracies will be printed after the evaluation is over. Make sure to load the correct model and features for evaluation.

## Notes
- All features are loaded and prepared in the `config.py` file, which is then imported to all the other scripts.
- The labels generated by the script `generate_and_save_features.py` are 200 one-hot encoded vectors. While using with the pretrained classification heads of either Inception v3 or Resnet, they need to be mapped to 1000 one-hot encoded vectors. We do this by using the `map_200_t0_100()` function in `helpers.py.`
- Features generated from a specific CNN model can only be used with this CNN model for training/evaluation, as the features are not compatible across the different CNNs.
