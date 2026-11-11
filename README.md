# Avocado Ripeness Classifier

A small computer-vision project for classifying avocado ripeness from an image. The primary interface is a Streamlit app powered by a fine-tuned ResNet18 model. The repository also contains scripts for training the classifier and experimenting with avocado detection using Ultralytics YOLO.

## Features

- Upload a JPG or PNG image and classify it into one of five ripeness stages.
- Run inference locally with the included `resnet_avocado.pt` checkpoint.
- Optionally prepare a classification dataset and fine-tune ResNet18.
- Optionally train and run an object-detection model with YOLO.

## Screenshots

Add application screenshots to a `screenshots/` directory, then embed them here. For example:

<!-- Uncomment after adding the screenshot file:
![Avocado ripeness classifier](screenshots/classifier.png)
-->

## Ripeness labels

| Class | Stage |
| --- | --- |
| 1 | Unripe |
| 2 | Breaking / transitional stage |
| 3 | Ripe, early stage |
| 4 | Ripe, peak stage |
| 5 | Overripe |

## Requirements

- Python 3.10 or later
- PyTorch and Torchvision
- Streamlit and Pillow
- For training: Matplotlib
- For YOLO experiments: Ultralytics

The app loads the model on the CPU. Training uses CUDA automatically for the ResNet classifier when available. Install a PyTorch build compatible with your operating system and CUDA version by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Run the application

Clone the repository and open a terminal in its root directory. Create and activate a virtual environment, then install the application dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the libraries and start Streamlit:

```bash
python -m pip install --upgrade pip
python -m pip install streamlit pillow torch torchvision
streamlit run app.py
```

The app opens in a browser. Upload a JPG, JPEG, or PNG image to see the predicted ripeness class. The checkpoint `resnet_avocado.pt` must be in the repository root, because the app loads it using a relative path.

## Training the classification model

The classification training scripts use a directory named `dataset_crop` in ImageFolder format, with one folder per class:

```text
dataset_crop/
  1/
  2/
  3/
  4/
  5/
```

If the source images are available under `Avocado Ripening Dataset/` and their filenames end in the class label (for example, `Tam_d10_164_b_5.jpg`), create this structure by running:

```bash
python sort_by_filename_label.py
```

Then run the augmented training script:

```bash
python -m pip install matplotlib torch torchvision pillow
python train_classifier_v2.py
```

`train_classifier_v2.py` fine-tunes a pretrained ResNet18 for 25 epochs, uses an 80/20 random train/validation split, and saves the resulting state dictionary to `resnet_avocado.pt`. It also displays a training/validation loss plot. The simpler `train_classifier.py` script runs a 10-epoch variant without augmentation; it displays the loss plot but does not save a checkpoint.

## YOLO detection experiments

The YOLO configuration in `data.yaml` expects this dataset structure in the repository root:

```text
images/
  train/
  test/
labels/
  train/
  test/
```

The labels must use YOLO annotation format. Check `data.yaml` and update its paths if your dataset is stored elsewhere. The training script `avo.py` uses `yolov8s.pt`, trains for 30 epochs at image size 640, and explicitly selects CUDA device 0:

```bash
python -m pip install ultralytics
python setup_checker.py
python avo.py
```

Training is therefore configured for a CUDA-capable machine. If CUDA is unavailable, update the `device` argument in `avo.py` before running it. The prediction script `predict_avocado.py` expects trained weights at `runs/detect/yolo_avocado_fixed/weights/best.pt` and test images in `images/test`:

```bash
python predict_avocado.py
```

Predictions are saved by Ultralytics. `test.py` is an additional experiment that expects a different checkpoint at `runs/detect/yolo_avocado42/weights/best.pt`; that checkpoint is not included in this repository.

### Dataset preparation note

`podziel_zdjecia.py` copies images and YOLO label files into train/test folders, but its `base_dir` currently points to a machine-specific Windows path. Update that path and ensure the expected `labels_all` source folder exists before running the script. It uses a fixed random seed and assigns 80% of images to training and 20% to testing.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit image-classification application |
| `resnet_avocado.pt` | ResNet18 checkpoint used by the app |
| `sort_by_filename_label.py` | Organizes classification images into class folders |
| `train_classifier.py` | Basic ResNet18 training experiment |
| `train_classifier_v2.py` | Augmented ResNet18 training and checkpoint export |
| `data.yaml` | YOLO dataset configuration |
| `avo.py` | YOLO training script |
| `predict_avocado.py`, `test.py` | YOLO inference experiments |
| `podziel_zdjecia.py` | Creates train/test folders for YOLO data |
| `setup_checker.py`, `ch.py` | CUDA availability checks |

## Notes before publishing

- Confirm that you have permission to redistribute the images in `Avocado Ripening Dataset/` and any model weights you publish. Check the original dataset and model licenses and include the required attribution.
- Do not publish private keys, credentials, or other secrets. The repository currently contains `awokado-key.pem`; verify what it is and remove it from the public repository if it is a private key. If a real private key has already been exposed, revoke or rotate it.
- The model's prediction is an estimate based on visual appearance and should not be treated as a definitive quality or food-safety assessment.

## License

No project license is currently specified. Add a `LICENSE` file before describing the project as open source or granting others permission to reuse its code. Dataset and pretrained-weight licenses may be separate from the project license.