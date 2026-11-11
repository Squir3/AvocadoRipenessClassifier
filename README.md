# Avocado Ripeness Classifier

A small computer-vision project for classifying avocado ripeness from an image. The primary interface is a Streamlit app powered by a fine-tuned ResNet18 model. The repository also contains scripts for training the classifier and experimenting with avocado detection using Ultralytics YOLO.

## Features

- Upload a JPG or PNG image and classify it into one of five ripeness stages.
- Run inference locally with the included `resnet_avocado.pt` checkpoint.
- Optionally prepare a classification dataset and fine-tune ResNet18.
- Optionally train and run an object-detection model with YOLO.

## Screenshots
<img width="1920" height="919" alt="chrome_iAMhvcOgjp" src="https://github.com/user-attachments/assets/301b7335-3540-4c73-b443-9faaea030484" />
<img width="1920" height="919" alt="chrome_vAQC6gtW6h" src="https://github.com/user-attachments/assets/25a48e98-9e91-4b2e-9381-7cc2ffddf078" />
<img width="1920" height="919" alt="chrome_rzCbNPECEk" src="https://github.com/user-attachments/assets/6c903c1a-6c31-4672-9769-99650df6f1f4" />
<img width="1920" height="919" alt="chrome_usGs2U8Vkg" src="https://github.com/user-attachments/assets/90cb4649-b31a-4d37-b378-54524e34a031" />
<img width="1920" height="919" alt="chrome_aXyLFkJpQF" src="https://github.com/user-attachments/assets/c01ca865-350d-45fa-bcb5-432007fe6f3d" />
<img width="1920" height="919" alt="chrome_t2A2gWJMSY" src="https://github.com/user-attachments/assets/3a09e732-2f2f-4e5a-834f-bb056739cc54" />

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
