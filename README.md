# Please download all files from github repository

https://github.com/Theshuvam/ECEN_Group6_Demo.git

# Fashion-MNIST Classification with Machine learning algoritms

The Fashion-MNIST dataset (https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST ) is used to train and test various image classification models and compare their performance. The algorithms that are implemented are K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machines (SVM), and Convolutional Neural Networks (CNN). You can find these algorithms developed for the Fashion-MNIST dataset and run them at (https://colab.research.google.com/drive/15nx68petPqHZp381uTuUUdvoUVuLR-x0?usp=sharing ). We also developed a Contrastive Language-Image Pretraining (CLIP) that was applied to the Fashion-MNIST dataset to perform zero-shot classification. You can find this algorithm developed for the Fashion-MNIST dataset and run it at ( https://colab.research.google.com/drive/1Gt28GaRqMyWwu576ps9g34Gzi-MqPfeb?usp=sharing ). The Convolutional Neural Networks (CNN) performed the best and therefore implemented further in this project.

# Fashion-MNIST Classification with CNN

This project implements a Convolutional Neural Network (CNN) to classify images in the Fashion-MNIST dataset, a collection of grayscale 28x28 images of clothing items. The project is divided into two main scripts: one for training and saving the model, and another for evaluating the model's performance.

---

## Project Structure

### Files:

1. **`cnn_model_train.py`**: Script for training the CNN and saving the trained model as `cnn_model.pth`.
2. **`cnn_main.py`**: Script for loading the saved model, evaluating it on the test dataset, and visualizing results.
3. **Fashion-MNIST Data**: Includes training, validation, and test datasets in CSV format.

### Outputs:

- **`cnn_model.pth`**: The saved weights of the trained CNN model.

---

## virtualenvironment

create a virtual environment to install the dependencies and libraries

```bash
virtualenv env_name
```

for MAC

1. Open your terminal.
2. Install virtualenv using pip:
   pip install virtualenv
3. Navigate to your project directory:
   cd /path/to/your/project
4. Create a virtual environment:
   virtualenv env_name
5. Run the following command to activate the virtual environment:
   source env_name/bin/activate

## Requirements

Install the libraries from the requirements.pdf

### Dependencies:

Install the required Python libraries using `pip`:

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
```

### Python Version:

- Python 3.7 or above

---

## Usage

### If you want to Train the Model (otherwise skip to step 1 below)

Run the `cnn_model_train.py` script to train the CNN using the training and validation datasets. The trained model will be saved as `cnn_model.pth`.

```bash
python  cnn_model_train.py
```

### Step 1: Evaluating the Model

Run the `cnn_main.py` script to load the saved model, test its performance on the test dataset, and display evaluation metrics.

```bash
python cnn_main.py
```

---

## Model Architecture

The CNN architecture consists of:

1. **Convolutional Layers**:
   - Layer 1: 32 filters with kernel size 3x3, followed by ReLU activation and MaxPooling.
   - Layer 2: 64 filters with kernel size 3x3, followed by ReLU activation and MaxPooling.
2. **Fully Connected Layers**:
   - Layer 1: 128 neurons with ReLU activation.
   - Output Layer: 10 neurons for class predictions.

---

## Evaluation Metrics

The model evaluates performance using:

- **Accuracy**: Percentage of correct predictions.
- **Precision, Recall, F1-Score**: Weighted averages for all classes.
- **Confusion Matrix**: Visual representation of prediction errors.

---

## Dataset

The Fashion-MNIST dataset contains 70,000 grayscale images of size 28x28 pixels, divided into 10 classes such as t-shirts, trousers, sneakers, etc.

| Class ID | Class Name  |
| -------- | ----------- |
| 0        | T-shirt/top |
| 1        | Trouser     |
| 2        | Pullover    |
| 3        | Dress       |
| 4        | Coat        |
| 5        | Sandal      |
| 6        | Shirt       |
| 7        | Sneaker     |
| 8        | Bag         |
| 9        | Ankle boot  |

---

## Acknowledgments

- The project uses the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) developed by Zalando Research.
- Built with [PyTorch](https://pytorch.org/).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
