# Resistor Value Prediction using CNN (PyTorch)

This project predicts resistor values from images of resistors using a Convolutional Neural Network (CNN) implemented in **PyTorch**.

## 📂 Project Structure

resistor-value-prediction/
├── notebooks/

│   └── Resistor\_Value\_Prediction\_CNN\_Model.ipynb  # Cleaned Jupyter Notebook

├── src/

  ├── data\_loader.py   # Loads dataset, applies transforms

  ├── model.py         # CNN model definition

  ├── train.py         # Training script

  ├── predict.py       # Prediction script

├── requirements.txt      # Dependencies

├── README.md             # Project documentation


## ✅ Features
- Load resistor image dataset and apply preprocessing
- Train a CNN to classify resistor values
- Save and load trained models
- Predict resistor value from a new image

## 🛠️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/resistor-value-prediction.git
cd resistor-value-prediction
pip install -r requirements.txt
````

## ▶️ Usage

### **Training**

```bash
python src/train.py --data_dir data --epochs 10 --batch_size 32
```

(Default values are inside the script.)

### **Prediction**

```bash
python src/predict.py --image sample_resistor.jpg
```

## 📊 Model Architecture

* Conv2D → ReLU → MaxPool
* Conv2D → ReLU → MaxPool
* Fully Connected Layer
* Output Layer

## 🔮 Future Improvements

* Add more layers for better accuracy
* Implement early stopping and learning rate scheduling
* Deploy as a web app

---

**Author:** Murede Adetiba 
**License:** MIT

```

