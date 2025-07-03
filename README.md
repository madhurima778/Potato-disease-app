# Potato Plant Disease Prediction

This project predicts potato plant diseases (Early Blight, Late Blight, Healthy) from leaf images using a combination of deep learning (ResNet50 feature extraction) and a Random Forest classifier, with data balancing via SMOTE. It includes a web app for easy image upload and prediction.

## Project Structure

- `app.py` — Flask web application for image upload and prediction.
- `Plant disease cnn rf CV and Smote.ipynb` — Jupyter notebook for model training and evaluation.
- `random_forest_model.pkl` — Trained Random Forest model.
- `resnet_feature_extractor.pth` — Trained ResNet50 feature extractor.
- `static/` — Static files (uploaded images, icons, etc.).
- `templates/index.html` — Web app HTML template.
- `potato/` — Dataset directory (with subfolders for each class).
- `Output/` — Output images (e.g., confusion matrix plots).
- `requirements.txt` — Python dependencies.

## Setup

1. **Clone the repository** and navigate to the project folder.

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   - Place your potato leaf images in the `potato/` directory, organized by class (e.g., `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`).

4. **Train the model (optional):**
   - Run the Jupyter notebook [`Plant disease cnn rf CV and Smote.ipynb`](Plant%20disease%20cnn%20rf%20CV%20and%20Smote.ipynb) to retrain the models and save new weights.

5. **Run the web app:**
   ```sh
   python app.py
   ```
   - Open your browser and go to `http://localhost:5000`.

## Usage

- Upload a potato leaf image using the web interface.
- The app will display the predicted disease class and confidence score, along with a description.

## Model Details

- **Feature Extraction:** ResNet50 (pretrained on ImageNet, last FC layer removed).
- **Classifier:** Random Forest (with cross-validation and SMOTE for class balancing).
- **Evaluation:** Confusion matrix and classification report are generated in the notebook.

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)
- [Flask](https://flask.palletsprojects.com/)

---
