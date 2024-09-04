# UI Screenshot Classification by Fine-tuning CLIP

This project implements a UI screenshot classification system by fine-tuning the OpenAI CLIP (Contrastive Language-Image Pre-Training) model. It classifies UI screenshots into 28 different app categories such as Sports, Travel, Dating, etc.
![Screenshot 2024-09-04 at 1 55 25 PM](https://github.com/user-attachments/assets/4f9a5bd7-290a-4dc9-8fa4-b3a6ab34c6bb)
![Screenshot 2024-09-04 at 1 55 25 PM](![Screenshot 2024-09-04 at 2 02 38 PM](https://github.com/user-attachments/assets/9c4a755a-0203-4d8d-80b0-99b66bcd2f4b)
![Screenshot 2024-09-04 at 1 53 11 PM](https://github.com/user-attachments/assets/2eeba6bb-8ccf-4f1d-a27d-773d0d343548)
![Screenshot 2024-09-04 at 1 46 23 PM](https://github.com/user-attachments/assets/44a24732-285f-4e65-925e-b4de40c7b32e)


## Project Structure

- `.gitattributes`: Git attributes file for managing large files.
- `categories.pkl`: Pickle file containing the list of unique classes.
- `clip_finetuned.pth`: The fine-tuned CLIP model weights (large file, managed with Git LFS).
- `finetuning-clip-notebook.py`: Script for fine-tuning the CLIP model.
- `README.md`: This file, containing project documentation.
- `streamlit_classification.py`: Streamlit app for deploying the classification model.

## Features


- Fine-tuned CLIP model for UI screenshot classification
- 28 different app categories for classification
- Streamlit web application for easy image upload and classification
- Visualization of top 5 predictions with probabilities

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/UI-Screenshot-Classification-by-fine-tuning-CLIP.git
   cd UI-Screenshot-Classification-by-fine-tuning-CLIP
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: The requirements.txt file is not shown in the project structure. Make sure to create one with all the necessary dependencies.

## Usage

### Fine-tuning the Model

To fine-tune the CLIP model on your own dataset:

1. Prepare your dataset in the format expected by the `finetuning-clip-notebook.py` script.
2. Run the fine-tuning script:
   ```
   python finetuning-clip-notebook.py
   ```

This script will fine-tune the model and save the weights to `clip_finetuned.pth`.

### Running the Streamlit App

To run the classification app locally:

1. Ensure you have the fine-tuned model (`clip_finetuned.pth`) in the project directory.
2. Run the Streamlit app:
   ```
   streamlit run streamlit_classification.py
   ```
3. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`).
4. Upload a UI screenshot image to get the classification results.

## Online Demo

You can try out the UI screenshot classification tool online at: [Placeholder for Streamlit Share link]

## Model Performance

The current model achieves:
- Accuracy: 65%
- F1 Score: 62%

Note: These metrics represent the initial performance. Further optimization and experimentation with hyperparameters could potentially improve these results.

## Limitations

- The current model's performance (65% accuracy, 62% F1 score) leaves room for improvement.
- Due to time and computational constraints, extensive hyperparameter tuning has not been performed.
- The model has been trained on a subset of 50,000 samples from the RICO-SCA dataset. Training on the full dataset might yield better results.

## Future Work

- Experiment with different CLIP model variants (e.g., ViT-L/14).
- Perform more extensive hyperparameter tuning.
- Train on the full RICO-SCA dataset or incorporate additional UI screenshot datasets.
- Implement data augmentation techniques to improve model generalization.
- Explore ensemble methods or other advanced techniques to boost performance.


## Acknowledgements

- This project uses the OpenAI CLIP model.
- The RICO-SCA dataset was used for training and evaluation.
- Streamlit was used for creating the web application interface.
