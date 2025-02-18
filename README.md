# Next Word Prediction With LSTM and Early Stopping

This project demonstrates the application of a Long Short-Term Memory (LSTM) neural network to predict the next word in a sentence. It utilizes TensorFlow to build and train the model and Streamlit to create an interactive web application that users can use to test the model's predictions.

## Live Application

Experience the live application of our Next Word Prediction model hosted on Streamlit. Try it out now: [Next Word Prediction App](https://next-word-prediction-with-lstm.streamlit.app/).

## Project Overview

The LSTM model is trained on a dataset (specify dataset here, e.g., "text from various sources" or a specific corpus like "Wikipedia articles") to predict the next word in a sequence of words. The model incorporates early stopping during training to prevent overfitting and to optimize the training duration.

## Features

- **LSTM Model**: Utilizes a recurrent neural network architecture for sequence prediction.
- **Early Stopping**: Implements early stopping to enhance training efficiency and model performance.
- **Streamlit Web Application**: Provides a user-friendly interface for real-time interaction with the trained model.

## Installation

To set up the project environment and run the application, follow these steps:

1. **Clone the repository**

```bash
   git clone https://github.com/yourusername/next-word-prediction.git
   cd next-word-prediction
```
2. **Create a Python virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. **Install the requirements**
```bash
pip install -r requirements.txt
```
4. **Run the Streamlit application**
```bash
streamlit run app.py
```
 ## Usage
After launching the Streamlit application, enter a sequence of words into the input field and press the "Predict Next Word" button. The application will display the predicted next word based on the input sequence

## Contributing
Contributions to this project are welcome! Please consider the following ways you can contribute:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Acknowledgments
- Thanks to TensorFlow and Streamlit for providing the tools that made this project possible.
- (Optional) List any other acknowledgments or credits, such as datasets used or inspiration for the project.

## Contact Information
For any additional questions or comments, you can reach out to me at:

- Email: tanvipatel3004@gmail.com
- LinkedIn: www.linkedin.com/in/tanvi-p-b86b37317
