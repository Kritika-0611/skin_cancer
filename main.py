import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Tensorflow Model Prediction
def model_prediction(image_data):
    # Load the trained model
    cnn = tf.keras.models.load_model('Train_Cancer_Disease.keras')
    
    # Open the uploaded image and resize
    image = Image.open(image_data)
    image = image.resize((128, 128))  # Resize to match the model's input size
    
    # Convert the image to array and normalize pixel values
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    
    # Get prediction
    prediction = cnn.predict(input_arr)
    return np.argmax(prediction)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("SKIN CANCER DETECTION SYSTEM")
    image_path = "shaily.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Skin Cancer Detection System! 🧑‍⚕️🔬

Our mission is to assist in the early detection of skin cancer by analyzing images of skin abnormalities. Upload an image of a mole or lesion, and our system will help identify potential signs of skin cancer. Together, we aim for healthier skin and early intervention!

### How It Works
1. **Upload Image:** Go to the **Skin Cancer Detection** page and upload an image of a mole or skin lesion.
2. **Analysis:** Our system processes the image using advanced machine learning algorithms to detect possible signs of skin cancer.
3. **Results:** View detailed results along with medical recommendations for the next steps.

### Why Choose Us?
- **Accuracy:** Our model leverages cutting-edge deep learning techniques to provide reliable and precise skin cancer detection.
- **User-Friendly:** A simple and intuitive platform designed for ease of use.
- **Fast and Efficient:** Get results quickly to make timely decisions regarding skin health.

### Get Started
Click on the **Skin Cancer Detection** page in the sidebar, upload an image, and let our system guide you through the detection process!

### About Us
Learn more about the project, our expert team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset has been meticulously curated and includes images of skin lesions for the purpose of skin cancer detection. The dataset is divided into various sets to support effective training and evaluation. 

The dataset consists of a total of 2,262 images categorized into different classes. It has been divided into training, validation, and test sets to ensure comprehensive model evaluation.

#### Content
1. **Train:** 1,792 images
2. **Validation:** 447 images
3. **Test:** 23 images

Each image in the dataset is labeled and organized to facilitate accurate training and testing of the skin cancer detection model. The dataset is designed to help in the efficient identification of skin cancer and to improve the overall accuracy of the detection system.
            
""")

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    
    # File uploader for image input
    test_image = st.file_uploader("Choose an Image:", type=["png", "jpg", "jpeg"])
    
    if test_image is not None:
        # Open the image with PIL
        image = Image.open(test_image)
        
        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # If predict button is clicked
        if st.button("Predict"):
            st.snow()  # Streamlit effect
            
            try:
                # Predict the result index
                result_index = model_prediction(test_image)

                # Load the validation set to get class names
                validation_set = tf.keras.utils.image_dataset_from_directory(
                    'Train',
                    labels="inferred",
                    label_mode="categorical",
                    class_names=None,
                    color_mode="rgb",
                    batch_size=32,
                    image_size=(128, 128),
                    shuffle=True,
                    seed=123,
                    validation_split=0.2,
                    subset="validation",
                    interpolation="bilinear",
                    follow_links=False,
                    crop_to_aspect_ratio=False
                )

                # Get class names from validation set
                class_name = validation_set.class_names
                print(class_name)
                
                # Show prediction result
                st.success(f"Model predicts it's a {class_name[result_index]}")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
    
    else:
        st.warning("Please upload an image for prediction.")
