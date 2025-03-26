import streamlit as st
from groq import Groq
import base64
from PIL import Image as PILImage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Groq API key from the environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq API client with API key
client = Groq(api_key=groq_api_key)
llava_model = 'llava-v1.5-7b-4096-preview'
llama31_model = 'llama-3.1-70b-versatile'

# Function to resize image if too large
def resize_image(image, max_size=(800, 800)):
    """Resizes the image if it's larger than max_size."""
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size)
    return image

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to generate image description using LLaVA
def image_to_text(client, model, base64_image, prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=model
    )

    return chat_completion.choices[0].message.content

# Function to generate short story using Llama 3.1
def short_story_generation(client, image_description):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a children's book author. Write a short story about the scene depicted in this image.",
            },
            {
                "role": "user",
                "content": image_description,
            }
        ],
        model=llama31_model
    )

    return chat_completion.choices[0].message.content

# Streamlit app title
st.title("LLaVA & Llama 3.1: Image Description and Story Generator")

# Image upload section
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Open the uploaded image using PIL
    image = PILImage.open(uploaded_image)

    # Resize the image if it's too large
    resized_image = resize_image(image)

    # Create temp_images directory if it doesn't exist
    if not os.path.exists("temp_images"):
        os.makedirs("temp_images")

    # Save resized image to the temp_images directory
    image_path = os.path.join("temp_images", uploaded_image.name)
    resized_image.save(image_path)

    # Display the uploaded image
    st.image(resized_image, caption="Uploaded Image (Resized)", use_column_width=True)

    # Encode the image to base64
    base64_image = encode_image(image_path)

    # Generate image description
    st.write("Generating image description using LLaVA...")
    description_prompt = "Describe this image in detail, including the appearance of the dog(s) and any notable actions or behaviors."
    image_description = image_to_text(client, llava_model, base64_image, description_prompt)

    st.write("### Image Description")
    st.write(image_description)

    # Generate short story based on image description
    st.write("Generating short story using Llama 3.1...")
    short_story = short_story_generation(client, image_description)

    st.write("### Generated Story")
    st.write(short_story)
