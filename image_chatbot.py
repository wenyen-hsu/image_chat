import os
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import logging
import threading
from functools import wraps
import requests
import json
import logging
import pytesseract
from PIL import Image


# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("image_chatbot.log"),
                        logging.StreamHandler()
                    ])

def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from {image_path}: {e}")
        return ""

def process_images(image_folder):
    """Process all images in the folder and extract text."""
    image_texts = {}
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(image_folder, filename)
            text = extract_text_from_image(image_path)
            if text:
                image_texts[filename] = text
                logging.info(f"Extracted text from {filename}:\n{text[:100]}...")  # 只記錄前100個字符
            else:
                logging.info(f"No text extracted from {filename}")
    return image_texts

def custom_normalize(tensor):
    """Custom normalization function to scale tensor values to [0, 1]"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    custom_normalize,  # Add custom normalization step
])

def chat_with_llama(prompt, max_chunk_size=4000):
    """Send a prompt to Llama 3 using Ollama and get a response."""
    def process_chunk(chunk):
        try:
            response = requests.post('http://localhost:11434/api/generate', 
                                     json={
                                         "model": "llama3",
                                         "prompt": chunk
                                     },
                                     stream=True)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
                    if json_response.get('done', False):
                        break
            
            return full_response.strip()
        except Exception as e:
            logging.error(f"Error communicating with Llama 2 via Ollama: {e}", exc_info=True)
            return ""

    # 分割提示如果它太長
    chunks = [prompt[i:i+max_chunk_size] for i in range(0, len(prompt), max_chunk_size)]
    
    full_response = ""
    for i, chunk in enumerate(chunks):
        if i == 0:
            chunk_prompt = chunk
        else:
            # 對於後續的chunks，添加上下文
            chunk_prompt = f"Continuing from previous response. {chunk}"
        
        logging.info(f"Processing chunk {i+1} of {len(chunks)}")
        chunk_response = process_chunk(chunk_prompt)
        full_response += chunk_response + " "

    return full_response.strip()

def simple_question_answering(question, image_descriptions):
    """A simple function to answer questions based on image descriptions."""
    answer = f"Based on the image descriptions, I can tell you that:\n\n"
    for filename, description in image_descriptions.items():
        answer += f"- {description}\n"
    answer += f"\nRegarding your question: '{question}', I can only provide the above information about the images."
    return answer

def main():
    image_folder = "images"
    if not os.path.exists(image_folder):
        logging.error(f"Image folder '{image_folder}' not found.")
        return

    image_texts = process_images(image_folder)

    if not image_texts:
        logging.warning("No text was extracted from any images.")
        return

    while True:
        user_input = input("\nAsk a question about the images (or type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        prompt = f"Based on the following text extracted from images, please answer this question: {user_input}\n\nExtracted text:\n"
        for filename, text in image_texts.items():
            prompt += f"From {filename}:\n{text}\n\n"

        logging.info(f"Full prompt being sent to Ollama:\n{prompt}")

        try:
            response = chat_with_llama(prompt)
            print("Assistant:", response)
        except Exception as e:
            logging.error(f"Error communicating with Ollama: {e}", exc_info=True)
            print("Sorry, I couldn't process that request.")

if __name__ == "__main__":
    main()

# Setup and Usage Instructions:
"""
To use this image chatbot:

1. Install required packages:
   pip install torch torchvision Pillow transformers requests

2. Install Ollama and download the Llama 2 model:
   - Follow the instructions at https://github.com/jmorganca/ollama to install Ollama
   - Run 'ollama pull llama2' to download the Llama 2 model

3. Create an 'images' folder in the same directory as this script and add your images to it.

4. Ensure Ollama is running with the Llama 2 model:
   - Run 'ollama run llama2' in a separate terminal window

5. Run the script:
   python image_chatbot.py

6. The script will process all images in the 'images' folder and generate descriptions.

7. You can then ask questions about the images. The chatbot will use Llama 3 via Ollama for answers.
   If Ollama is not available, it will provide simple answers based on the image descriptions.

8. Type 'quit' to exit the chatbot.

Note: This script uses a pre-trained ViT model for image recognition, which may not be perfect for all types of images.
The quality of the answers depends on the accuracy of the image descriptions and the capabilities of the Llama 3 model.
"""