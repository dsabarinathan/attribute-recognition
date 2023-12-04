import argparse
import numpy as np
from PIL import Image
import cv2
import torch
from tqdm import tqdm

# Assuming model_ft is defined elsewhere in your code
# model_ft = ...

# Define the label_col
label_col = np.array(['Age-Young', 'Age-Adult', 'Age-Old', 'Gender-Female',
   'Hair-Length-Short', 'Hair-Length-Long', 'Hair-Length-Bald',
   'UpperBody-Length-Short', 'UpperBody-Color-Black',
   'UpperBody-Color-Blue', 'UpperBody-Color-Brown',
   'UpperBody-Color-Green', 'UpperBody-Color-Grey',
   'UpperBody-Color-Orange', 'UpperBody-Color-Pink',
   'UpperBody-Color-Purple', 'UpperBody-Color-Red',
   'UpperBody-Color-White', 'UpperBody-Color-Yellow',
   'UpperBody-Color-Other', 'LowerBody-Length-Short',
   'LowerBody-Color-Black', 'LowerBody-Color-Blue',
   'LowerBody-Color-Brown', 'LowerBody-Color-Green',
   'LowerBody-Color-Grey', 'LowerBody-Color-Orange',
   'LowerBody-Color-Pink', 'LowerBody-Color-Purple', 'LowerBody-Color-Red',
   'LowerBody-Color-White', 'LowerBody-Color-Yellow',
   'LowerBody-Color-Other', 'LowerBody-Type-Trousers&Shorts',
   'LowerBody-Type-Skirt&Dress', 'Accessory-Backpack', 'Accessory-Bag',
   'Accessory-Glasses-Normal', 'Accessory-Glasses-Sun', 'Accessory-Hat'])

def preprocess_image(image_path, resize=(224, 224)):
    # Open image with OpenCV
    image = cv2.imread(image_path)
    # Make sure image is in RGB format (3 channels)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if resize is not None:
        image = cv2.resize(image, resize)

    # Normalize using mean and std
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalized_img = (image / 255.0 - mean) / std

    # Convert NumPy array to PyTorch tensor
    img_tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).float()

    return img_tensor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def perform_inference(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predicted_results = []

    normalized_image = preprocess_image(image_path)
    normalized_image_tensor = normalized_image.to(device)
    normalized_image_tensor = normalized_image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(normalized_image_tensor)

#    print(output)
    predicted_probs = output.cpu().numpy().astype(float)
    predicted_probs = sigmoid(predicted_probs)
        
    predicted_results = predicted_probs[0] >0.5
    
    pos = np.where(predicted_results==1)[0]
    
   
    return {"labels" :label_col[pos],"prob":predicted_probs[0][pos]}

def get_label_from_index(index):
    return label_col[index]

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ... (previous code remains unchanged)

def perform_inference_with_visualization(model, image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create an empty white image
    white_image = np.ones((256, 256, 3), dtype=np.uint8) * 255

    # Load the person image
    person_image = cv2.imread(image_path)
    person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)

    # Resize the person image to fit within the white image
    person_image = cv2.resize(person_image, (128, 64))

    # Calculate the position to center the person image in the white image
    y_offset = (256 - person_image.shape[0]) // 2
    x_offset = (256 - person_image.shape[1]) // 2

    # Place the person image on the white image
    white_image[y_offset:y_offset + person_image.shape[0], x_offset:x_offset + person_image.shape[1]] = person_image

    predicted_results = []

    normalized_image = preprocess_image(image_path)
    normalized_image_tensor = normalized_image.to(device)
    normalized_image_tensor = normalized_image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(normalized_image_tensor)

    predicted_probs = output.cpu().numpy().astype(float)
    predicted_probs = sigmoid(predicted_probs)

    predicted_results = predicted_probs[0] > 0.5

    pos = np.where(predicted_results == 1)[0]

    labels = label_col[pos]
    probs = predicted_probs[0][pos]

    # Draw text labels on the image
    for label, prob in zip(labels, probs):
        text = f"{label}: {prob:.2f}"
        cv2.putText(white_image, text, ( person_image.shape[0], person_image.shape[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the result image
    plt.imshow(white_image)
    plt.axis('off')
    plt.show()

    # Save the result image
    cv2.imwrite(output_path, cv2.cvtColor(white_image, cv2.COLOR_RGB2BGR))

    return {"labels": labels, "prob": probs}

# ... (main function remains unchanged)


def main():
    parser = argparse.ArgumentParser(description='Perform inference on an image using a trained PyTorch model.')
    parser.add_argument('--model_path', type=str, default='./models/ResNet18_best_model.pth', help='Path to the trained PyTorch model file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image for inference')
    args = parser.parse_args()
    
    print(args.model_path)
    print(args.image_path)
    # Load the model
    model_ft = torch.load(args.model_path)

    # Perform inference on the input image
    results = perform_inference(model_ft, args.image_path)


    print("Predicted results:", results)

if __name__ == "__main__":
    main()
