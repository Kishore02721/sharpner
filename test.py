import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import os
from train import Generator

# Load model
def load_model(model_path, device):
    model = Generator()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Transform for grayscale images
transform = transforms.Compose([
    transforms.ToTensor()
])

# Test function
def test_image(generator, low_res_image_path, output_path, device):
    # Load the low-res image
    low_res_image = Image.open(low_res_image_path).convert("L")  # Convert to grayscale
    original_size = low_res_image.size  # Store the original size
    low_res_image = transform(low_res_image).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        # Generate the high-res image
        high_res_image = generator(low_res_image)

    # Convert the generated image back to PIL
    high_res_image = high_res_image.squeeze(0).squeeze(0).cpu()  # Remove batch and channel dimensions
    high_res_image = transforms.ToPILImage()(high_res_image)

    # Resize the generated high-res image to the original input size
    high_res_image = high_res_image.resize(original_size, Image.LANCZOS)

    # Save the output image in grayscale
    high_res_image.save(output_path, format="PNG")
    print(f"Super-resolved image saved at {output_path}")

# Main
if __name__ == "__main__":
    low_res_image_path = input("Enter the path to the low-res image: ")
    output_dir = input("Enter the directory to save the output image: ")
    epoch = input("Enter the epoch number: ")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"generator_epoch_{epoch}.pth"  # Correctly use the epoch number
    generator = load_model(model_path, device)
    
    test_image(generator, low_res_image_path, output_path, device)
