import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import os
from train import Generator

def load_model(model_path, device):
    model = Generator()
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove "module." prefix from the state dict keys if it exists
    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    
    # Load the state dict into the model
    model.load_state_dict(state_dict, strict = False)
    
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
    # Set paths for Kaggle environment
    low_res_image_path = '/kaggle/working/dl_model/sample/001.bmp'  # Adjust this path
    output_dir = '/kaggle/working/output'  # Directory where output will be saved
    epoch = '19'  # Adjust this to the epoch number you want

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.png")

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"/kaggle/working/dl_model/generator_epoch_{epoch}.pth"  # Adjust this path

    # Load the model
    generator = load_model(model_path, device)
    
    # Test and save image
    test_image(generator, low_res_image_path, output_path, device)    
