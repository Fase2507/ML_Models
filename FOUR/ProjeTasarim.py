
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

##LINK https://www.kaggle.com/code/fase22/gan-data-generating
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import os
import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torch.utils.data import DataLoader, Subset
import matplotlib as plt
import numpy as np
from torchvision.utils import save_image
import cv2
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

bleedingoriginal_dir = "/kaggle/input/brain-stroke-ct-dataset/Brain_Stroke_CT_Dataset/Bleeding/PNG"
normaloriginal_dir = "/kaggle/input/brain-stroke-ct-dataset/Brain_Stroke_CT_Dataset/Normal/PNG"


target_dir = '/kaggle/working/data/'
bleeding_dir = os.path.join(target_dir, 'bleeding')
normal_dir = os.path.join(target_dir, 'normal')

os.makedirs(target_dir,exist_ok=True)
os.makedirs(bleeding_dir, exist_ok=True)
os.makedirs(normal_dir, exist_ok=True)


for filename in os.listdir(bleedingoriginal_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        shutil.copy(os.path.join(bleedingoriginal_dir, filename), bleeding_dir)

for filename in os.listdir(normaloriginal_dir):
    if filename.lower().endswith(".png"):
        shutil.copy(os.path.join(normaloriginal_dir,filename),normal_dir)

temp_transform = Compose([ToTensor()])
dataset = ImageFolder(target_dir+'/', transform=temp_transform)
sample_img, _ = dataset[0]
original_h, original_w = sample_img.shape[1], sample_img.shape[2]
print(f"Original dataset image size: {original_h}x{original_w}")
print(f"Sample image shape: {sample_img.shape}")

latent_size = 50
batch_size = 16
labels = torch.ones(batch_size, 1).to(device)


class ConvGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            # latent vector input: (B, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=1, padding=0),  # 1 -> 4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh()
        )

    def forward(self, z, labels):
        return self.net(z)


# IMPROVED Discriminator - Added dropout and label smoothing
class ConvDiscriminator(nn.Module):
    def __init__(self, input_height, input_width):
        super().__init__()
        self.features = nn.Sequential(
            # No batch norm in first layer
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(1024, 1, kernel_size=8, stride=1, padding=0),  # 8 -> 1
            # No activation - using BCEWithLogitsLoss
        )

        # Store dimensions for later initialization
        self.input_height = input_height
        self.input_width = input_width
        self.classifier = None  # Will be initialized after moving to device

    def _init_classifier(self):
        """Initialize classifier after model is moved to device"""
        if self.classifier is None:
            with torch.no_grad():
                # Now both model and dummy input are on the same device
                dummy_input = torch.zeros(1, 3, self.input_height, self.input_width).to(next(self.parameters()).device)
                dummy_output = self.features(dummy_input)
                flattened_size = dummy_output.view(1, -1).size(1)
                print(f"Discriminator input size: {self.input_height}x{self.input_width}")
                print(f"After conv features: {dummy_output.shape}")
                print(f"Flattened size: {flattened_size}")

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_size, 1)
            ).to(next(self.parameters()).device)

    def forward(self, x, labels):
        # Initialize classifier on first forward pass
        if self.classifier is None:
            self._init_classifier()

        features = self.features(x)
        return self.classifier(features)


# Create generator and test its output size
print("\nCreating generator...")
G = ConvGenerator(latent_size).to(device)

# Test generator output to get actual dimensions
z = torch.randn(1, latent_size, 1, 1).to(device)
g_out = G(z, labels)
gen_h, gen_w = g_out.shape[2], g_out.shape[3]
print(f"Generator actual output shape: {g_out.shape}")
print(f"Generator dimensions: {gen_h}x{gen_w}")

# Use generator output dimensions for consistency
target_h, target_w = gen_h, gen_w
print(f"Will resize all images to: {target_h}x{target_w}")

# Define transformer with generator output dimensions
transformer = Compose([
    Resize((target_h, target_w)),
    ToTensor(),
    Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

num_classes = 2  # bleeding=0, normal=1


# Generate specific class images
def generate_class_images(class_name, num_images=4):
    """Generate images of a specific class"""
    G.eval()
    class_idx = dataset.class_to_idx[class_name]

    with torch.no_grad():
        noise = torch.randn(num_images, latent_size, 1, 1).to(device)
        labels = torch.full((num_images,), class_idx, dtype=torch.long).to(device)
        fake_images = G(noise, labels)
        fake_images = denorm(fake_images)

        # Display images
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))
        if num_images == 1:
            axes = [axes]

        for i in range(num_images):
            img = fake_images[i]
            if img.shape[0] == 3:
                axes[i].imshow(img.permute(1, 2, 0))
            else:
                axes[i].imshow(img[0], cmap='gray')
            axes[i].set_title(f'Generated {class_name} {i + 1}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    G.train()
    return fake_images

dataset = ImageFolder(target_dir+'/', transform=transformer)
print(f"Dataset size: {len(dataset)}")

# Test dataset image after transform
test_img, test_label = dataset[0]
print(f"Transformed dataset image shape: {test_img.shape}")

# Create discriminator with consistent dimensions
print(f"\nCreating discriminator with dimensions: {target_h}x{target_w}")
D = ConvDiscriminator(target_h, target_w).to(device)

# Set image dimensions
image_h, image_w = target_h, target_w

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Verify dimensions match
print(f"\nDimension verification:")
print(f"Generator output: {gen_h}x{gen_w}")
print(f"Dataset images: {test_img.shape[1]}x{test_img.shape[2]}")
print(f"Match: {gen_h == test_img.shape[1] and gen_w == test_img.shape[2]}")

batch_size = 16  # Reduced batch size for faster training
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
print(f"DataLoader batches: {len(data_loader)}")


index=np.random.randint(0,len(dataset)-1)
img,label=dataset[index]
print('label: ',label)
#print(img[:,10:15,10:15])
print('Pixel range:',torch.min(img).item(),'to',torch.max(img).item())

# %matplotlib inline

img_norm=denorm(img)
plt.imshow(img_norm.permute(1,2,0))##plt.imshow(img_norm[0],cmap='gray')
print("label",label)

# Final tests
print("\n=== Final Model Tests ===")
z_test = torch.randn(2, latent_size, 1, 1).to(device)
g_out_test = G(z_test,labels)
print("Generator output shape:", g_out_test.shape)

d_out_test = D(g_out_test,labels)
print("Discriminator output shape:", d_out_test.shape)

# Test with real images from dataset
real_images = torch.randn(2, 3, image_h, image_w).to(device)
d_real_out = D(real_images,labels)
print("Discriminator on real images shape:", d_real_out.shape)

# Test generator visualization
z = torch.randn(1, latent_size, 1, 1).to(device)
y = G(z,labels)
gen_imgs = denorm(y.detach().cpu())
plt.figure(figsize=(6, 6))
if gen_imgs.shape[1] == 3:  # RGB
    plt.imshow(gen_imgs[0].permute(1, 2, 0))
else:  # Grayscale
    plt.imshow(gen_imgs[0, 0], cmap='gray')
plt.title("Generated image test")
plt.axis('off')
plt.show()


z = torch.randn(1, latent_size, 1, 1).to(device)  # Add spatial dimensions
y = G(z,labels)
gen_imgs = denorm(y.detach().cpu())
plt.imshow(gen_imgs[0].permute(1, 2, 0))
plt.axis('off')
plt.savefig('generatedimg.png',)
plt.show()

# IMPROVED Training setup with better loss and optimizers
criterion = nn.BCEWithLogitsLoss()  # More stable than BCELoss
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))  # Lower LR
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Add learning rate schedulers
d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.99)
g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.99)


# Label smoothing for more stable training
# torch.empty(batch_size, 1).uniform_(0.8, 1.0).to(device)  # real
# torch.empty(batch_size, 1).uniform_(0.0, 0.2).to(device)  # fake

def get_real_labels(batch_size, device):
    return torch.ones(batch_size, 1).to(device) * 0.9  # 0.9 instead of 1.0


def get_fake_labels(batch_size, device):
    return torch.zeros(batch_size, 1).to(device) + 0.1  # 0.1 instead of 0.0


def train_discriminator(real_images):
    real_images = real_images.to(device)
    batch_size = real_images.size(0)

    # Label smoothing
    real_labels = get_real_labels(batch_size, device)
    fake_labels = get_fake_labels(batch_size, device)

    # Train on real images
    d_optimizer.zero_grad()
    outputs = D(real_images, labels)
    d_loss_real = criterion(outputs, real_labels)

    # Train on fake images
    z = torch.randn(batch_size, latent_size, 1, 1).to(device)
    fake_images = G(z, labels).detach()  # Detach to avoid training G
    outputs = D(fake_images, labels)
    d_loss_fake = criterion(outputs, fake_labels)

    # Combine losses and backprop
    d_loss = (d_loss_real + d_loss_fake) / 2
    d_loss.backward()
    d_optimizer.step()

    return d_loss, torch.sigmoid(outputs).mean().item()


def train_generator(batch_size):
    g_optimizer.zero_grad()
    z = torch.randn(batch_size, latent_size, 1, 1).to(device)
    labels = torch.ones(batch_size, 1).to(device)  # Want discriminator to think these are real
    fake_images = G(z, labels)
    outputs = D(fake_images, labels)
    g_loss = criterion(outputs, labels)
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images


# Better weight initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


D.apply(weights_init)
G.apply(weights_init)


sample_dir='/kaggle/working/samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


#def save_fake_images(epoch, fixed_noise):
    #fake_images = G(fixed_noise)
    #fake_images = denorm(fake_images)
   # save_image(fake_images, os.path.join(sample_dir, f'fake_epoch_{epoch}.png'), nrow=8)
  #  print(f'[Epoch {epoch}] Saved fake images.')
def save_fake_images(epoch, fixed_noise):
    G.eval()
    with torch.no_grad():
        fake_images = G(fixed_noise,labels)
        fake_images = denorm(fake_images)
        save_image(fake_images, f'{sample_dir}/fake_images_epoch_{epoch}.png', nrow=4)
        print(f"[Epoch {epoch}] Saved fake images.")
    G.train()


print(f"\nModels ready! Image dimensions: {image_h}x{image_w}")
print("Starting optimized training...")

# OPTIMIZED Training loop
num_epochs = 25  # Reduced epochs for faster experimentation
fixed_noise = torch.randn(16, latent_size, 1, 1).to(device)  # Reduced for faster generation

# Track training metrics
d_losses = []
g_losses = []

for epoch in range(1, num_epochs + 1):
    epoch_d_loss = 0
    epoch_g_loss = 0
    # for i, (real_images, labels) ...
    for i, (real_images, _) in enumerate(data_loader):

        # Train discriminator every iteration
        d_loss, d_score = train_discriminator(real_images)
        epoch_d_loss += d_loss.item()

        # Train generator every iteration (balanced training)
        g_loss, fake_images = train_generator(real_images.size(0))
        epoch_g_loss += g_loss.item()

        if (i + 1) % 5 == 0:  # More frequent logging
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, D(x): {d_score:.4f}')

    # Learning rate decay
    d_scheduler.step()
    g_scheduler.step()

    # Track average losses
    d_losses.append(epoch_d_loss / len(data_loader))
    g_losses.append(epoch_g_loss / len(data_loader))

    # Save images less frequently to speed up training
    if epoch % 2 == 0:  # Every 2 epochs
        save_fake_images(epoch, fixed_noise)

    print(f'Epoch [{epoch}/{num_epochs}] completed. Avg D Loss: {d_losses[-1]:.4f}, Avg G Loss: {g_losses[-1]:.4f}')

# Plot training progress
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.show()

# Show final generated images
G.eval()
with torch.no_grad():
    sample = G(fixed_noise, labels).cpu()
    sample = denorm(sample)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.title('Final Generated Images')
    grid = torchvision.utils.make_grid(sample, nrow=4)
    if grid.shape[0] == 3:
        plt.imshow(grid.permute(1, 2, 0))
    else:
        plt.imshow(grid[0], cmap='gray')
    plt.show()

print("Optimized training completed!")

# Alternative: Show just one single image
print("\nSingle generated image:")
with torch.no_grad():
    single_noise = torch.randn(1, latent_size, 1, 1).to(device)
    single_img = G(single_noise, labels).cpu()
    single_img = denorm(single_img)

    plt.figure(figsize=(8, 8))
    if single_img.shape[1] == 3:  # RGB
        plt.imshow(single_img[0].permute(1, 2, 0))
    else:  # Grayscale
        plt.imshow(single_img[0, 0], cmap='gray')
    plt.title('Single Generated Image')
    plt.axis('off')
    plt.show()


# Function to generate and save individual images
def generate_individual_images(num_images=5):
    G.eval()
    os.makedirs('/kaggle/working/individual_images/', exist_ok=True)

    with torch.no_grad():
        for i in range(num_images):
            # Generate one image at a time
            noise = torch.randn(1, latent_size, 1, 1).to(device)
            img = G(noise, labels).cpu()
            img = denorm(img)

            # Save individual image
            save_image(img, f'/kaggle/working/individual_images/generated_{i + 1}.png')

            # Display individual image
            plt.figure(figsize=(6, 6))
            if img.shape[1] == 3:  # RGB
                plt.imshow(img[0].permute(1, 2, 0))
            else:  # Grayscale
                plt.imshow(img[0, 0], cmap='gray')
            plt.title(f'Generated Image {i + 1}')
            plt.axis('off')
            plt.show()

    print(f"Generated and saved {num_images} individual images!")


# Generate 5 individual images
generate_individual_images(15)

print("Optimized training completed!")
#torch.save(model.state_dict(), 'kaggle/working')

plt.plot(d_losses, label='D Loss')
plt.plot(g_losses, label='G Loss')
plt.legend()
plt.title("Losses")
plt.show()