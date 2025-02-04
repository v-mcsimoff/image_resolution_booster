# 📸 Image Resolution Booster Telegram Bot

This is a **Telegram bot** that utilizes a **deep learning model** to upscale images using **Swift-SRGAN**. The bot runs in a **Jupyter Notebook** and employs **PyTorch** for inference.

## 🚀 Features
- Accepts images from Telegram users
- Processes images with a **Super-Resolution GAN (SRGAN)** model
- Upscales images by **4x** while preserving quality
- Returns the upscaled image to the user via Telegram

## 🛠️ Requirements
Ensure you have the following dependencies installed:

```bash
pip install torch torchvision aiogram numpy opencv-python pillow tqdm
```

Additionally, you need to set up **Jupyter Notebook** if you are running this in a notebook environment.

## 🏗️ Model Architecture
The bot uses a **Swift-SRGAN Generator** built with PyTorch:
- **Separable Convolutions** for efficient computation
- **Residual Blocks** to preserve image features
- **Upsample Blocks** with **Pixel Shuffle** to enhance resolution
- **Discriminator Network** for adversarial training

### 🔬 Generator Architecture
The `Generator` class follows the SRGAN approach, using:
- **Convolutional layers**
- **Residual blocks**
- **PixelShuffle upsampling**
- **Tanh activation** to output normalized images

```python
class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16, upscale_factor=4):
        super(Generator, self).__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residual = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsampler = nn.Sequential(
            *[UpsampleBlock(num_channels, scale_factor=2) for _ in range(upscale_factor//2)]
        )
        self.final_conv = SeperableConv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
    
    def forward(self, x):
        initial = self.initial(x)
        x = self.residual(initial)
        x = self.convblock(x) + initial
        x = self.upsampler(x)
        return (torch.tanh(self.final_conv(x)) + 1) / 2
```

### 🎭 Discriminator Architecture
The `Discriminator` is a convolutional neural network that distinguishes between real and generated high-resolution images.

```python
class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.input_conv = SeperableConv2d(in_channel, 64, 3, 2, bias=False)
        self.activation = nn.LeakyReLU()
        self.blocks = [DiscriminatorBlock(64, 128, 1), DiscriminatorBlock(128, 128, 2),
                       DiscriminatorBlock(128, 256, 1), DiscriminatorBlock(256, 256, 2),
                       DiscriminatorBlock(256, 512, 1), DiscriminatorBlock(512, 512, 2),
                       DiscriminatorBlock(512, 512, 1)]
    
    def forward(self, x):
        x = self.input_conv(x)
        x = self.activation(x)
        for layer in self.blocks:
            x = layer(x)
        return x
```

## 🤖 Bot Setup
1. **Create a Telegram Bot**
   - Use [BotFather](https://t.me/botfather) to create a bot and obtain an API token.
   
2. **Run the Jupyter Notebook**
   - Open the notebook and ensure all dependencies are installed.
   - Set your Telegram API token in the bot script.

3. **Run the bot**
   - Execute the script inside the Jupyter Notebook:
   
   ```python
   !python bot.py
   ```

## 📩 Usage
- Send an image to the bot on Telegram.
- The bot will process and upscale the image.
- The upscaled image is sent back to the user.

## 🏗️ Future Improvements
- Support for different upscale factors (2x, 4x, 8x)
- Deploy on a cloud server (AWS, Google Colab, etc.)
- Add more deep learning-based enhancements

## 📝 License
This project is open-source and available under the MIT License.

