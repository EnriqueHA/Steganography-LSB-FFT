import numpy as np
import cv2
import matplotlib.pyplot as plt

def EncodeImage(hiddenImage, carrierImage):
    carrierImageLab = cv2.cvtColor(carrierImage, cv2.COLOR_BGR2Lab)  # BGR to CIE Lab format
    l, a, b = cv2.split(carrierImageLab)  # Split into channels

    if len(hiddenImage.shape) > 2:  # If the input is a color image
        hiddenImage = cv2.cvtColor(hiddenImage, cv2.COLOR_BGR2GRAY)  # Convert hidden image to grayscale
        print("Hidden image will be embedded in grayscale.")
    
    # Embed the image in the channel
    aMod, dimHiddenImage = EmbedImage(hiddenImage, a)  # Modified channel and dimensions of hidden image
    aMod = aMod.astype(np.uint8)  # Match data type for merging
    
    # Encode dimensions of hidden image in the other channel with LSB
    height, width = dimHiddenImage
    dimStr = f"{height}-{width}"  # Store dimensions as a string
    bMod = LsbEncodeMessage(dimStr, b)

    # Merge the channels to create the Lab image
    stegoImageLab = cv2.merge((l, aMod, bMod))

    return stegoImageLab

def EmbedImage(hiddenImage, carrierChannel):
    hiddenLong, hiddenShort = max(hiddenImage.shape), min(hiddenImage.shape)
    carrierLong, carrierShort = max(carrierChannel.shape), min(carrierChannel.shape)

    # Resize hidden image if dimensions are incompatible
    if (hiddenLong > carrierLong // 2) or (hiddenShort > carrierShort // 2):
        hiddenImage = cv2.resize(hiddenImage, (carrierChannel.shape[1] // 2, carrierChannel.shape[0] // 2))
        dimHiddenImage = hiddenImage.shape
        print("Hidden image has been resized.")
    else:
        dimHiddenImage = hiddenImage.shape

    # Perform FFT on the channel
    channelFFT = np.fft.fft2(carrierChannel)
    channelFFTShifted = np.fft.fftshift(channelFFT)
    magChannel, phaseChannel = np.abs(channelFFTShifted), np.angle(channelFFTShifted)

    # Embed hidden image in the high-frequency region
    magChannel[-hiddenImage.shape[0]:, -hiddenImage.shape[1]:] += hiddenImage

    # Reassemble the modified channel
    channelFFTMod = magChannel * np.exp(1j * phaseChannel)
    channelFFTModShifted = np.fft.ifftshift(channelFFTMod)
    channelMod = np.fft.ifft2(channelFFTModShifted).real

    return channelMod, dimHiddenImage

def DecodeImage(stegoImage):
    l, aMod, bMod = cv2.split(stegoImage)  # Split into channels
    
    # Decode dimensions of hidden image from LSB
    dimHiddenImage = LsbDecodeMessage(bMod)
    height, width = map(int, dimHiddenImage.split("-"))  # Parse dimensions
    dimHiddenImage = (height, width)
    
    # Extract hidden image
    hiddenImage = ExtractImage(aMod, dimHiddenImage)
    hiddenImage = cv2.normalize(hiddenImage, None, 0, 255, cv2.NORM_MINMAX)  # Normalize for display
    hiddenImage = hiddenImage.astype(np.uint8)

    return hiddenImage

def ExtractImage(channel, dimHiddenImage):
    channelFFT = np.fft.fft2(channel)
    channelFFTShifted = np.fft.fftshift(channelFFT)
    magChannel = np.abs(channelFFTShifted)

    hiddenImage = magChannel[-dimHiddenImage[0]:, -dimHiddenImage[1]:]
    print(f"Dimensions of extracted image: {hiddenImage.shape}")  # Debugging

    return hiddenImage

def LsbEncodeMessage(msg, channel):
    encodedChannel = channel.copy()
    height, width = channel.shape

    msg += "\0"  # Null terminator to mark end of message
    binMsg = ''.join([format(ord(char), '08b') for char in msg])  # Convert message to binary

    msgIdx = 0
    for x in range(height):
        for y in range(width):
            if msgIdx < len(binMsg):
                pixel = int(channel[x, y])
                pixel = (pixel & ~1) | int(binMsg[msgIdx])  # Modify LSB
                encodedChannel[x, y] = pixel
                msgIdx += 1
            else:
                return encodedChannel

def LsbDecodeMessage(channel):
    binMsg = ""
    for x in range(channel.shape[0]):
        for y in range(channel.shape[1]):
            binMsg += str(channel[x, y] & 1)

    msg = ""
    for i in range(0, len(binMsg), 8):
        byte = binMsg[i:i+8]
        if byte == "00000000":  # Null terminator
            break
        msg += chr(int(byte, 2))

    print(f"Decoded message: {msg}")  # Debugging
    return msg

# Images to use
carrierImage = cv2.imread(r"spiral-galaxy.jpg")  # Loaded in BGR
hiddenImage = cv2.imread(r"quanteem_logo.jpg")  # Loaded in BGR

# Encode the hidden image into the carrier
stegoImage = EncodeImage(hiddenImage, carrierImage)
extractedImage = DecodeImage(stegoImage)

# Results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
carrierImageRGB = cv2.cvtColor(carrierImage, cv2.COLOR_BGR2RGB)
stegoImageRGB = cv2.cvtColor(cv2.cvtColor(stegoImage, cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2RGB)
# Carrier image
axs[0].imshow(carrierImageRGB)
axs[0].set_title("Carrier Image")
axs[0].axis("off")

# Stego image
axs[1].imshow(stegoImageRGB)
axs[1].set_title("Stego Image")
axs[1].axis("off")

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original hidden image (RGB)
axs[0].imshow(cv2.cvtColor(hiddenImage, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
axs[0].set_title("Original Hidden Image")
axs[0].axis("off")

# Extracted hidden image (Grayscale)
axs[1].imshow(extractedImage, cmap="gray")
axs[1].set_title("Extracted Hidden Image")
axs[1].axis("off")

plt.tight_layout()
plt.show()
