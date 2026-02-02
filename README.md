# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** R.Mounish Vamsi Kumar 
- **Register Number:** 212224240096

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('DIPT image-1.jpg', cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### 2. Print the image width, height & Channel.
```
img.shape
```

#### 3. Display the image using matplotlib imshow().
```
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray,cmap='grey')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```
img=cv2.imread('DIPT image-1.jpg')
cv2.imwrite('DIPT_image.png',img)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```
img=cv2.imread('DIPT image-1.jpg')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```
plt.imshow(img)
plt.show()
img.shape
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```
crop = img_rgb[0:450,200:550] 
plt.imshow(crop[:,:,::-1])
plt.title("Cropped Region")
plt.axis("off")
plt.show()
crop.shape
```

#### 8. Resize the image up by a factor of 2x.
```
res= cv2.resize(crop,(200*2, 200*2))
```

#### 9. Flip the cropped/resized image horizontally.
```
crop = img_rgb[0:450,200:550] 
plt.imshow(crop[:,:,::-1])
plt.title("Cropped Region")
plt.axis("off")
plt.show()
crop.shape
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```
flip= cv2.flip(res,1)
plt.imshow(flip[:,:,::-1])
plt.title("Flipped Horizontally")
plt.axis("off")
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```
img=cv2.imread('DIPT image-2.jpg',cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb.shape
text = cv2.putText(img_rgb, "Apollo 11 Saturn V Launch, July 16, 1969", (300, 700),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
plt.imshow(text, cmap='gray')  
plt.title("New image")
plt.show()
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rect_color = magenta
rcol= (255, 0, 255)
cv2.rectangle(img_rgb, (400, 100), (800, 650), rcol, 3)
```
##Output:
```
array([[[ 47,  75, 114],
        [ 47,  75, 114],
        [ 47,  75, 114],
        ...,
        [ 48,  64,  97],
        [ 48,  64,  97],
        [ 48,  64,  97]],

       [[ 47,  75, 114],
        [ 47,  75, 114],
        [ 47,  75, 114],
        ...,
        [ 48,  64,  97],
        [ 48,  64,  97],
        [ 48,  64,  97]],

       [[ 47,  75, 114],
        [ 47,  75, 114],
        [ 47,  75, 114],
        ...,
        [ 48,  64,  97],
        [ 48,  64,  97],
        [ 48,  64,  97]],

       ...,

       [[ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41],
        ...,
        [ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41]],

       [[ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41],
        ...,
        [ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41]],

       [[ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41],
        ...,
        [ 41,  41,  41],
        [ 41,  41,  41],
        [ 41,  41,  41]]], shape=(508, 603, 3), dtype=uint8)
```
#### 13. Display the final annotated image.
```
plt.title("Annotated image")
plt.imshow(img_rgb)
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```
img =cv2.imread('DIPT image-3.png',cv2.IMREAD_COLOR)
img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### 15. Adjust the brightness of the image.
```python
# Create a matrix of ones (with data type float64)
# matrix_ones = 
# YOUR CODE HERE
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(img, matrix)
img_darker = cv2.subtract(img, matrix)
# YOUR CODE HERE
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```
m = np.ones(img_rgb.shape, dtype="uint8") * 50
```

#### 18. Modify the image contrast.
```
img_brighter = cv2.add(img_rgb, m)  
img_darker = cv2.subtract(img_rgb, m)
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img_rgb), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_brighter), plt.title("Brighter Image"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_darker), plt.title("Darker Image"), plt.axis("off")
plt.show()
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```
matrix1 = np.ones(img_rgb.shape, dtype="float32") * 1.1
matrix2 = np.ones(img_rgb.shape, dtype="float32") * 1.2
img_higher1 = cv2.multiply(img.astype("float32"), matrix1).clip(0,255).astype("uint8")
img_higher2 = cv2.multiply(img.astype("float32"), matrix2).clip(0,255).astype("uint8")
```

#### 21. Merged the R, G, B , displays along with the original image
```
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_higher1), plt.title("Higher Contrast (1.1x)"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_higher2), plt.title("Higher Contrast (1.2x)"), plt.axis("off")
plt.show()
b, g, r = cv2.split(img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(b, cmap='gray'), plt.title("Blue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(g, cmap='gray'), plt.title("Green Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(r, cmap='gray'), plt.title("Red Channel"), plt.axis("off")
plt.show()
merged_rgb = cv2.merge([r, g, b])
plt.figure(figsize=(5,5))
plt.imshow(merged_rgb)
plt.title("Merged RGB Image")
plt.axis("off")
plt.show()
```

#### 22. Split the image into the H, S, V components & Display the channels.
```
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(h, cmap='gray'), plt.title("Hue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(s, cmap='gray'), plt.title("Saturation Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(v, cmap='gray'), plt.title("Value Channel"), plt.axis("off")
plt.show()
```
#### 23. Merged the H, S, V, displays along with original image.
```
merged_hsv = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
combined = np.concatenate((img_rgb, merged_hsv), axis=1)
plt.figure(figsize=(10, 5))
plt.imshow(combined)
plt.title("Original Image  &  Merged HSV Image")
plt.axis("off")
plt.show()

```

## Output:
- **i)** Read and Display an Image.
- 
*1.Read 'DIPT image-1.jpg' as grayscale and display:
  <img width="774" height="406" alt="image" src="https://github.com/user-attachments/assets/715f1173-038f-4516-b8c0-8a444c5ecd95" />

  2.Save image as PNG and display:
  
  <img width="685" height="419" alt="image" src="https://github.com/user-attachments/assets/3939fcae-b30b-49f9-84aa-bb290eff4417" />

  3.Cropped image:
  
  <img width="486" height="584" alt="image" src="https://github.com/user-attachments/assets/7b2e3fc3-f30d-4888-b2fd-c6f1e82895f9" />

  4.Resize and flip Horizontally:
  
  <img width="553" height="515" alt="image" src="https://github.com/user-attachments/assets/984f1c31-9e02-4ef9-8bba-acdaa0809a4a" />

  5.Read 'DIPT image-2.jpg' and Display the final annotated image:
  
<img width="672" height="560" alt="image" src="https://github.com/user-attachments/assets/89607e73-2117-4117-9be6-3c11fcafd181" />

 
- **ii)** Adjust Image Brightness.
- 1.Create brighter and darker images and display:

<img width="875" height="369" alt="image" src="https://github.com/user-attachments/assets/672ae7f5-0d52-458a-89ab-6dc65016880b" />

- **iii)** Modify Image Contrast.
- <img width="906" height="394" alt="image" src="https://github.com/user-attachments/assets/89d7daa9-456d-4ec4-badd-3cd91486a7d7" />
 
- **iv)** Generate Third Image Using Bitwise Operations.
- 
1.Split 'Boy.jpg' into B, G, R components and display:

<img width="892" height="394" alt="image" src="https://github.com/user-attachments/assets/343af3c9-ee5c-4f0a-b2da-9b0b57acf102" />

2.Merge the R, G, B channels and display:
<img width="493" height="550" alt="image" src="https://github.com/user-attachments/assets/6c3e8c9b-4c6e-4637-8a01-0ec933a94299" />

3.Split the image into H, S, V components and display:
<img width="898" height="383" alt="image" src="https://github.com/user-attachments/assets/33d1384c-98da-4cf7-b7a4-83c87e27ead4" />

4.Merge the H, S, V channels and display:

<img width="863" height="537" alt="image" src="https://github.com/user-attachments/assets/42c70d27-806f-46dd-a0a5-71778cb3ffd7" />

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

