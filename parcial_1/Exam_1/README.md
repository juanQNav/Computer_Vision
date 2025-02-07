# First Partial Exam on Computer Vision

<table align="center">
  <tr>
    <td align="center">
      <img src="./images/Jorge_Campos.jpg" alt="Original Image" height="500px" style="max-width: 100%;"><br>
      <b>Original Image</b>
    </td>
    <td align="center">
      <img src="./images/Rotated Image.png" alt="Processed Image" height="500px" style="max-width: 100%;"><br>
      <b>Processed Image</b>
    </td>
  </tr>
</table>

## Description
This project contains the development of the first partial exam on Computer Vision. It works with an image of the former Mexican soccer player Jorge Campos in action, which has been edited to include noise and blur. The objective is to apply various image processing techniques to enhance its quality and extract relevant information.

## Requirements
To run the provided code in this exam, the following Python libraries must be installed:

```bash
pip install opencv-python matplotlib numpy
```

## Project Content
The code includes several image processing techniques using OpenCV and NumPy. Below are the steps applied:

### 1. **Loading and displaying the original image**
The image "Jorge_Campos.jpg" is loaded and displayed in RGB format.

### 2. **Sharpening filter (Sharpened filter - Median Kernel)**
A sharpening filter is applied using a Laplacian kernel to highlight edges and enhance image details.

### 3. **Noise reduction (Non-local means)**
The `cv2.fastNlMeansDenoisingColored` method is used to remove noise present in the image and improve its clarity.

### 4. **Low-pass filter (Fourier Transform)**
A low-pass filter based on the Fourier Transform is applied to smooth the image by eliminating high frequencies.

### 5. **Additional sharpening filter (Smooth Kernel)**
A sharpening filter is applied again using a smoothing kernel to further improve the image quality.

### 6. **Image cropping (Cropped Image)**
A specific section of the image is extracted to focus on the player's key action.

### 7. **Comparison with the cropped original image**
The processed image is compared with the cropped version of the original image.

### 8. **Image rotation**
The final image is rotated by -35 degrees to change the perspective and enable visual analysis of the scene.

## Usage
To run the code, simply execute the script in a Python environment that supports Jupyter Notebook or any other environment compatible with OpenCV and Matplotlib.

## References
- **Image filtering**: [OpenCV Documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html). Accessed on September 15, 2024.
- **Non-local means denoising**: [OpenCV Tutorial](https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html). Accessed on September 15, 2024.
- **Python Computer Vision Tutorials - Image Fourier Transform**: [WSTHUB](https://wsthub.medium.com/python-computer-vision-tutorials-image-fourier-transform-part-3-e65d10be4492). Accessed on September 15, 2024.