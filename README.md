# Image-slider-ai

Automatically rotates images (in 90° steps) so that all faces appear upright using AI-powered face landmark detection.

---

## 📦 Installation

1. Clone the repository or download the files.
2. Make sure you are using **Python 3.10** or newer.
3. Install dependencies:

```bash
pip install -r requirements.txt
```


## How to Run
Place your .jpg, .jpeg, or .png images in the images/input folder.

### Run the script:

bash
Copy code
python rotate_faces.py
This will:

Detect faces in each image.

Determine the correct upright orientation using facial landmarks.

Rotate each image to the closest upright orientation (0°, 90°, -90°, or 180°).

Save the rotated images back in the same folder, overwriting the originals.

### Notes
* The algorithm uses eyes, nose, and mouth positions to check whether a face is upside down.

* Hebrew or Unicode filenames are supported.

* If no face is found in an image, it will be skipped.

* The script uses insightface, which provides robust face landmark detection.

Requirements.txt
```txt
insightface
opencv-python
numpy
pillow
```
