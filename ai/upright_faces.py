import os
import sys
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from .generate_final_images import compose_blurred_image

def imread_unicode(path):
    """Reads an image with Unicode file path using PIL and converts to BGR"""
    try:
        img_pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Failed to read image: {path} — {e}")
        return None


def get_rotation_angle(landmarks):
    # InsightFace keypoints: [right eye, left eye, nose, mouth right, mouth left]
    eye_r, eye_l = landmarks[0], landmarks[1]
    nose = landmarks[2]
    mouth_r, mouth_l = landmarks[3], landmarks[4]

    # Calculate angle between the eyes
    dx = eye_l[0] - eye_r[0]
    dy = eye_l[1] - eye_r[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Average Y positions
    eye_y = (eye_r[1] + eye_l[1]) / 2
    mouth_y = (mouth_r[1] + mouth_l[1]) / 2
    nose_y = nose[1]

    # Determine rotation
    if abs(angle) < 45:
        if eye_y < mouth_y and eye_y < nose_y:
            return 0
        else:
            return 180
    elif 45 <= angle <= 135:
        return -90  # Face is rotated left
    elif -135 <= angle <= -45:
        return 90   # Face is rotated right
    else:
        return 180  # Fallback


def rotate_image(img, angle):
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    else:
        return img


def process_image(image_path, output_path, face_app, size):
    img = imread_unicode(image_path)
    if img is None:
        print(f"Skipping unreadable image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img_rgb)

    if not faces:
        print(f"No face detected in {image_path}")
        return

    face = faces[0]  # Use the first detected face
    landmarks = face.kps
    angle = get_rotation_angle(landmarks)
    rotated_img = rotate_image(img, angle)
    
    if size:
        # Create the blurred background composite
        composite_image = compose_blurred_image(rotated_img, size)
        composite_image.save(output_path)
        print(f"Saved composite image: {output_path} (rotated {angle}°)")
    else:
        # Save the rotated image normally
        Image.fromarray(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)).save(output_path)
        print(f"Saved upright image: {output_path} (rotated {angle}°)")


def rotate_faces_in_folder(input_folder, output_folder, size=None):
    # Create new output folder
    os.makedirs(output_folder, exist_ok=True)

    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)
            process_image(in_path, out_path, face_app, size)

    return output_folder


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
        
    print("Rotating faces in:", input_folder)
    
    if len(sys.argv) >= 4:
        print("Creating composite images with blurred backgrounds")
        rotate_faces_in_folder(input_folder, output_folder, (int(sys.argv[3]), int(sys.argv[4])))
    else:
        rotate_faces_in_folder(input_folder, output_folder)