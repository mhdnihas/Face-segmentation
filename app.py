import cv2
import numpy as np
import mediapipe as mp
import gradio as gr
from PIL import Image
import os

IMAGE_DIR = "/home/muhammed/Desktop/Face Segmentation"
DEFAULT_IMAGES = [
    os.path.join(IMAGE_DIR, "input_image1.png"),
    os.path.join(IMAGE_DIR, "input_image2.png")
]


def load_resized_images(target_width=100):
    images = []
    for img_path in DEFAULT_IMAGES:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            aspect_ratio = img.height / img.width
            target_height = int(target_width * aspect_ratio) 
            img_resized = img.resize((target_width, target_height))  
            images.append((img, img_resized))  
    return images


def initialize_models():
    return (
        mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5),
        mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1),
    )


def preprocess_image(input_image):
    img_array = np.array(input_image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)



def detect_face(face_detection, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    if not results.detections:
        return False, "No face detected in the image."
    if len(results.detections) > 1:
        return False, "Multiple faces detected."
    return True, None




def segment_face(selfie_segmentation, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(image_rgb)
    segmentation_mask = results.segmentation_mask



    binary_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255



    h, w = image_rgb.shape[:2]
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_image[:, :, :3] = image_rgb[:, :, :3]
    rgba_image[:, :, 3] = binary_mask

    return Image.fromarray(rgba_image)



def process_image(input_image):
    if input_image is None:
        return None, "No image selected or uploaded."
    try:
        face_detection, selfie_segmentation = initialize_models()
        image_bgr = preprocess_image(input_image)
        face_detected, message = detect_face(face_detection, image_bgr)
        if not face_detected:
            return None, message
        segmented_image = segment_face(selfie_segmentation, image_bgr)
        return segmented_image, "Face segmentation successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"



def create_interface():
    default_images = load_resized_images()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ Face Segmentation Tool")
        gr.Markdown("Upload an image or select a sample below, then click **Process Image**.")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil")
                process_btn = gr.Button("Process Image", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Segmented Face", type="pil")
                status = gr.Textbox(label="Status", value="Waiting for image...")



        if default_images:
            gr.Markdown("### **Click a sample image to use it:**")
            with gr.Row():
                for idx, (original_img, resized_img) in enumerate(default_images):
                    gr.Image(
                        value=resized_img, 
                        label=f"Sample {idx + 1}",
                        type="pil",
                        interactive=True
                    ).select(
                        fn=lambda img=original_img: (img, "Sample selected. Click Process Image."),
                        inputs=None,
                        outputs=[input_image, status]
                    )


        process_btn.click(
            fn=process_image,
            inputs=input_image,
            outputs=[output_image, status]
        )

    return demo


if __name__ == "__main__":
    app = create_interface()
    app.launch(debug=True, show_error=True)
