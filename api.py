from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from typing import List

from image_similarity import *

app = FastAPI()

def get_score(image1_rgb, image2_rgb, naed, ssim, cosine):
    # Determine the divisor for total percent calculation based on how many metrics are selected
    percent_proportion = 100.0 / sum([naed, ssim, cosine])
    total_percent = 0.0

    if naed:
        # Get euclidean distance between both neighbor-averaged images (requires grayscale images)
        ref_img_gray = cv2.cvtColor(np.float32(image1_rgb), cv2.COLOR_BGR2GRAY)
        target_img_gray = cv2.cvtColor(np.float32(image2_rgb), cv2.COLOR_BGR2GRAY)
        l2_na_value = get_L2_norm_neighbor_avg(image1_rgb, image2_rgb)
        l2_percent_of_img_size = l2_na_value / ((IMG_SIZE[0] * IMG_SIZE[1]) / 255) # smaller percentage is better
        normalized_l2_percent = 100 - (l2_percent_of_img_size * 100) # convert the small percentage to a large percentage equivalent

        naed_percent = (((((l2_percent_of_img_size) - L2_NA_THRESHOLD) * 100) / (0.0 - L2_NA_THRESHOLD)) * percent_proportion) / 100
        total_percent += naed_percent

    if ssim:
        # Get SSIM value
        ssim_value = get_ssim(image1_rgb, image2_rgb)
        ssim_percent = ((((ssim_value - 0.0) * 100) / (1.0 - 0.0)) * percent_proportion) / 100
        total_percent += ssim_percent

    if cosine:
        # Flatten to 1-D for cosine similarity
        image1_flattened = image1_rgb.flatten()
        image2_flattened = image2_rgb.flatten()
        cosine_sim_value = get_cosine_similarity(image1_flattened, image2_flattened)

        cosine_sim_percent = ((((cosine_sim_value - 0.0) * 100) / (1.0 - 0.0)) * percent_proportion) / 100
        total_percent += cosine_sim_percent

    return total_percent


@app.get('/image_similarity')
def get_root():
	return {
        'message': 'Welcome to the Image Similarity API!'
    }


@app.post('/image_similarity/compare')
async def compare_images(images: List[UploadFile] = File(...), naed: int = 1, ssim: int = 1, cosine: int = 1):
    img1 = images[0].file.read()
    img2 = images[1].file.read()

    file_bytes = np.asarray(bytearray(img1), dtype=np.uint8)
    image1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    file_bytes = np.asarray(bytearray(img2), dtype=np.uint8)
    image2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize
    image1_resized = cv2.resize(image1, (512, 512), interpolation=cv2.INTER_AREA) / 255
    image2_resized = cv2.resize(image2, (512, 512), interpolation=cv2.INTER_AREA) / 255

    # Convert to RGB color space
    image1_rgb = cv2.cvtColor(np.float32(image1_resized), cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(np.float32(image2_resized), cv2.COLOR_BGR2RGB)

    # Calculate the scores
    score = get_score(image1_rgb, image2_rgb, naed, ssim, cosine)

    return {'Similarity Score': f"{score:.2f}%"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)