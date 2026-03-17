from dataclasses import dataclass
from typing import List
import numpy as np
from .image import gaussian_blur, image_to_binary, sobel_filter, write_image

@dataclass
class Blob:
    blob_image: np.ndarray
    histrogram: np.ndarray
    gradient_histrogram: np.ndarray
    center: tuple[float, float]

def add_to_gradient_histrogram(gradient_histrogram, angle):
    if angle < 45 - 22.5:
        gradient_histrogram[0] += 1
    elif angle < 90 - 22.5:
        gradient_histrogram[1] += 1
    elif angle < 135 - 22.5:
        gradient_histrogram[2] += 1
    elif angle < 180 - 22.5:
        gradient_histrogram[3] += 1
    elif angle < 225 - 22.5:
        gradient_histrogram[4] += 1
    elif angle < 270 - 22.5:
        gradient_histrogram[5] += 1
    elif angle < 315 - 22.5:
        gradient_histrogram[6] += 1
    elif angle < 360 - 22.5:
        gradient_histrogram[7] += 1
    else:
        gradient_histrogram[0] += 1

def get_blob_center(blob_image):
    # Get the center of the blob using vectorized operations
    mask = blob_image[:, :, 3] == 255
    
    # Get coordinates of valid pixels
    y_coords, x_coords = np.where(mask)
    
    sample_count = len(x_coords)
    if sample_count == 0:
        return 0.0, 0.0
    
    return np.sum(x_coords) / sample_count, np.sum(y_coords) / sample_count

def blobize(image, gradient_image, threshold=0.1) -> list[Blob]:
    
    magnitude = np.sqrt(np.sum(gradient_image**2, axis=-1))
    edge_image = (magnitude * 255).astype(np.uint8)
    write_image("outputs/edge_image.png", edge_image)
    edge_image = image_to_binary(edge_image, threshold)
    write_image("outputs/edge_image_binary.png", edge_image)
    
    visited = np.zeros(image.shape[:2], dtype=np.bool)
    
    def is_skippable(i, j):
        if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
            return True
        if edge_image[i][j] >= 128:
            # Is an edge pixel, skip
            visited[i][j] = True
            return True
        return visited[i][j]
    
    jobs = []
    blob_images = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if is_skippable(i, j):
                continue
            jobs.append((i, j))
            # Image with alpha channel
            blob_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            histrogram = np.zeros((8, 8, 8), dtype=np.int32)
            gradient_histrogram = np.zeros((8), dtype=np.int32)
            # Using while instead of recursion
            while len(jobs) > 0:
                i, j = jobs.pop()
                if is_skippable(i, j):
                    continue
                pixel = image[i][j]
                
                # Histrogram
                r = pixel[0]
                g = pixel[1]
                b = pixel[2]
                histrogram[r // 32][g // 32][b // 32] += 1
                
                # Gradient histrogram
                angle = np.degrees(np.arctan2(gradient_image[i][j][1], gradient_image[i][j][0]))
                add_to_gradient_histrogram(gradient_histrogram, angle)
                
                # Blob image
                blob_image[i][j][:3] = pixel
                blob_image[i][j][3] = 255
                
                visited[i][j] = True
                jobs.append((i + 1, j))
                jobs.append((i - 1, j))
                jobs.append((i, j + 1))
                jobs.append((i, j - 1))
            blob_images.append(Blob(
                blob_image=blob_image,
                histrogram=histrogram,
                gradient_histrogram=gradient_histrogram,
                center=get_blob_center(blob_image)
            ))
    return blob_images

def filter_blobs_by_pixel_count(blobs: List[Blob], min_pixel_count: int):
    return [blob for blob in blobs if np.sum(blob.blob_image[:, :, 3] == 255) > min_pixel_count]

def blob_distance(center1, center2):
    x1, y1 = center1
    x2, y2 = center2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def histrogram_distance(histrogram1, histrogram2):
    sum = np.sum(np.sqrt(histrogram1 * histrogram2))
    if sum == 0:
        return float('inf')
    return np.log(sum) * -1

def group_blobs(blobs1: List[Blob], blobs2: List[Blob], k1 = 1, k2 = 0.1, k3 = 1):
    groups = []
    for blob1 in blobs1:
        min_distance = float('inf')
        min_distance_blob_image = None
        for blob2 in blobs2:
            his_dist = histrogram_distance(blob1.histrogram, blob2.histrogram)
            center_dist = blob_distance(blob1.center, blob2.center)
            his_gradient_dist = histrogram_distance(blob1.gradient_histrogram, blob2.gradient_histrogram)
            distance = k1*his_dist + k2*center_dist + k3*his_gradient_dist
            if distance < min_distance:
                min_distance = distance
                min_distance_blob_image = blob2.blob_image
        if min_distance_blob_image is not None:
            groups.append([blob1.blob_image, min_distance_blob_image])
    return groups