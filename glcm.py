import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte


def glcm_process(image: Image.Image) -> pd.DataFrame:
    """
    Extract GLCM features from a PIL Image.
    
    Args:
        image: PIL Image object (RGB or grayscale)
    
    Returns:
        DataFrame with GLCM features (1 row)
    """
    # GLCM configuration
    properties = [
        'contrast',
        'dissimilarity',
        'homogeneity',
        'energy',
        'correlation'
    ]

    angles_deg = [0, 45, 90, 135, 180]
    angles_rad = [np.deg2rad(a) for a in angles_deg]

    # Preprocess image
    im = image.convert("RGB")
    im = im.resize((128, 128))
    gray = img_as_ubyte(rgb2gray(np.array(im)))

    # Compute GLCM
    glcm = graycomatrix(
        gray,
        distances=[50],
        angles=angles_rad,
        levels=256,
        symmetric=True,
        normed=True
    )

    # Extract features
    feature_dict = {}
    for prop in properties:
        values = graycoprops(glcm, prop).flatten()
        for angle, val in zip(angles_deg, values):
            col_name = f"glcm_{prop}_angle_{angle}"
            feature_dict[col_name] = val

    # Convert to DataFrame (1 row)
    df = pd.DataFrame([feature_dict])
    return df
