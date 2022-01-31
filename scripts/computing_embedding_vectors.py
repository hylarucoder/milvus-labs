import numpy as np
import os
from pathlib import Path
from towhee import pipeline

dataset_path = './data/dataset/'
images = []
vectors = []

embedding_pipeline = pipeline('towhee/image-embedding-resnet50')

for img_path in Path(dataset_path).glob('*'):
    vec = embedding_pipeline(str(img_path))
    norm_vec = vec / np.linalg.norm(vec)
    vectors.append(norm_vec.tolist())
    images.append(str(img_path.resolve()))

len(vectors)
