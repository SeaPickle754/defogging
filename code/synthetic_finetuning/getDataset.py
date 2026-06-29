import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggleprollc/mapillary-vistas-image-data-collection")

print("Path to dataset files:", path)