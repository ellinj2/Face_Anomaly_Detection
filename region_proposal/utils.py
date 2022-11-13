import os
from glob import glob

def get_image_paths(data_path, recursive=False):
    if recursive:
        data_path = os.path.join(data_path, "**")
   
    image_paths, image_extensions = ([], ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"])
    for ie in image_extensions:
        image_paths = image_paths + glob(os.path.join(data_path, ie), recursive=recursive)

    return image_paths
