from cytomine.models import Annotation
from shapely.affinity import affine_transform
import numpy as np

from segmentation import opts


def create_annotation_from_location(location, id_image, image_height, id_project):
    def change_referential(p, height):
        return affine_transform(p, [1, 0, 0, -1, 0, height])

    parameters = {
        "location": change_referential(location, image_height).wkt,
        "id_image": id_image,
        "id_project": id_project,
    }

    return Annotation(**parameters)


class CytomineJobPartialUpdate:
    def __init__(self, cj, start, end, max_index, prefix=""):
        self.cj = cj
        self.start = start
        self.start_end_delta = end - self.start
        self.max_index = max_index
        self.prefix = prefix
        self.current_index = 0

    def update(self):
        self.current_index += 1
        progress = self.start + int(self.start_end_delta * (self.current_index / self.max_index))
        status_comment = self.prefix + " ({}/{})".format(self.current_index, self.max_index)
        self.cj.job.update(progress=progress, statusComment=status_comment)


def calculate_batch_split(tile, batch_size):
    batch_bytes = tile.size * tile.itemsize * batch_size

    return int(opts.max_store_bytes // batch_bytes)
