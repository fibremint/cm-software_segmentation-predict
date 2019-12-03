from cytomine.models import Annotation
from shapely.affinity import affine_transform


def create_annotation_from_location(location, id_image, image_height, id_project):
    def change_referential(p, height):
        return affine_transform(p, [1, 0, 0, -1, 0, height])

    parameters = {
        "location": change_referential(location, image_height).wkt,
        "id_image": id_image,
        "id_project": id_project,
    }

    return Annotation(**parameters)
