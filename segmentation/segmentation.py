import sys
from cytomine import CytomineJob
from cytomine.models import AnnotationCollection, ImageInstance
from cytomine.utilities import ObjectFinder
from shapely.geometry import Polygon
from neubiaswg5.exporter.mask_to_objects import fix_geometry
import tensorflow as tf
from segmentation.unet import UNet, preprocess_input
from segmentation.slide import SlideCrop
import numpy as np
import ray
from tensorflow import keras
from skimage.transform import resize

from segmentation.util import create_annotation_from_location, CytomineJobPartialUpdate

K = keras.backend


class SlideSegmentation:
    def __init__(self, cj: CytomineJob, tf_sess, image_instance: ImageInstance, image_path,
                 batch_size=8, threshold=0.5):
        self.cj = cj
        self.batch_size = batch_size
        self.threshold = threshold
        self.sess = tf_sess
        K.set_session(self.sess)

        self.slide_crop = SlideCrop(slide_path=image_path)
        self.image_instance = image_instance

        tile_size = self.slide_crop.tile_size
        self.model_input = tf.placeholder(tf.float32, shape=(None, tile_size, tile_size, 3))
        model = UNet().create_model(img_shape=[tile_size, tile_size, 3], num_class=2, rate=0.0,
                                    input_tensor=preprocess_input(self.model_input))
        self.model_output = tf.nn.softmax(model.output)

        try:
            model.load_weights('/app/model/checkpoint_5.h5')
            # model.load_weights('./model/checkpoint_5.h5')

        except Exception as e:
            sys.exit("ERROR: failed to load weights")

    def predict(self):
        wsi_seg_res = np.memmap("segmentation_result.bin", dtype=np.float16, mode='w+',
                                shape=self.slide_crop.predicted_slide_size())
        crop_batch_iterator, batch_len = self.slide_crop.crop(batch_size=self.batch_size)

        partial_update = CytomineJobPartialUpdate(cj=self.cj, start=5, end=85,
                                                  max_index=batch_len, prefix="Predict slide")

        with self.sess.as_default():
            for crop_batches_ids in crop_batch_iterator:
                for crop_batch_ids in crop_batches_ids:
                    partial_update.update()
                    self._predict_crop_batch(crop_batch=ray.get(crop_batch_ids), segmentation_result=wsi_seg_res)

        self.cj.job.update(progress=90, statusComment="Processing predicted results")
        wsi_seg_res = resize(wsi_seg_res, self.slide_crop.original_slide_size())
        wsi_seg_res = wsi_seg_res.astype(np.float32) / wsi_seg_res.max()
        wsi_seg_res[wsi_seg_res < self.threshold] = 0
        wsi_seg_res = (wsi_seg_res * 255.0).astype(np.uint8)

        return wsi_seg_res

    def _predict_crop_batch(self, crop_batch, segmentation_result):
        batch_images = [result[0] for result in crop_batch]

        feed_dict = {
            self.model_input: batch_images,
            K.learning_phase(): False
        }

        batch_predicted = self.sess.run(self.model_output, feed_dict=feed_dict)
        batch_logits = batch_predicted[:, :, :, 1]

        for i in range(len(batch_logits)):
            crop_batch[i][0] = batch_logits[i, :, :]

        for seg, loc in crop_batch:
            y, x = loc[1], loc[0]
            # there is overlapping
            seg_h, seg_w = seg.shape
            # prevent overflow, not happen usually
            if seg_h + y > segmentation_result.shape[0]:
                y = segmentation_result.shape[0] - seg_h
            if seg_w + x > segmentation_result.shape[1]:
                x = segmentation_result.shape[1] - seg_w
            # wsi_mask[y:y + seg_h, x:x + seg_w] = wsi_mask[y:y + seg_h, x:x + seg_w] + 1

            ## maximum
            segmentation_result[y:y + seg_h, x:x + seg_w] = np.maximum(segmentation_result[y:y + seg_h, x:x + seg_w],
                                                                       seg.astype(np.float16))

    def upload_annotation(self, predicted_data, project_id):
        self.cj.job.update(progress=95, statusComment="Uploading annotations")

        annotations = AnnotationCollection()
        components = ObjectFinder(predicted_data).find_components()
        locations = []
        for component in components:
            location = Polygon(component[0], component[1])

            if location.is_valid:
                locations.append(location)
            else:
                fixed = fix_geometry(location)

                if fixed.is_valid and not fixed.is_empty:
                    locations.append(fixed)

        for idx, loc in enumerate(locations):
            if not loc.is_valid:
                fixed = fix_geometry(loc)
                if fixed.is_valid and not fixed.is_empty:
                    loc[idx] = fixed

        annotations.extend([create_annotation_from_location(
            loc, self.image_instance.id, self.image_instance.height, project_id
        ) for loc in locations])

        annotations.save(chunk=20)
