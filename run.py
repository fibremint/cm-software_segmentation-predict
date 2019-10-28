"""
Workflow:
1. fetch whole slide

"""
import os, sys, time, socket
from multiprocessing import cpu_count

from cytomine import CytomineJob
from cytomine.models import Job, Annotation, AnnotationCollection
from cytomine.utilities import ObjectFinder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from imageio import imread, imwrite
from neubiaswg5.exporter.mask_to_objects import fix_geometry
from segmentation import unet
from segmentation.wsi_util import SlideLoader
K = keras.backend


def change_referential(p, height):
    return affine_transform(p, [1, 0, 0, -1, 0, height])


def create_annotation_from_location(location, id_image, image_height, id_project):
    parameters = {
        "location": change_referential(location, image_height).wkt,
        "id_image": id_image,
        "id_project": id_project,
    }

    return Annotation(**parameters)


def init_model():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_input = tf.placeholder(tf.float32, shape=(None, 512, 512, 3))
    model = unet.UNet().create_model(img_shape=[512, 512, 3], num_class=2, rate=0.0,
                                     input_tensor=unet.preprocess_input(model_input))
    model_output = tf.nn.softmax(model.output)

    sess.run(tf.global_variables_initializer())

    try:
        model.load_weights('./model/checkpoint_5.h5')
    except:
        sys.exit("ERR: failed to load weights")

    return sess, model_input, model_output


def predict_img(sess: tf.Session, model_input, model_pred, in_image, in_path, out_path, cj: CytomineJob):
    slide_handler = SlideLoader(batch_size=8, to_real_scale=4, level=0, imsize=512)

    try:
        slide_iter, num_batches, slide_name, actual_slide_size, num_patches = slide_handler.get_slide_iterator(
            path=os.path.join(in_path, in_image.originalFilename), down_scale_rate=1, overlapp=512)

        wsi_seg_res = np.zeros(actual_slide_size, dtype=np.float16)
        wsi_mask = np.zeros(actual_slide_size, dtype=np.float16)

        with sess.as_default():
            for step, (batch_imgs, locs) in cj.monitor(enumerate(slide_iter),
                                                       start=20, end=95, period=0.05, prefix="Generate segmentation : "):
                # sys.stdout.write('{}-{},'.format(step, (batch_imgs.shape[0])))
                # sys.stdout.flush()
                cj.job.update(statusComment="{} / {}".format(step, len(batch_imgs)))

                feed_dict = {
                    model_input: batch_imgs,
                    K.learning_phase(): False
                }

                batch_pred = sess.run(model_pred, feed_dict=feed_dict)
                batch_logits = batch_pred[:,:,:,1]
                for id, (seg, im, loc) in enumerate(zip(batch_logits, batch_imgs, locs)):
                    y, x = loc[0], loc[1]
                    # there is overlapping
                    seg_h, seg_w = seg.shape
                    # prevent overflow, not happen usually
                    if seg_h + y > wsi_seg_res.shape[0]:
                        y = wsi_seg_res.shape[0] - seg_h
                    if seg_w + x > wsi_seg_res.shape[1]:
                        x = wsi_seg_res.shape[1] - seg_w
                    wsi_mask[y:y + seg_h, x:x + seg_w] = wsi_mask[y:y + seg_h, x:x + seg_w] + 1

                    ## maximum
                    wsi_seg_res[y:y + seg_h, x:x + seg_w] = np.maximum(wsi_seg_res[y:y + seg_h, x:x + seg_w],
                                                                       seg.astype(np.float16))
        wsi_seg_res = resize(wsi_seg_res, (actual_slide_size[0] * 4, actual_slide_size[1] * 4))
        wsi_seg_res = wsi_seg_res.astype(np.float32) / wsi_seg_res.max()
        wsi_seg_res[wsi_seg_res < cj.parameters.threshold] = 0
        res_as_im = (wsi_seg_res * 255.0).astype(np.uint8)

        result_filename = "{}.tif".format(in_image.originalFilename.split('.')[0])
        imwrite(os.path.join(out_path, result_filename), res_as_im)

        return result_filename

    except Exception as e:
        print(e)


def upload_data(out_path, out_filename, in_image, project_id):
    data = imread(os.path.join(out_path, out_filename))
    annotations = AnnotationCollection()
    # slices = mask_to_objects_2d(data)
    components = ObjectFinder(data).find_components()
    # locations = get_geometries(components)
    locations = []

    for component in components:
        location = Polygon(component[0], component[1])

        if location.is_valid:
            locations.append(location)
        else:
            fixed = fix_geometry(location)

            if fixed.is_valid and not fixed.is_empty:
                locations.append(fixed)

    # for idx, loc in enumerate(locations):
    #     if not loc.is_valid:
    #         fixed = fix_geometry(loc)
    #         if fixed.is_valid and not fixed.is_empty:
    #             loc[idx] = fixed

    annotations.extend([create_annotation_from_location(
        loc, in_image.id, in_image.height, project_id
    ) for loc in locations])
    # annotations.extend(create_annotation_from_location(locations, in_image.id, in_image.height, project_id))

    # annotations.extend([create_annotation_from_slice(
    #     s, in_image.id, in_image.height, project_id) for s in slices
    # ])

    annotations.save(chunk=20, n_workers=cpu_count() * 2)


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        cj.job.update(Job.RUNNING, progress=0, statusComment="Initializing...")

        sess, model_input, model_output = init_model()

        base_path = "{}".format(os.getenv("HOME"))
        working_path = os.path.join(base_path, str(cj.job.id))
        in_path = os.path.join(working_path, "in")
        out_path = os.path.join(working_path, "out")

        if not os.path.exists(working_path):
            os.makedirs(working_path)
            os.makedirs(in_path)
            os.makedirs(out_path)

        cj.job.update(progress=1, statusComment="Fetching image...")
        image = cj.get_image_instance(cj.parameters.cytomine_id_image)
        image.download(os.path.join(in_path, image.originalFilename))

        predicted_filename = predict_img(
            sess=sess,
            model_input=model_input,
            model_pred=model_output,
            in_image=image,
            in_path=in_path,
            out_path=out_path,
            cj=cj
        )

        cj.job.update(progress=95, statusComment="Uploading annotations...")
        upload_data(out_path, predicted_filename, image, cj.parameters.cytomine_id_project)

        cj.job.update(Job.SUCCESS, progress=100, statusComment="Complete")


if __name__ == "__main__":
    main(sys.argv[1:])
