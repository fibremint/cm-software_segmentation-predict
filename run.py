import os
import sys

from cytomine import CytomineJob
from cytomine.models import Job
import tensorflow as tf
import ray

from segmentation import SlideSegmentation


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        cj.job.update(status=Job.RUNNING, progress=0, statusComment="Initializing...")

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_sess = tf.Session(config=tf_config)
        tf_sess.run(tf.global_variables_initializer())

        ray.init(num_cpus=os.cpu_count(), include_webui=False)

        cj.job.update(progress=1, statusComment="Fetching image...")
        image = cj.get_image_instance(cj.parameters.cytomine_id_image)
        image_path = os.path.join("/tmp", image.originalFilename)
        image.download(image_path)

        batch_size = cj.parameters.batch_size
        if batch_size is None:
            batch_size = os.cpu_count()

        slide_seg = SlideSegmentation(cj=cj, tf_sess=tf_sess, image_instance=image, image_path=image_path,
                                      batch_size=batch_size, num_slide_actor=cj.parameters.num_slide_actor,
                                      threshold=cj.parameters.threshold)
        predicted = slide_seg.predict()

        slide_seg.upload_annotation(predicted_data=predicted, project_id=cj.parameters.cytomine_id_project)

        # TODO: delete data saved on disk

        ray.shutdown()
        cj.job.update(status=Job.SUCCESS, progress=100, statusComment="Complete")


if __name__ == "__main__":
    main(sys.argv[1:])
