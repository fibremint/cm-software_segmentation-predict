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
        image_path = os.path.join(in_path, image.originalFilename)
        image.download(image_path)

        slide_seg = SlideSegmentation(cj=cj, tf_sess=tf_sess, image_instance=image, image_path=image_path,
                                      batch_size=4, threshold=0.5)

        predicted = slide_seg.predict()

        slide_seg.upload_annotation(predicted_data=predicted, project_id=cj.parameters.cytomine_id_project)

        # TODO: delete data saved on disk

        ray.shutdown()
        cj.job.update(status=Job.SUCCESS, progress=100, statusComment="Complete")


if __name__ == "__main__":
    main(sys.argv[1:])
