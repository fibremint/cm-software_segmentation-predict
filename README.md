# Segmentation Predict
This software is [Cytomine extension software](https://doc.uliege.cytomine.org/display/ALGODOC/Overview) that provides prediction of bladder cancer on a medical slide image that uploaded on [Cytomine](https://uliege.cytomine.org/). 

Prediction is accomplished with image segmentation with 2 classes (positive or negative), and find indpendent area on a result images represented as WKT polygon. The WKT format data would be coverted into Cytomine compatible data (Annotation), and then uploaded to the Cytomine server.

## Pre-processing of the data
A medical slide image is quite big (over 100M pixels). 
The input that provided to deep learning model is prepared by crop (2k * 2k px., overlap 512px.) and resize (0.25) in sequential order.

## Improvement
With the introduction of the [Ray](https://github.com/ray-project/ray), make improvement on the spended time in inference by the process of pre-processing is running as concurrently.

In previous implementation of the inference process, the inference code have to wait until pre-processing is done and input batch is ready. But with this improvement, pre-processing can be run as asynchronoulsy and concurrently while the inference is ongoing and the inference doesn't have to wait for the pre-processing.

The class ```SlideActor``` in [slide.py](https://github.com/fibremint/cm-software_segmentation-predict/blob/master/segmentation/slide.py) is Ray remote task that retrieving a value from invocation with ```.remote()``` that provided by ```@ray.remote``` decorator can be evaluated asynchronously. In this case, when invoke the method that in this class remotely with ```.crop.remote()```, the cropping task starts asynchronously and stores its results in [Ray object store memory](https://docs.ray.io/en/releases-0.7.6/memory-management.html). And a number of ```SlideActor``` instances are created with given arguments and the tasks that cropping the slide image with specific region are allocated to each of them. This created instances are running on available resources (CPU threads) and it is an answer for how cropping can be run as concurrently. In the process of inference, pre-processed data can be evaluated with ```ray.get()``` that reads Ray objects that already evaluated and stored.

## Requirements
* [**Cytomine-bootstrap**](https://github.com/Cytomine-ULiege/Cytomine-bootstrap/tree/2019.1): commit@4eda498

## Requirements (dev)
* [**Cytomine-python-client**](https://github.com/Cytomine-ULiege/Cytomine-python-client): v2.4.0
* [**Neubias-WG5 utilities**](https://github.com/Neubias-WG5/neubiaswg5-utilities): v0.8.6

## Reference
* **Convert into Cytomine compatible data**: [Neubias-WG5/neubiaswg5-utilities](https://github.com/Neubias-WG5/neubiaswg5-utilities)
* **Inference**: [zizhaozhang/nmi-wsi-diagnosis](https://github.com/zizhaozhang/nmi-wsi-diagnosis)