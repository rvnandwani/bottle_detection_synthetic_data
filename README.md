# Note regarding compute resource
In case you do not have computing resources of your own most of the time all of the work can be done through Google Cloud / Google Collab free trial.
Let us know what resource you have used, and we will evaluate your assignment taking that into account.
If you have still problems on acquiring enough computing resources to train for many epochs, please at __minimum__ make the model trainable so we can test it on our side.

# Telexistence Take Home Assignment

Your assignment is to generate synthetic dataset of bottles and cans and train an object detector model on the generated dataset.

Model performance will be evaluated using the `target_images`. Apply your detector on these images and save the result.

Note that the complexity (image lighting conditions, drink occlusion, etc.) of each drink's detection will be taken into account in our evaluation.

However you are not expected to try to detect everything perfectly, since you will be asked to train a network PURELY from the synthetic images you have generated.

NOTE: You are not allowed to use a fully pre-trained model for this assignment, but you can use a pretrained backbone for feature extraction. If you choose to use a pretrained
backbone then make sure to inform us of this decision either through Slack or on the repository README.md.

NOTE2: For the object detector training and evaluation we encourage you to implement parts from "scratch" (e.g. your own code) when you can, thought not required. E.g. avoiding high-level APIs when training and evaluating your detector in your implementation would be ideal and will allow us to evaluate your skills in more depth.

## Task objective:
- Detection output needs to be the object category class (e.g. bottle or can), and its bounding box or segmentation
- Generate synthetic dataset for detecting bottles and cans
    - As the assignment time is limited we recommend using existing repository for the data generation. Some examples of good choices would be `blenderproc` (github.com/DLR-RM/BlenderProc) or `kubric` (github.com/google-research/kubric) that use blender for the simulation, but please choose the one you are most comfortable with!
    - Domain randomization techinques is encouraged to be used to boost the performance
    - We provide simple example syntethic data generation script using blender. It uses blender's `Eevee` to render the image. It is just meant to showcase how automated data generation could be done if you are not familiar with it, but feel free to use it as the base for your implementation in case you plan to also use blender. After installing blender you can execute the code with `{your-blender-executable} -b -P blender_generate_annotations.py` on Linux. You can also run it within blender from the `Text Editor` section. Script was tested on blender versions: 2.9, 3.2, 3.3 & 3.4
- For creating the object detector model to train on your synthethic data:
    1. You may choose the model of your preference and use fully "prebuild" model (like the one on `ssdlite_example.py`).
    2. (Optional) you can make your own custom object detector class following the instructions on `template_model.py`. You may use "prebuild" parts gotten from other repos for the detector (e.g. backbone, neck, head, etc.). Main point is to have all of them connected on the forward pass so the detection works and everything is initilized from the `__init__` function. If you are able to make the custom model working with decent results we will consider it more favorably than when using a fully "prebuild" model mentioned on (1.).
    3. (Optional) if you plan to make a "prebuild" model mentioned on (1.), modifying the default loss function used with the model and showing that it can achieve better results will be considered as a plus.
    4. (Optional) if don't have time for (2.) and (3.), still proposing ideas on how to improve the current chosen model architecture or loss function can also be potentially considered as a plus.
- Apply detector to `target_images` and save detected images as well as the annotations in any format (.pickle / .json etc)
- We have supplied an example model in the `ssdlite_example.py` that can be used in case you want to do quick initial experiments on the synthethic data quality and augmentation pipeline. Thought you can use any model for this purpose too!

NOTE: For the synthethic data generation you are welcome to use/add other 3d models than the ones we initially provide. There are no limitations regarding it.

NOTE2: In case you decided to use a guide or an existing implementation for data generation, please do not simply copy paste their implementation and add your own touch to it. For example most of the time the guides just offer basic introduction to domain randomization and not actually promise great results when used for training so it's better to add your own too. Also it helps us to review your skills better the more the implementation is made by you. Thought of course we know the fact that you have limited time available for the assignment so just do what you can here!

## Deliverables
- Source code for the whole implementation, e.g. synthethic data generation, training & eval script, etc.
    - Properly git managed repository is required. Using github private repository would be easy to track the intermediate development
    - NOTE: in case you have been using notebooks (.ipynb) in your experimenting phase please turn all the source code to python scripts (.py) for the final submission
- Reasoning of the network architecture selection
- future improvement section is required
- Synthetic dataset and model weights
- Object detector's best weights training session log(s) that contains all the info (hyperparameters used, random seeds, image ids of training and validation dataset, etc.) needed to replicate the same results. In case if you could not get this far then this can be left out!

### Additionally
- Active communication through slack if there're any questions is required, and is part of the process
- You have 7 days max duration for this assignment.

You can use `submission_template.md` as reference for writing your own implementation's README.md

## Other notes
- __We DO NOT expect everything to be done perfectly. Even if there're components that aren't met, do not worry about it, and please try to deliver your best instead.__
- Feel free to chat with the team members via slack any time, as we are also evaluating the discussion / communication skills here.
- Any further explanation requests are also welcome.
