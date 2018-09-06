# Facial-Emotion-Recognition-CNN
A Project of Facial Emotion Recognition on CK+ Dataset by CNN, OpenCV for Classifying 8 basic emotions

# Pre-requisites:
1. Install Python 3.6.* (Tensorflow not support 3.7 yet)
2. Install OpenCV
3. Install Numpy
4. Install Tensorflow-GPU for 3.6.*
5. Install other packages used in the python files

# Pre-process images: (this step can be skipped if just use the attached prcessed datasets)
1. Unzip the original images from CK+ dataset to "extended-cohn-kanade-images\cohn-kanade-images"
2. Unzip the original labels from CK+ dataset to "Emotion_labels\Emotion"
3. Keep "haar" folder same level with the above folders (the xml files inside are for haar filter running on GPU enabled machine)
4. Run "pre_process.py" in Python, final testing and training datasets will be created in "dataset" folder

# Run CNN model:
1. Run "run.py" in Python to wait for the output.csv generated and charts of testing accuracy presented
