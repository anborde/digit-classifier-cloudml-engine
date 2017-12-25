# Digit Classifier using Google CloudML Engine
A Image Based Classifier Using Google Cloud ML and Mnist Data

## Contents
- trainer/task.py
  - This is the job that we will perform on the Google's ML cloud. It consists of training model, the training task and the prediction task. The program is designed to work on MNIST image data which can be downloaded from the [Kaggle's Digit Recognizer Competetion](https://www.kaggle.com/c/digit-recognizer).

## Usage
- Download the repo and extract it on your machine.
- The code has been written to work on Google Cloud and hence would require some modification to run it locally.
- Use the following command in CMD or Terminal to run the code on the Google Cloud:

```
gcloud ml-engine jobs submit training <job-name> --module-name <module-name> --job-dir <job-dir> --package-path <package-path> --region <region> --runtime version 1.4 --scale-tier basic-gpu -- --data-dir <data-dir>
```

**job-name:** Give any job name as per your choice

**module-name:** In our case `trainer.task` i.e. the name of the code that you wish to run along with the outer package name. Here the code is task and package is trainer

**job-dir**: This is the path of your Google Cloud Storage Bucket which you wish to be your working directory while the job is running

**package-path:** The path of your job i.e. the path of 'trainer' folder on your local machine which will be uploaded on the cloud to run.

**region:** Cloud region where you wish to run this job. **Note: CloudML region and Google Cloud Storage Bucket Region should be same**

**scale-tier:** For fast processing as per our budget we have selected basic-gpu you can change it according to your preferences.

**data-dir:** Google Cloud Storage location of directory in which you have uploaded the training and test data.

- Enjoy the magic of Cloud and ML through this classic ML problem and do report if you face any issues. It took me days to figure out the working of Google Cloud ML, hope this repo eases out your efforts.

## License
- Provided in the repo
