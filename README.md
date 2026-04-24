Overview:

This repository contains the implementation of our model for image segmentation using the SAM2 Large checkpoint with training configured on the UAVid Dataset. The project is designed for aerial imagery understanding and can be adapted for related remote sensing and flood-scene segmentation tasks.

Datasets:

Please download the required datasets and update their paths in the code before running the project.

1. FloodNet Dataset
Used for evaluation / experimentation on flood-scene aerial imagery.

Dataset link:
https://www.kaggle.com/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge

2. UAVid Dataset

Used for training the model.

Dataset link:
https://www.kaggle.com/datasets/dasmehdixtr/uavid-v1

Model Checkpoint:

Download the SAM2 Large checkpoint and update its path in the configuration file or script before execution.


Notes:

The current training pipeline is configured for the UAVid Dataset.
FloodNet can be used for additional testing or transfer experiments.
Ensure sufficient GPU memory when using the SAM2 Large model.


Architecture:

For detailed architecture understanding, please refer to the attached diagram included in this repository.

Contact:

For questions or architecture-related doubts, please refer to the diagram or open an issue in the repository.
