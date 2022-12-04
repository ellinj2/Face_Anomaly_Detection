# ADIFI

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/ellinj2/Face_Anomaly_Detection?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/ellinj2/Face_Anomaly_Detection)
![GitHub issues](https://img.shields.io/github/issues-raw/ellinj2/Face_Anomaly_Detection)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ellinj2/Face_Anomaly_Detection)
![GitHub](https://img.shields.io/github/license/ellinj2/Face_Anomaly_Detection)

## Anomaly Detection in Face Images

This project hosts the code for implementing the ADIFI pipeline architecture for face anomaly detection in images as presented in our project for CSCI6962 Projects for ML and AI.

The full paper will be available upon release.

## Highlights
- **Object Detection** performed by [**FCOS**](https://github.com/tianzhi0549/FCOS/)
- **Anomaly Detection** using a deep two-class convolutional network and heuristic distribution transformation
- **Labeled Image Generation** with negative-log-likelihood scores displayed in bounding boxes

## Required Hardware
Our model was run on two machines. The model was successfully trained and inferenced using a Ryzen 7 5700G CPU with NVIDIA RTX 3060 GPU, and 11th gen Intel core vPRO i9 with NVIDIA A5000.

## Results
Below is a sample output from the ADIFI pipeline using an image selected from the [**WIDER Face**](https://deepai.org/dataset/wider-face) dataset.
![](https://github.com/ellinj2/Face_Anomaly_Detection/blob/readme/images/0_Parade_Parade_0_782.jpg)
