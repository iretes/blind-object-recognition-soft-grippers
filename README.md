# Blind object recognition with soft grippers

In scenarios where visual feedback is unavailable or unreliable, soft tactile sensing provides crucial information for object identification. Leveraging the compliance of soft grippers enables rich contact signals that can be used to recognize objects by touch alone.

Develop a machine learning pipeline that processes tactile and proprioceptive data from a soft gripper to classify objects. You can choose between using the provided simulation environment or the real-world dataset. The project involves preprocessing time-series, feature extraction, and training classifiers to recognize a set of known objects.

BONUS: Implement few-shot or metric-learning techniques to generalize recognition to new objects with minimal additional samples.

## Directory Structure

The project's directory structure includes the following main files and folders:
```
blind-object-recognition-soft-grippers
  ├── dataset                     # contains the dataset
  ├── deep_attentive_time_warping # original implementation of the Deep Attentive Time Warping algorithm [1]
  ├── results                     # stores the results of the experiments
  ├── CNN.ipynb                   # experiments with a Convolutional Neural Network [2]
  ├── DATW.ipynb                  # experiments with the Deep Attentive Time Warping approach
  ├── preprocessing.ipynb         # data preprocessing steps
  └── utils.py                    # utility functions
```

## References

- [1] Matsuo, Shinnosuke, et al. "Attention to warp: Deep metric learning for multivariate time series." Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part III 16. Springer International Publishing, 2021. **GitHub repository**: https://github.com/matsuo-shinnosuke/deep-attentive-time-warping/.
- [2] Wang, Zhiguang, Weizhong Yan, and Tim Oates. "Time series classification from scratch with deep neural networks: A strong baseline." 2017 International joint conference on neural networks (IJCNN). IEEE, 2017.

---

This project was developed for the “Robotics" course at the University of Pisa (a.y. 2024/2025).