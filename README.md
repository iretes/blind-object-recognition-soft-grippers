# Blind object recognition with soft grippers

In scenarios where visual feedback is unavailable or unreliable, soft tactile sensing provides crucial information for object identification. Leveraging the compliance of soft grippers enables rich contact signals that can be used to recognize objects by touch alone.

This repository includes experiments with various machine learning models for object classification from tactile and proprioceptive data collected from a real-world soft robotic gripper. Metric learning techniques were also explored to enable few-shot object recognition, allowing the model to generalize to previously unseen objects using minimal additional data.

## Directory Structure

The project's directory structure includes the following main files and folders:
```
blind-object-recognition-soft-grippers
  ├── dataset                     # contains the dataset
  ├── deep_attentive_time_warping # original implementation of the Deep Attentive Time Warping approach [1]
  ├── results                     # stores the results of the experiments
  ├── siamese_network             # implementation of a Siamese Network for metric learning
  ├── CNN.ipynb                   # experiments with a Convolutional Neural Network
  ├── DATW.ipynb                  # experiments with the Deep Attentive Time Warping approach
  ├── LSTM.ipynb                  # experiments with a Long Short-Term Memory network
  ├── results_summary.ipynb       # summary of all experiments results
  ├── preprocessing.ipynb         # data preprocessing steps
  ├── README.md                   # this file
  ├── SN.ipynb                    # experiments with the Siamese Network
  └── utils.py                    # utility functions
```

## References

[1] Matsuo, Shinnosuke, et al. "Attention to warp: Deep metric learning for multivariate time series." Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part III 16. Springer International Publishing, 2021. **GitHub repository**: https://github.com/matsuo-shinnosuke/deep-attentive-time-warping/.

---

This project was developed for the “Robotics" course at the University of Pisa (a.y. 2024/2025).