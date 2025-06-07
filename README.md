# Blind object recognition with soft grippers

In scenarios where visual feedback is unavailable or unreliable, soft tactile sensing provides crucial information for object identification. Leveraging the compliance of soft grippers enables rich contact signals that can be used to recognize objects by touch alone.

This repository includes experiments with various machine learning models for object classification from tactile and proprioceptive data collected from a real-world soft robotic gripper. Metric learning techniques (e.g., [1]) were also investigated to enable few-shot object recognition, thereby empowering the model to generalize to novel objects using only a limited amount of additional data.

## Directory Structure

The project's directory structure includes the following main files and folders:
```
blind-object-recognition-soft-grippers/
  ├── dataset/                     # contains the dataset
  ├── deep_attentive_time_warping/ # original implementation of the Deep Attentive Time Warping [1] method
  │     ├── dataloader.py          # data loading utilities
  │     ├── DATW.py                # Deep Attentive Time Warping core class
  │     ├── experiments.sh         # script to run full training and few-shot evaluation pipeline
  │     ├── few_shot_eval.py       # script for few-shot evaluation
  │     ├── model.py               # model definition
  │     ├── training.py            # script to run training
  │     └── utils.py               # utility functions
  ├── results/                     # stores the results of the experiments
  ├── siamese_network/             # implementation of a Siamese Network
  │     ├── dataloader.py          # data loading utilities
  │     ├── SN.py                  # Siamese Network core class
  │     ├── experiments.sh         # script to run full training and few-shot evaluation pipeline
  │     ├── few_shot_eval.py       # script for few-shot evaluation
  │     ├── model.py               # model definition
  │     └── training.py            # script to run training
  ├── CNN.ipynb                    # experiments with a Convolutional Neural Network
  ├── DATW.ipynb                   # experiments with the Deep Attentive Time Warping method
  ├── DTW.ipynb                    # experiments with Dynamic Time Warping
  ├── LSTM.ipynb                   # experiments with a Long Short-Term Memory network
  ├── preprocessing.ipynb          # data preprocessing steps
  ├── results_summary.ipynb        # summary of all experiments' results
  ├── shapelet+XGB.ipynb           # experiments with XGBoost on Shapelets features
  ├── SN.ipynb                     # experiments with the Siamese Network
  ├── stats+XGB.ipynb              # experiments with XGBoost on time and frequency domain features
  ├── transformer.ipynb            # experiments with a Transformer model
  └── utils.py                     # utility functions
```

## References

[1] Matsuo, Shinnosuke, et al. "Attention to warp: Deep metric learning for multivariate time series." Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part III 16. Springer International Publishing, 2021. **GitHub repository**: https://github.com/matsuo-shinnosuke/deep-attentive-time-warping/.

---

This project was developed for the "Robotics" course at the University of Pisa (a.y. 2024/2025).