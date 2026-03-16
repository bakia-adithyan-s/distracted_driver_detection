# Distracted Driver Detection ML Project

A machine learning project for classifying distracted driver behaviors (classes `c0` to `c9`) using classical ML, deep learning, and reinforcement-learning-style experiments, with a simple Flask GUI for inference.

## Repository Structure

- `dataset/`: dataset metadata and image folders
- `processed_data/`: preprocessed NumPy arrays (`X_train`, `X_test`, `y_train`, etc.)
- `notebooks/`: end-to-end experimentation notebooks
- `models/`: saved trained model artifacts
- `src/`: utility scripts (dataset checks, Grad-CAM)
- `gui/`: web app for running predictions

## Requirements

- Python 3.10+
- Virtual environment recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the GUI

From project root:

```bash
python gui/app.py
```

If using the existing venv path from this workspace:

```bash
c:/Users/bakia/projects/distracted_driver_ml_project/.venv/Scripts/python.exe gui/app.py
```

## Notebook Workflow

Suggested order:

1. `notebooks/01_view_images.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_ml_models.ipynb`
4. `notebooks/04_unsupervised_learning.ipynb`
5. `notebooks/05_deep_learning.ipynb`
6. `notebooks/06_reinforcement_learning.ipynb`

## Reinforcement Learning Notebook Notes

The reinforcement learning notebook now uses a DQN-style loop with:

- epsilon-greedy exploration
- replay buffer
- target network updates
- Bellman targets with discount factor `gamma`

This is a valid RL training setup, though the environment is synthetic for a classification-style task.

## Outputs

- Saved models are written to `models/`
- Visual outputs are generated in notebooks and GUI views

## License

Add a license file if you plan to distribute this project publicly.
