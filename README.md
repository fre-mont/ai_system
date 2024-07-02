# ai_system
2023-1 : Emotion-based Music Recommendation System  

### Goal ###
- Classify emotions from webcam input.
- Perform EDA and clustering on music dataset audio features.
- Recommend music based on vector similarity, aligning with the user's current emotional state and year-specific music preferences.

### Dataset ###
- Emotion dataset : https://www.kaggle.com/datasets/msambare/fer2013
- Music dataset : https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset

### Algorithm ###
- Emotion Classification (Emotion_detection.ipynb)
- Audio Feature Clustering (EDA_Clustering.ipynb)

### Run ###
```bash
python recommend.py
```

> - Input: Emotion inference from webcam (5 seconds) & year selection.
> - Output: Ranked list of top 1-5 song recommendations based on the input year and inferred emotion.
> - Additional feature: Linking to YouTube for the recommended songs.
