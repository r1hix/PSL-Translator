"""Real-time inference and spoken Urdu synthesis engine.

Maintains a 30-frame rolling keypoint buffer, evaluates ST-GCN predictions
in real-time at >= 30 FPS, and plays pre-cached Urdu pronunciation audio.
"""
