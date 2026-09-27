"""Image preprocessing shared by training AND serving (implemented in Phase 2).

One implementation for both sides prevents training/serving skew: the model must
see images resized and normalized exactly the way it saw them during training.
"""
