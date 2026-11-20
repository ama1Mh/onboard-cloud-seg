Idea 1: Give the model location and season
Idea 2: A semi-labelling system to cut labelling work

onboard theme through a teacher-student setup:

Train a big, accurate model on the 10,000 labelled patches (the teacher).
Use it to label the 29,250 unlabelled patches automatically. These are called pseudo-labels.
Train your tiny onboard model (the student) on real plus pseudo-labels.

That gives you a clean research question: does pseudo-labelled data make a tiny onboard model more accurate, and by how much? A “daily or weekly” version, which labels new images as they arrive, is the engineering form of the same idea. For a paper, frame it as semi-supervised learning or knowledge distillation.

