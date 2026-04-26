# AdaptiveSRE: Theory of Mind in Agentic Systems

## The Problem
Current SRE benchmarks test if an agent can fix a server. We test if an agent can understand when the definition of "correct" silently changes mid-incident.

## The Innovation
Two simultaneous hidden states: (1) which service is broken, and (2) what does the Lead Engineer currently value. The second state can drift at a random step without warning.

## Results
- Easy task: Gen 0 = -0.195, Gen 1 = -0.167 (+14.4% improvement)
- Hard task: Baseline = -0.424 (confirms genuine difficulty)
- Training: Reproducible across 2 independent CPU runs

## Demo
Live environment: https://huggingface.co/spaces/ashifsekh/adaptive-sre

## Training
Reproduce: https://colab.research.google.com/github/ashifsekh/Adaptive-SRE/blob/main/train_colab.ipynb
