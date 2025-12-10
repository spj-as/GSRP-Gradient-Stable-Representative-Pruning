# GSRP-Gradient-Stable-RepresentativeActivation Descriptor


GSRP is a two-stage one-shot pruning framework designed to improve robustness and compactness under noisy training conditions.
It removes redundant convolutional filters and identifies stable, meaningful weights using cross-batch gradient statistics.
GSRP achieves strong accuracy at high sparsity levels, even on noisy-label datasets.


# Method 

GSRP contains two complementary pruning components:
	1.	Filter-level GSRP — removes redundant filters via activation-space clustering
	2.	Weight-level GSRP — prunes unstable weights using gradient sign consistency + magnitude

Together, these form a two-stage pruning pipeline that produces subnetworks that are both sparse and structurally efficient.


## Stage 1: Filter-level GSRP (Redundancy-aware pruning)

Filter-level pruning identifies and removes convolutional filters that exhibit redundant activation behavior.

Activation Descriptor

For each filter:

$$a_i = \text{AvgActivation}(i)$$
This summarizes the filter’s average response across samples and spatial positions.

Redundancy Grouping

Cluster $\{a_i\}$ to find filters with similar activation behavior.
All filters in the same cluster $C$ are considered redundant.

Representative Selection

Keep the filter with the highest importance score $S_i$:

$$i^* = \arg\max_{i \in C} S_i$$

Remove all remaining filters in the cluster.

This step reduces model width, giving real inference-time speedups.


## Stage 2: Weight-level GSRP (Stable Gradient Scoring)

Weight-level GSRP constructs a noise-resilient importance score using gradients across multiple batches.

1. Cross-batch gradient sign consistency

$$C_i = \left| \frac{1}{T} \sum_{t=1}^{T} \text{sign}(g_i^{(t)}) \right|$$

Measures stability of gradient direction.

2. Average gradient magnitude

$$M_i = \frac{1}{T} \sum_{t=1}^{T} |g_i^{(t)}|$$

Measures the strength of the gradient signal.

3. Stability-aware importance score

$$S_i = C_i \cdot M_i$$

4. Pruning

Prune weights with low $S_i$.

This produces a noise-robust sparse subnetwork.


# Two-Stage GSRP Pruning Pipeline

Stage 1 → Filter-level GSRP
          Removes activation redundancy → compact backbone

Stage 2 → Weight-level GSRP
          Removes unstable or uninformative weights → sparse subnetwork

This hierarchical pruning achieves:
	•	high sparsity
	•	strong robustness to label noise
	•	good accuracy preservation

⸻

# Experimental Results

Evaluated on:
	•	CIFAR-10
	•	CIFAR-10N
	•	CIFAR-100N

