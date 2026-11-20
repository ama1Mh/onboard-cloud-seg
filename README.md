# Onboard Cloud Segmentation on CloudSEN12

## Research question
How do band reduction and INT8 quantization affect onboard cloud
detection when evaluated against human expert labels on CloudSEN12,
including the hard classes (thin cloud and cloud shadow) that binary,
Sen2Cor-labelled studies ignore?

## Contributions
1. First onboard-style compression study on a human-labelled benchmark
   (CloudSEN12 official split and metrics).
2. Per-class analysis: does INT8's small overall cost hide larger drops
   on thin cloud and shadow?
3. Band reduction: 4 vs 6 bands, compared with 13-band UNetMobV2.

## Hypothesis
INT8 costs < 0.01 BOA overall but more on thin cloud and shadow.
A 4-band model stays within 0.03 BOA of UNetMobV2 (0.92) on
cloud/no-cloud, while being __x smaller and __x faster per 512x512 patch.

## Data
CloudSEN12 high-quality split (Aybar et al., 2022), CC BY-NC 4.0.
Labels: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow.

## Closest prior work
Cratere et al. 2025 (arXiv 2504.03891): pruned + INT8 U-Net on FPGA,
trained on Sen2Cor masks (binary, 12 bands, accuracy only).