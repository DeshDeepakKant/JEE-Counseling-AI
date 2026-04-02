# Results and Discussion: Comparison with Other Methods

## Benchmarking the PSO Ensemble approach

SwarmFusionDNN was directly contrasted against similar ensemble architectures (averaging, Gompertz function, Condorcet’s Jury theorem, Ant Colony Optimization, Grey Wolf Optimizer) that rely on static, experimental, or metaherustic weights.

- On the **Br35H dataset (99.66%)**, SwarmFusionDNN comfortably beat the next highest algorithm (ACO Weighted ensemble: 99.50%). 
- On the **Figshare dataset (96.14%)**, SwarmFusionDNN outcompeted alternative ensembles consistently, while maintaining identical hardware runtimes alongside comparable base models.

### Comparison to External Studies

When reviewing analogous automated diagnosis methodologies referenced within the study's literature review: 
- Anaraki et al. (CNN + Genetic Algorithms) achieved 94.20% on Figshare.
- Sultan et al. achieved 98.48% on Br35H.
- Asif et al. (Stacked Ensemble) originally attained 98.69% on brain tumor classifiers.

SwarmFusionDNN pushes accuracy boundaries up to **99.66% (Br35H)** and **96.14% (Figshare)** by employing ConvMixer structures explicitly paired with swarm-based optimizer rules. 

### Why Does It Prevail?
Traditional algorithms weigh sub-models somewhat arbitrarily. If MobileNetV2 stutters unexpectedly on a specific image type, simple aggregation metrics process incorrect predictions evenly. With PSO fine-tuning weights based on minimal misclassifications directly, models failing to represent certain domains logically inherit fractionally lower voting scopes, directly combating error overlap.
