# Methodology: Particle Swarm Optimization (PSO)

In SwarmFusionDNN, the challenge of determining the best weights ($w_i$) for the base models is solved by using **Particle Swarm Optimization (PSO)**.

## What is PSO?
PSO is a meta-heuristic algorithm imitating the swarm behavior of animals (e.g., bird flocking, fish schooling). Within a multidimensional search space, "particles" (solutions) move and explore continuously to locate the best global minimum. 

Unlike gradient-based methods, PSO excels at continuous multidimensional optimization without requiring a differentiable objective function.

### Particle Mechanics
In the context of the ensemble model:
- The **Position** of each particle represents the given weights assigned to the four models.
- The **Velocity** denotes how drastically these weights are updated iteratively.
- By tracking their **personal best** and the **global best** positions, the swarm converges efficiently on an optimal point in fewer iterations compared to typical genetic algorithms.

## The Objective Function
Every optimization task requires an objective. In SwarmFusionDNN, the primary goal is minimizing the **Classifier Error Rate**:

$$ \text{ClassifierErrorRate} = \left( \frac{\text{Misclassified Samples}}{\text{Total Samples}} \right) \times 100 $$

With each iteration, the PSO attempts to drastically minimize the Classifier Error Rate, indirectly maximizing the network's overall classification accuracy.

## Parameter Setup for PSO
After a rigorous grid search to adjust the swarm's hyperparameters, the model uses:

- **Population Size:** 50
- **Iterations:** 100
- **Weight Bounds:** [0, 1] (prevents individual base models from completely drowning out the rest).
- **Inertia Weight ($w$):** 0.5
- **Acceleration Coefficients:** $c_1 = 1, c_2 = 2$

## Convergence Behavior
The PSO architecture generally achieves severe error rate reduction within the first 20 iterations. After this initial dive, the optimization fine-tunes the weights slowly until stabilization, leveraging the accuracy of strong models and minimizing the errors of specific edge cases missed by standalone algorithms.
