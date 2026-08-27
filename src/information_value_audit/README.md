# Historical Information-Value Audit and R4 Stopping Analysis

## Overview

This module performs a retrospective information-value analysis for sequential LNP formulation optimization.

The purpose is to quantify how much information was gained from each experimental round and evaluate whether R4 represents a reasonable model-freeze point before prospective validation.

Importantly, this analysis does not claim that historical experiments were originally selected by this algorithm.

---

## Workflow

Historical rounds were reconstructed using a prior-round-only strategy:

- R2 candidates evaluated using R1 information
- R3 candidates evaluated using R1+R2 information
- R4 candidates evaluated using R1+R2+R3 information
- R5 candidates evaluated using frozen R1-R4 model only

R5 was never used for model training.

---

## Method

### Gaussian Process surrogate model

A Gaussian Process regression model was used to estimate:

- predicted transfection efficiency
- predictive uncertainty

For candidate formulation x:

\[
y(x)\sim N(\mu(x),\sigma^2(x))
\]

where:

- μ(x): predicted performance
- σ(x): predictive uncertainty

---

## Acquisition metrics

### Expected Improvement (EI)

Expected Improvement estimates remaining optimization opportunity:

\[
EI(x)=E[max(f(x)-y_{best},0)]
\]

A lower EI after additional rounds indicates reduced probability of discovering substantially improved formulations.

---

### Information Gain

Information gain estimates expected uncertainty reduction obtained from additional experiments.

---

### Novelty

Novelty measures distance between new formulations and previously explored formulation space.

---

## Virtual formulation space

A virtual feasible formulation pool was generated using Latin Hypercube Sampling within experimentally defined formulation boundaries.

Optimization opportunity was compared after:

- R1
- R1+R2
- R1+R2+R3
- R1+R2+R3+R4

---

## Main findings

P95 Expected Improvement:

| Stage | P95 EI |
|---|---|
| R1 | 0.016754 |
| R1+R2 | 0.007051 |
| R1+R2+R3 | 0.001489 |
| R1+R2+R3+R4 | 0.000317 |

The remaining optimization opportunity decreased substantially after R4.

The R4/R3 EI ratio was:

0.213

indicating diminishing information return from additional retrospective exploration.

---

## Repeatability analysis

Exact duplicate formulations were used to estimate experimental repeatability.

Empirical repeatability:

≈0.70

This suggests that biological and experimental variation contributes substantially to the remaining prediction uncertainty.

---

## Files
