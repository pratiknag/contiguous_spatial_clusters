# Contiguous spatial clusters

## Overview

This repository contains a pipeline for **spatial clustering of 2D data** using a combination of k-nearest neighbors, local averaging, and k-means clustering. The pipeline enforces **spatial contiguity**, merges small clusters, and outputs a clean visualization of the resulting clusters.  

The Python implementation uses **FAISS on CPU** for efficient kNN computations.  

---

## Features

- kNN-based local mean calculation  
- Feature scaling and k-means clustering  
- Construction of a kNN graph and spatially-connected component analysis  
- Merging of small clusters into large contiguous clusters  
- Generation of a plot showing final spatial clusters  

---

## Dataset

The pipeline uses the datasets provided by the KAUST Spatial Statistics Large Datasets competition. The dataset used for the computations can be found here:  

[KAUST Spatial Statistics Dataset](https://repository.kaust.edu.sa/items/29f6ba54-93c7-45db-b83c-9749a2c226c1)

Download the relevant CSV files and place them in the current working directory.

---

## Python Setup (CPU)

### 1. Create a virtual environment

```bash
virtualenv -p python3.9 faisscpu_env
source faisscpu_env/bin/activate
````

(On Windows)

```bash
virtualenv -p python3.9 faisscpu_env
faisscpu_env\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python clustering.py
```

This will generate a plot `clusters.png` showing the spatial clusters along with the csv file.

---

## R Script

An equivalent **R implementation** is also included (`clustering.R`) which performs the same pipeline.

Required R packages:

* `nabor`
* `igraph`
* `dplyr`
* `ggplot2`
* `fields` (for `quilt.plot` visualization)

The R script produces similar spatial cluster visualizations and can be used as a reference or for comparison.

---

## File Structure

```
contiguous_spatial_clusters/
│
├─ clustering.py       # Python CPU implementation
├─ clustering.R        # Equivalent R implementation
├─ requirements.txt    # Python dependencies
├─ README.md           # readme
```

---

## Usage Example

### Python

```python

!python clustering.py
```

### R

```r

Rscript clustering.R
```