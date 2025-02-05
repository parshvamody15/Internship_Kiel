# Kiel Climate Model: Internship at Kiel University

## Mathematics and Computer Science Department

## Table of Contents

- [Objective](#Objective)
- [Tasks Performed](#Tasks-Performed)
  - [Data Preparation](#Data-Preparation)
  - [Data Reshaping and Reduction](#Data-Reshaping-and-Reduction)
  - [Transformation and Matrix Construction](#Transformation-and-Matrix-Construction)
  - [Data Cleaning and Regularization](#Data-Cleaning-and-Regularization)
  - [Singular Value Decomposition (SVD)](#Singular-Value-Decomposition-SVD)
  - [Choosing the Number of Singular Values Required](#Choosing-the-Number-of-Singular-Values-Required)
  - [Plotting the Chosen Singular Values](#Plotting-the-Chosen-Singular-Values)
  - [Data Reconstruction](#Data-Reconstruction)
  - [Reshaping and Comparison](#Reshaping-and-Comparison)
  - [Model Testing on Test Dataset (`Parameter 00000`)](#Model-Testing-on-Test-Dataset-`Parameter-00000`)
  - [Error Analysis](#Error-Analysis)
- [Inference](#Inference)
- [Conclusion](#Conclusion)


## Objective

During my two-month internship at Kiel University, I worked in the Mathematics and Computer Science department to perform statistical data analysis tasks using Python. The primary goal was to gain practical experience in data analysis and apply my skills to real-world problems. My main objective was to successfully reconstruct marine ecosystem data using singular value decomposition (SVD).

## Tasks Performed

### Data Preparation
- Read `.petsc` files from the folders `Parameter 00000` to `Parameter 00099` using an optimized method.
- Used the `read_PETSc_vec` function for file reading, streamlining data acquisition.
- Read the `landSeaMask.petsc` file using the `read_PETSc_matrix` function for further analysis.

### Data Reshaping and Reduction
- Reshaped all datasets into a 3D vector using the `reshape_vector_to_3d` function, utilizing `landSeaMask.petsc` as a reference.
- Extracted relevant data from the upper surface after reshaping.

### Transformation and Matrix Construction
- Transformed reduced data into a 1D vector for further processing.
- Constructed a matrix to store all datasets (excluding `Parameter 00000`, which was used as the test dataset).

### Data Cleaning and Regularization
- Replaced NaN values with `0` in both the matrix and the test data.
- Calculated the mean value of the dataset and subtracted it to normalize the data.

### Singular Value Decomposition (SVD)
- Used `np.linalg.svd()` to perform SVD on the matrix (excluding `Parameter 00000`).
- Plotted the singular values for analysis.
- Calculated the percentage of data representation by dividing the sum of the squares of selected singular values by the sum of the squares of all singular values.

### Choosing the Number of Singular Values Required
- After performing the Singular Value Decomposition, I plotted the curve for the singular value for every dataset. This plot helped me determine the number of singular values required to represent the data accurately.
- I also calculated the percentage of data that could be represented by dividing the sum of the squares of the chosen values by the sum of the squares of all singular values.

### Plotting the Chosen Singular Values
- After selecting the required Singular Values, I plotted them as a world map. Below is an example of the plot of a Singular Value.

### Data Reconstruction
- Reconstructed all datasets using selected singular values (truncated) from the SVD calculation.
- Used truncated left and right singular matrices and computed their dot product with the selected singular values.

### Reshaping and Comparison
- Reshaped the reconstructed and training datasets into 2D fields for visualization.
- Plotted both fields, replacing zero values with NaN for better clarity.
- Compared the original and reconstructed training datasets to assess accuracy.

### Model Testing on Test Dataset (`Parameter 00000`)
- Found coefficients by computing the dot product between the test dataset and the truncated right singular vector values.
- Multiplied coefficients with the dataset and summed them to reconstruct the test dataset.
- Plotted original and reconstructed test datasets for visualization.

### Error Analysis
- Calculated percentage error between the original and reconstructed datasets.
- Used `np.linalg.norm()` to compute L2 norm differences.
- Determined the percentage difference in L2 norm between the reconstructed and original test datasets.

## Inference
- At least **4 singular values** are required for effective reconstruction of training datasets, but for higher accuracy, up to **6 singular values** can be chosen.
- A **minimum of 15 to 16 training datasets** is necessary for extracting singular values to reconstruct the test dataset with high fidelity.
- The **L2 norm difference** between the original and reconstructed training dataset falls within **[0, 3)**.
- The percentage difference in **L2 norm for test datasets** is within **[0, 1.5)**.

## Conclusion

The internship at Kiel University provided valuable hands-on experience in statistical data analysis using Python. I successfully completed tasks such as:

✔ Efficient PETSc file reading  
✔ Data preparation, reshaping, reduction, and transformation  
✔ Matrix construction and regularization  
✔ Singular Value Decomposition (SVD)  
✔ Data reconstruction and visualization  
✔ Error analysis  

I extend my heartfelt gratitude to **Professor Dr. Thomas Slawig** for this opportunity and to the Mathematics and Computer Science department team for their guidance and support. This experience has significantly contributed to my professional growth.
