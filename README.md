# Data Science Portfolio - Mohamed Sabbah  

This portfolio contains all my data science projects from both my academic and personal work. It includes projects from different domains, such as Natural Language Processing (NLP), Machine Learning, and Deep Learning, each organized under their respective sections.  

## Contact  
- 📧 Email: sabbah.2056136@studenti.uniroma1.it  
- 🔗 [LinkedIn](https://www.linkedin.com/in/mohamed-sabbah-ab601bab/)  

## Content  

### Machine Learning  

- **[Census Data Income Prediction](https://github.com/masabbah-97/Data-Science-Portfolio/blob/main/Machine%20Learning%20Projects/Census-Data-Income-Prediction/Census%20Data%20Income%20Prediction.ipynb)**  
  - **Objective:** Developed a machine learning model to predict whether an adult earns more than $50K per year based on U.S. census data.  
  - **Approach:**  
    - Explored demographic and economic features (age, education, occupation, etc.).  
    - Trained multiple classifiers (Logistic Regression, Support Vector Machine, Gradient Boosting, K-Nearest Neighbors, Decision Tree, Random Forest, and Neural Network). 
    - Applied cross-validation and hyperparameter tuning to optimize performance.  
  - **Results:**  
    - Achieved an **accuracy of 88%** with Gradient Boosting, outperforming other models.  
    - Identified key income-influencing factors such as education level and occupation.  
  - **Tools & Technologies:** `Python, Scikit-Learn, Keras, Matplotlib, Plotly, XGBoost`
  - [Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income/)

### Statistical Analysis  

- **[Insurance Charge Statistical Analysis](https://github.com/masabbah-97/Data-Science-Portfolio/blob/main/Statistical%20Analysis%20Projects/Insurance-Charge-Statistical-Analysis/Insurance-Charge-Statistical-Analysis.pdf)**  
  - **Objective:** Developed models to predict individual insurance charges based on demographic and health-related features.  
  - **Approach:**  
    - Explored demographic features (age, sex, region, etc.) and health-related features (BMI, smoking, etc.).  
    - Applied **Frequentist** and **Bayesian** approaches for model creation, models were a standard linear regression and one with polynomial features.
    - In the **Frequentist Approach**, models were evaluated using **Root Mean Square Error (RMSE)** to assess prediction accuracy.  
    - In the **Bayesian Approach**, model evaluation included:
      - **Deviance Information Criterion (DIC)** for model fit comparison.  
      - **Potential Scale Reduction Factor (R̂)** to check MCMC convergence.  
      - **Effective Sample Size (n.eff.)** to measure independence in samples.  
      - **RMSE** for model accuracy.  
  - **Results:**  
    - Both Frequentist and Bayesian models performed well, with polynomial features enhancing model performance.  
    - The model with polynomial features provided lower **DIC** and better **RMSE** scores.  
    - Identified significant variables like **age**, **BMI**, and **smoking** as key predictors of insurance charges.  
  - **Tools & Technologies:** `R, JAGS, caret, ggplot2, R2jags, dplyr`
  - [Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

### Deep Learning  

- **[Saliency Map and SmoothGrad on CNN](https://github.com/masabbah-97/Data-Science-Portfolio/blob/main/Deep%20Learning%20Projects/Saliency-Map-and-SmoothGrad-on-CNN/Saliency_map_implementation.ipynb)**  
  - **Objective:** Implemented a saliency map and SmoothGrad on a Convolutional Neural Network (CNN) using the "Cats vs Dogs" dataset from Kaggle.  
  - **Approach:**  
    - Built a CNN to classify images of cats vs. dogs.  
    - Implemented the **saliency map** and **SmoothGrad** techniques by hand to visualize the areas of the image that influence the model’s predictions the most.  
    - The dataset was retrieved from Kaggle.  
  - **Results:**  
    - Successfully visualized model predictions, enhancing interpretability of the CNN’s decision-making process.  
    - Demonstrated the effectiveness of saliency maps and SmoothGrad for understanding CNN behavior on image classification tasks.  
  - **Tools & Technologies:** `TensorFlow, Keras, Python`  
  - [Dataset](https://www.kaggle.com/competitions/dogs-vs-cats/)

### [Thesis](https://github.com/masabbah-97/Data-Science-Portfolio/tree/main/Thesis)

