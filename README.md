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

### Thesis
- **[Leveraging Large Language Models for Developing Subjective Well-being Indicators](https://github.com/masabbah-97/Data-Science-Portfolio/tree/main/Thesis)**
  - **Objective:** Develop a subjective well-being indicator based on the mental health of communities by utilizing big data resources and LLMs.
  - **Approach:**
    - By utilizing a [mental health dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health), 3 models were trained on the classification task. The models are a basic deep learning model (Attention-based BiLSTM), a bidirectional LLM (BERT), and a more modern LLM that is much larger (Meta's Llama 3.2 3B).
    - Use three models (Llama, BERT, and an Attention-Based BiLSTM) and train them on the task of classifying the mental health status according to text data.
    - Use another dataset (source: Istat) for the first 5 months of 2020 to create well-being indicators related to mental health. The models are given these tweets as input and a time series was created for each class.
    - The results were analyzed using the previously mentioned graph, a confidence graph, and word clouds and trigram graphs to further analyze the trends.
  - **Results:**
    - The model that generalized the best when creating the indicators was the BERT model, while the Llama is second (with some cases of overfitting for specific classes), and the worst model is the BiLSTM as it shows obvious signs of underfitting.
    - The LLMs managed to capture trends that were in line with real-life events (the onset of the pandemic).
    - There were obvious spikes in negative classes when the lockdown measures were first introduced, and upon further analysis it can be deduced that the panic buying phenomenon as well as having to isolate was taking a toll on Italian communities.
    - LLMs can be a very valuable tool for gaining real-time insights about the public, allowing for quicker interventions to improve overall well-being and respond to worrying trends faster than traditional methods.
  - **Tools & Technologies:** `TensorFlow, Keras, HuggingFace, PEFT, NLTK, Python`

