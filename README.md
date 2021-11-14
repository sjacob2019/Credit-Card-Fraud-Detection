# CS 7641 Team 1 - Credit Card Fraud Detection
Joshua Hsu, Shaun Jacob, Andrew Novokshanov, Wanli Qian, Tongdi Zhou

## Introduction 

Every year the recorded number of fraudulent incidents increases worldwide. In 2020, the Federal Trade Commission reported nearly 2.2 million such cases in the United States alone [^fn1]. As technology continues to develop at blistering speeds, so do the ways that criminals steal information. As a result, the technology we use to deal with these problems must also evolve. The goal of our project is to distinguish incidents of fraudulent credit card use from those that are legitimate.

<img src="https://www.ftc.gov/sites/default/files/u544718/explore-data-consumer-sentinel-jan-2021.jpg" width="350"/>

*Image courtesy of https://www.ftc.gov/reports/consumer-sentinel-network-data-book-2020*


## Problem Definition

By taking advantage of Machine Learning we can develop models to determine the legitimacy of a transaction. 

The primary premise of our project is to create Machine Learning models that can accurately predict whether a transaction is fraudulent or legitimate. In other words, when a new credit card transaction is attempted, we will be able to determine with a high degree of certainty whether the transaction should be labeled as fraudulent. 

We will be analyzing the problem at hand using supervised and unsupervised learning methods. Since both supervised and unsupervised methods can contribute to our problem analysis in unique ways, especially with respect to the high level of imbalance our dataset exhibits. 

Supervised learning models are commonly attributed to working well on classification problems. However, supervised learning requires labels, and we do not always have labels provided to us in a real-world scenario. In this circumstance, we have decided to also use unsupervised learning models. Unsupervised models do not require labels, and although unsupervised learning models are not commonly used for classification problems, they are useful for determining clustering, association, and performing dimensionality reduction. Unsupervised learning will help us determine what factors are relevant when analyzing the features of our data. 

Supervised learning models do have a limitation though – they are typically more susceptible to outliers and data imbalances, a prominent issue in our data. To alleviate the class imbalance, we propose to combine the under-sampling technique as well as the SMOTE algorithm to decrease the amount of data in the majority class (legitimate transactions) and increase the amount of data in the minority class (fraudulent transactions). 

Through a combination of these learning techniques, we hope to develop a holistic analysis of how various machine learning algorithms handle the classification of the legitimacy of credit card transactions. 

## Data Collection

Fortunately, the problem we are asking has been researched before and there are datasets available for us to use online. We ultimately decided to work with a dataset found on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud), for the reasons that it was composed of real-life data samples and had 284,807 transactions listed. These transactions came from a period of two days by European card holders in September 2013. 

Although the data being grouped by a similar time range helps us analyze similar data, it also means that we have severely unbalanced data. Of the 284,807 transactions in the dataset, only 492 are fraud (0.173%). 

Moreover, due to the sensitive nature of using real life transaction data, much of the identifying data had to be anonymized. Most of the original features were not provided, and instead we were given twenty-eight principal components obtained using PCA. We were also provided the original labels for the datasets to use for testing accuracy and for supervised learning.

## PCA Dataset Preprocessing and Visualization

![plot](./Images-MidTerm/Preprocess/preprocess1.jpeg)

Figure 1 shows the scatter plot of three PCA components with the highest variances. We can see the data set has a significant imbalance with the genuine class overwhelming the fraud class. We also notice a large overlap between the two classes of data, which brings challenges to the classification tasks.  

![plot](./Images-MidTerm/Preprocess/preprocess2.jpeg)

One way to combat class imbalance is under-sampling the majority class. We under-sample the genuine class by randomly retaining 10% of the genuine transactions. The resulting scatter plot is shown in Figure 2. A drawback of under-sampling is some information about the majority class may be lost. As a result, the under-sampling ratio cannot be too low. 

![plot](./Images-MidTerm/Preprocess/preprocess3.jpeg)

To further alleviate class imbalance, we use Synthetic Minority Over-sampling Technique (SMOTE) to increase the number of samples in the minority class. SMOTE synthesizes samples feature-by-feature: for each feature in an original sample, SMOTE finds its k-nearest neighbors in the feature space, chooses one neighbor at random, and picks a point at random along the line segment between the sample and its neighbor as the synthesized new sample. Figure 3 shows the result of SMOTE where we generate nine new samples from each original sample using its 10-nearest neighbors.  

 

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, 321-357. 

![plot](./Images-MidTerm/Preprocess/preprocess4.png)

Figure 4 shows the cumulative total variance of the PCA components for the original data and the data generated by SMOTE. As we can see, for both sets of data, the first 7 PCA components capture over 50% of the total variance, while the first 16 PCA components capture over 80% of the total variance. This information can guide the number of PCA components required for training classification models. We expect the first 16 PCA components to contribute the most to our models, while the rest of the PCA components may only bring negligible improvement.  

![plot](./Images-MidTerm/Preprocess/preprocess5.png)

Figure 5 is the scatterplot matrix showing the relationship between the first 7 PCA components, where the red dots represent fraudulent transactions and the blue dots are genuine transactions.  We can see although no two features can perfectly separate the two classes, each combination of the features can separate some parts of the data, which will be exploited by our models to obtain better classification performance. We also observe that the boundaries between the two classes are highly nonlinear. As a result, we expect machine learning techniques that produce nonlinear decision boundaries will have better performance in classifying the two classes.  

## Supervised Learning Method

For this method we will be training a Neural Network to classify the data points as fraudulent or not fraudulent. Neural Networks work best when we have large amounts of data, often outperforming traditional machine learning algorithms [^fn2]. Since we can use the simulator to generate as much data as we want, using a Neural Network will give us more accurate results. A factor that comes into play in the success of our algorithm is domain knowledge, which in traditional machine learning algorithms is used to identify features in order to reduce the complexity of the raw data and make patterns clearer to the algorithm. Another advantage of Neural Networks is that they also work well when there is a general lack of domain knowledge. This is because they learn the high-level features of the dataset in an incremental fashion [^fn2]. Due to that, we don’t have to worry about feature engineering or domain knowledge.



## Unsupervised Learning Method

For our particular dataset, unsupervised learning becomes particularly useful. Because of confidentiality issues, the original data and labels are not provided. Our dataset only contains numerical input variables that are the result of a PCA transformation. Although we have simulated data, it is still inconvenient to integrate our models with real-world datasets because they are unlikely to be labeled. However, by applying clustering algorithm such as GMM(Gaussian Mixture Models) we could circumvent this problem. 

Prior work has shown that using GMM in conjunction with PCA is effective in reducing the complexity from high-dimensional data. Johannes Hertrich from TU Berlin proposed a Novel Gaussian Mixture Model with a PCA dimensionality reduction module in each component of the model [^fn3]. Additionally, Nada Alqahtani proposed a bivariate mixture model and showed that learning with a bivariate Gaussian mixture model is able to decrease the complexity brought by the high dimensionality of the input data[^fn4].



## Potential Results and Discussion 

### Cross-validation 

Over-fitting happens when the model performs well on the training data but not on the testing data[^fn5]. To avoid over-fitting, we use cross-validation where the training data is partitioned into two complementary sets: one for training and one for validating. The overall performance is evaluated on the average accuracy of multiple rounds of partitioning[^fn6].
  
  <img src="https://scikit-learn.org/stable/_images/grid_search_cross_validation.png" width="350"/>
  
  *K-Fold (K=5) Cross Validation. Image courtesy of https://scikit-learn.org/stable/modules/cross_validation.html*

### Confusion Matrix 

The final results are shown by a confusion matrix where each row is ground truth, and each column is a prediction[^fn7]. Our goal is to maximize the detection rate while minimizing the false positive rate. 


| Prediction  | Genuine             | Fraudulent                     | 
|-------------|---------------------|--------------------------------| 
|  Genuine    | True Negative Rate  | False Positive Rate            | 
|  Fraudulent | False Negative Rate | True Positive (Detection) Rate | 



## Timeline

At the current time, all members of the team intend to participate in all tasks equally.

Week 7:   Publish Proposal

Week 8:   Data Generation

Week 9:   Pre Processing

Week 10:  Implement Unsupervised Method (GMM)

Week 12:  Results Analysis

Week 13:  Project Midpoint Report

Week 14:  Implement Supervised Method (Neural Networks)

Week 15:  Hyperparameter Tuning / Explore other Architectures

Week 16:  Organize Results and Write Conclusion

### Sources:

[^fn1]: Federal Trade Commission. (2021, February). Consumer Sentinel Network Data book 2020. Consumer Sentinel Network. Retrieved October 2, 2021, from https://www.ftc.gov/system/files/documents/reports/consumer-sentinel-network-data-book-2020/csn_annual_data_book_2020.pdf. 

[^fn2]: Mahapatra, S. (2019, January 22). Why deep learning over traditional machine learning? Medium. Retrieved October 1, 2021, from https://towardsdatascience.com/why-deep-learning-is-needed-over-traditional-machine-learning-1b6a99177063. 

[^fn3]: Hertrich, J., Nguyen, D., Aujol, J., Bernard, D., Berthoumieu, Y., Saadaldin, A., & Steidl, G. (2021). PCA reduced Gaussian mixture models with applications in superresolution. Inverse Problems & Imaging, 0(0), 0. doi:10.3934/ipi.2021053 

[^fn4]: Alqahtani, N. A., & Kalantan, Z. I. (2020). Gaussian Mixture Models Based on Principal Components and Applications. Mathematical Problems in Engineering, 2020, 1-13. doi:10.1155/2020/1202307

[^fn5]: Cawley, G. C., & Talbot, N. L. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. The Journal of Machine Learning Research, 11, 2079-2107. 

[^fn6]: Stone, M. (1974). Cross‐validatory choice and assessment of statistical predictions. Journal of the royal statistical society: Series B (Methodological), 36(2), 111-133. 

[^fn7]: Fawcett, T. (2006). An introduction to ROC analysis. Pattern recognition letters, 27(8), 861-874.

