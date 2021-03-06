# CS 7641 Team 1 - Credit Card Fraud Detection
Joshua Hsu, Shaun Jacob, Andrew Novokshanov, Wanli Qian, Tongdi Zhou

## Introduction 

Every year the recorded number of fraudulent incidents increases worldwide. In 2020, the Federal Trade Commission reported nearly 2.2 million such cases in the United States alone [^fn1]. As technology continues to develop at blistering speeds, so do the ways that criminals steal information. As a result, the technology we use to deal with these problems must also evolve. The goal of our project is to distinguish incidents of fraudulent credit card use from those that are legitimate.

<img src="https://www.ftc.gov/sites/default/files/u544718/explore-data-consumer-sentinel-jan-2021.jpg" width="350"/>

*Image courtesy of https://www.ftc.gov/reports/consumer-sentinel-network-data-book-2020*


## Problem Definition

Our dataset was found on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud), and one of the main characteristics is that the dataset is primarily composed of principle components from PCA. The dataset contains real-world credit card transactions from a period of only two days by European card holders in September 2013. Due to this, there is a large number of legitimate transactions compared to fraudulent transactions. To combat this imbalance, our dataset comes with a data simulator to augment the number of fraudulent transactions.



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

