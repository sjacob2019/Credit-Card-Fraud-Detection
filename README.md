# CS 7641 Team 1 - Credit Card Fraud Detection
Joshu Hsu, Shaun Jacob, Andrew Novokshanov, Wanli Qian, Tongdi Zhou

## Introduction 

Every year the recorded number of fraudulent incidents increases worldwide. In 2020 the Federal Trade Commission reported nearly 2.2 million such cases in the United States alone [^fn1]. As technology continues to develop at blistering speeds, so do the ways that criminals steal information. As a result, the technology we use to deal with these problems must also evolve. The goal of our project is to distinguish incidents of fraudulent credit card use from those that are legitimate.



## Problem Definition

Our dataset was found on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud), and some of the main characteristics are that the dataset is comprised of principle components from PCA. The dataset contains real-world credit card transactions from a period of only two days by European card holders in September 2013. Due to this, there is a large number of legitimate transactions compared to fraudulent transactions. To combat this imbalance, our dataset comes with a data simulator to augment the number of fraudulent transactions. 



## Supervised Learning Method

For this method, we will use a publicly available credit card fraud data simulator to obtain simulated data for credit card transactions. We will then train an Artificial Neural Network to classify the data points as fraud or not fraud. Neural Networks work best when we have large amounts of data, often outperforming traditional machine learning algorithms [^fn2]. Since we can use the simulator to generate as much data as we want, using a Neural Network will give us more accurate results. A factor that comes into play in the success of our algorithm is domain knowledge, which can be used to extract useful features from raw data and is used in traditional machine learning algorithms. Another advantage of Neural Networks is that they also work well when there is a general lack of domain knowledge. This is because they learn the high-level features of the dataset in an incremental fashion [^fn2]. Due to that, we don’t have to worry about feature engineering and domain knowledge.  



## Unsupervised Learning Method

For our particular dataset, unsupervised learning becomes particularly useful. Because of confidentiality issues, the original data and labels are not provided. our dataset only contains numerical input variables which are the result of a PCA transformation. Although we have simulator data at our disposal, it is still inconvenient to integrate our model with other real world datasets, because they are more likely to be unlabeled as well. Yet, by applying clustering algorithm such as GMM(Gaussian Mixture Models), we could circumvent this problem. 

Many Prior works has shown that, using GMM in conjunction with PCA is effective. (Hertrich, 2021) proposed a Gaussian Mixture Model in conjunction with a reduction of the dimensionality of the data in each component of the model by principal component analysis [^fn3]. And (Alqahatani, 2020) used a bivariate mixture model and learning with a bivariate Gaussian mixture model [^fn4].



## Potential Results and Discussion 

### Cross-validation 

A common challenge for training classification models is how to avoid over-fitting. Over-fitting happens when the model performs well on the data which it was trained on but not on the data it has yet seen[^fn5]. For example, this can happen in clustering algorithms when the number of clusters is set too large.  

  

To overcome this challenge, we evaluate the performance of our model using cross-validation where the dataset is partitioned into two complementary sets: one for training and one for validating. There will be multiple rounds of partitioning, and we take the average accuracy on the validating data as the overall performance [^fn6].  

  

### Confusion Matrix 

To evaluate our final results, we will use a confusion matrix where each row is an instance of the ground truth, and each column is an instance of the prediction [^fn7]. Our goal is to maximize the detection rate while minimizing the false positive rate. 

  

| Prediction  | Genuine             | Fraudulent                     | 
|-------------|---------------------|--------------------------------| 
|  Genuine    | True Negative Rate  | False Positive Rate            | 
|  Fraudulent | False Negative Rate | True Positive (Detection) Rate | 






Sources:

[^fn1]: Federal Trade Commission. (2021, February). Consumer Sentinel Network Data book 2020. Consumer Sentinel Network. Retrieved October 2, 2021, from https://www.ftc.gov/system/files/documents/reports/consumer-sentinel-network-data-book-2020/csn_annual_data_book_2020.pdf. 

[^fn2]: Mahapatra, S. (2019, January 22). Why deep learning over traditional machine learning? Medium. Retrieved October 1, 2021, from https://towardsdatascience.com/why-deep-learning-is-needed-over-traditional-machine-learning-1b6a99177063. 

[^fn3]: Hertrich, J., Nguyen, D., Aujol, J., Bernard, D., Berthoumieu, Y., Saadaldin, A., & Steidl, G. (2021). PCA reduced Gaussian mixture models with applications in superresolution. Inverse Problems & Imaging, 0(0), 0. doi:10.3934/ipi.2021053 

[^fn4]: Alqahtani, N. A., & Kalantan, Z. I. (2020). Gaussian Mixture Models Based on Principal Components and Applications. Mathematical Problems in Engineering, 2020, 1-13. doi:10.1155/2020/1202307

[^fn5]: Cawley, G. C., & Talbot, N. L. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. The Journal of Machine Learning Research, 11, 2079-2107. 

[^fn6]: Stone, M. (1974). Cross‐validatory choice and assessment of statistical predictions. Journal of the royal statistical society: Series B (Methodological), 36(2), 111-133. 

[^fn7]: Fawcett, T. (2006). An introduction to ROC analysis. Pattern recognition letters, 27(8), 861-874.

