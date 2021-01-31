# Predicting Music Genres Using Waveform Features

Can we predict music genres using features extracted from their waveforms? To answer that question, I used this [dataset](https://www.kaggle.com/insiyeah/musicfeatures) I found on Kaggle; the goal was to classify music using quantitative characteristics presented in the dataset, like *beats*, *spectral centroid* and other mathematical stratifications of songs. So, without further ado, let's get through the project!

## Dataset

The dataset is composed of waveform features extracted from 1000 audio tracks of 30 seconds each, labeled according to the music genre they belong to, from a total of 10 different categories: **blues**, **classical**, **country**, **disco**, **hip-hop**, **jazz**, **metal**, **pop**, **reggae**, and **rock**.

The set of features was extracted utilizing the [libROSA](https://librosa.org/doc/latest/index.html) library, by using Musical Information Retrieval techniques. The credits to this dataset go to [MARSYAS](http://marsyas.info/).

## Achieved Results

The best model was able to achieve an accuracy of **68.5%** on unseen data taken from the dataset. Given the low number of samples and the high number of classes (only 100 examples per music genre), this can be considered a good outcome.

## Requirements and Setup

To replicate this project, you'll need to download the files in this [GitHub repo](https://github.com/Pedrohgv/Music_Genre_Classification) and run the [jupyter notebook](https://github.com/Pedrohgv/Music_Genre_Classification/blob/master/notebook.ipynb) (which contains all the code if you want to check it), but not before installing the required packages (it is recommended to create a new virtual environment in order to avoid dependency conflicts). As I usually develop my projects in both Linux and Windows simultaneously, you can choose which one fits you best. If you are on Linux, open a terminal and type:

    pip install -r requirements-linux.txt

Or if you prefer Windows, use: 

    pip install -r requirements-windows.txt

## Feature Exploration and Engineering

The dataset was divided into a training set and a test set, so we could score our predictor later on data that was not used in the training process. All the data exploration will be done using the training data, as we keep our test set as *unseen data*.

Now, we will take a closer look at the features of the dataset. There are a total of 8 different numerical features, plus the *Mel Frequency Cepstral Coefficients* that are numbered from 1 to 20. We will first explore the first 8 features and then move into analyzing the *MFCC* coefficients.

The distribution of the features can be seen below:

<img src="images/feature distribution.png" alt="drawing" width="900" height="444"/>

As we can see by the plot above, the features are distributed in a fairly normal fashion with some minor deviations; a standardization process will be welcomed in order to turn the data more digestable to some of our models.
Now, let's see the correlation between our features:

<img src="images/feature correlation heatmap.png" alt="drawing" width="600" height="444"/>

This can give us some interesting insights about some of our columns; `spectral_bandwidth`, `rolloff` and `zero_crossing_rate` all share a high correlation with `spectral_centroid`, and `beats` is highly correlated with `tempo`. Therefore, we will keep only `tempo`, `chroma_stft`, `rmse` and `spectral_centroid` in our dataset, reducing the number of features in this step from 8 down to 4.

Now, we will investigate the features we're going to be working on. For that purpose, we'll plot the kernel density estimation across different target classes, as well as give a brief description of each feature. It is important to note that the vertical axis on the plots represents the density of the variable being plotted; this means that the scale will be influenced by the range of each variable.

### `tempo`
<details>
<summary>Click to expand</summary>

The `tempo` feature tells us the rithm of a song. The higher the value, the fastest the song plays.

<img src="images/tempo distribution.png" alt="drawing" width="900" height="444"/>

</details>

### `chroma_stft`
<details>
<summary>Click to expand</summary>

The `chroma_stft` feature represents the [Short-time Fourier transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) of each song. This feature is, however, reduced to a single value and the dataset doesn't include an explanation of what it represents (the complete Fourier transform would have a single value for every frequency, representing the contribution of that frequency to the entirety of the song). Given the range of values present in the dataset, we can assume they correspond to the proportion of the dominant frequency in each song.

<img src="images/chroma_stft distribution.png" alt="drawing" width="900" height="444"/>

</details>

### `rmse`
<details>
<summary>Click to expand</summary>

`rmse` stands for [root mean square energy](https://musicinformationretrieval.com/energy.html#:~:text=The%20root%2Dmean%2Dsquare%20energy,x%2C%20sr%20%3D%20librosa) and represents the total amount of energy in a given signal.

<img src="images/rmse distribution.png" alt="drawing" width="900" height="444"/>

</details>

### `spectral_centroid`
<details>
<summary>Click to expand</summary>

The `spectral_centroid` [property](https://en.wikipedia.org/wiki/Spectral_centroid) indicates the frequency that corresponds to the center of mass of a signal.

<img src="images/spectral_centroid distribution.png" alt="drawing" width="900" height="444"/>

</details>
&nbsp;

The plots above show us that all features have significant differences in distribution across the different target classes; thus, they are going to be useful in helping us build models to predict music genre.

### MFCC Coefficients

MFCCs stands for [mel-frequency cepstral coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) and collectively make up an MFC (mel-frequency cepstrum). In our dataset there are 20 MFCC's. 

After checking the one-dimensional features, we can now similarly analyze the MFCC coefficients by verifying how they are correlated to each other.

<img src="images/MFCC index correlation heatmap.png" alt="drawing" width="600" height="500"/>

By observing the plot above we can see a high correlation in groups of odd and even coefficients, excluding the first and second MFCC's. To reduce the dimensionality of this feature, we will apply the PCA method to reduce the numbers of both odd and even MFCC's. By decomposing both groups of odd and even indexes we can see how much redundant the information is; to measure this, we will use the explained variance of each component generated by the PCA transformation:

    Explained variance (on tranining set) of reduced odd MFCC's: [0.68552199 0.1823339 
    0.04741692 0.02504977 0.01711308 0.01659338 0.01124799 0.00869031 0.00603266]

    Explained variance (on tranining set) of reduced even MFCC's: [0.71564189 0.13864236 
    0.05718861 0.02319806 0.01853898 0.0147636 0.01390881 0.00991512 0.00820257]

Following the results above, we can conclude that the majority of the information (roughly 97%) found in groups of odd and even coefficients can be synthesized in the first 3 components; we will keep, however, 7 components from each group, reducing the total number of dimensions from 18 to 14 in order to keep 99% of the information from the features.

The next step is to standardize the columns of our dataset, so they can be more easily digested by some of our models. After that is done, we are finally ready to start training some models.

## Model Training and Optimization

We will train some machine learning models using the default parametrization, in a 5-fold cross-validation approach to scoring the models, according to their mean accuracy across the validation folds. The *Naive Bayes* model will be used as our baseline. Once we are done with that, the best models will be picked and fine-tuned so we can achieve the best results. Below are the accuracies achieved by each model:


| Model                              | Accuracy |
|:-----------------------------------|:---------|
| Naive Bayes                        |53.37%    |
| Logistic Regression                |59.00%    |
| K-nearest Neighbors                |55.75%    |
| Decision Tree                      |47.00%    |
| Random Forest                      |62.62%    |
| Support Vector Machine             |63.75%    |
| Gradient Boosting                  |61.12%    |
| Neural Network                     |63.12%    |

Of the models shown above, we will choose only 4 to further tunning: the **Random Forest**, the **Support Vector**, the **Gradient Boosting**, and the **Neural Network** classifiers, which achieved the best results. The strategy will be to first do a wide randomized search of parameters using the `RandomizedSeacrhCV` function and then use the `GridSearchCV` function to do a more narrow search for the best hyperparameters. The results of this process can be seen below:

| Model                   | Wide search best accuracy | Narrow search best accuracy |
|:------------------------|:--------------------------|:----------------------------|
| Random Forest           | 63.00%                    | 64.12%                      |       
| Support Vector Machine  | 64.75%                    | **65.50%**                  |
| Gradient Boosting       | 63.00%                    | 64.00%                      |
| Neural Network          | 64.75%                    | 64.87%                      |

According to the results above, the model that achieved the highest accuracy after hyperparameter tunning was the **Support Vector Classifier**, and it will be chosen as the best model for our problem. We can finally use our test set to verify how well it can predict the genres of music. After scoring the model on our test set, the classifier obtained a score of **68.5%**.

## Final Conclusions

Despite the best scoring model achieving an accuracy below 70%, it is still remarkable that such a small dataset (that contains only waveform information) with only 100 samples per class and a total of 10 different classes was enough to allow a classification model to correctly predict music genres more than 65% of the time. Maybe if we had more samples or even more features, we could achieve even better results.