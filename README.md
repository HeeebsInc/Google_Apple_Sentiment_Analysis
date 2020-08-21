# Twitter_Sentiment_Analysis

![CoverPhoto](figures/TwitterUnsplash.jpg)

## Contributers 
- Samuel Mohebban
  - Samuel.MohebbanJob@gmail.com
  - [LinkedIn](https://www.linkedin.com/in/samuel-mohebban-b50732139/) 
  - [Medium](https://medium.com/@HeeebsInc)
- Raven Welch
  - email 
  - [LinkedIn](https://www.linkedin.com/in/raven-welch/)
  - [Medium]()

[Google Slide Presentation](https://docs.google.com/presentation/d/15voSS3ctPLzh_cXnql9N3kaXL4PJ-3-6xRGRzFsgyzc/edit?usp=sharing)
### Problem 
- Ideally, customers are always satisfied with a company's services, however, this is rarely the case 
- Twitter is a cornerstrone for communication as many customers share their issues on the platform.  For customers, Twitter is accessible and can offer ways for a customer to directly communicate with larger companies such as Google and Apple
- Manually screening each tweet can be costly and time consuming 

### Solution 
- We wanted to create a Machine Learning algorithm that can correctly detect sentiment within tweets.  
- By accomplishing this, a company can better screen customer feedback and attend to any issues they may be having 

### Data 
- [Brands and Product Emotion Dataset](https://data.world/crowdflower/brands-and-product-emotions)
- Comprised of 9000 tweets regarding sentiment towards Apple and Google Products 
- Gathered in August 2013
- Removed 'No Emotion' and 'Unknown' 
- Data was pruned to 600 Positive Tweets and 570 Negative Tweets so that our final dataset had an equal class balance for each target

![ClassImbalance](figures/ClassImbalance.png)

### Data Cleaning 
1. Cleaned the data to remove the following: 
    - names
    - usernames
    - non-alphabetical characters
    - and stop words such as “so”, “a”, and “we”
2. Stemmed each word within our data
    - Ex: “Likes”, “Liked”, “Likely”, “Liking” all become “Like”
3. Applied KMeans clustering with n_clusters = 6 
4. Split into Training and Test sets for modeling 
      - **Train** → 994 (50% Positive, 50% Negative) 
      - **Test** → 176 (50% Positive, 50% Negative)
- [code](https://github.com/HeeebsInc/Google_Apple_Sentiment_Analysis/blob/master/Functions.py#L148)

### KMeans Clustering 


![KMeans](figures/KMeans.png)

![PCA_Cluster-Gif](figures/MyVideo_122.gif)












![TwitterBird](figures/BirdTwitter.jpg)

