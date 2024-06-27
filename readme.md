# Intelliwrite Blogging Application: Matrix Factorization(SVD)

## Introduction

My name is Chhatra Rana, and I am currently working on my Final Year Project (FYP) at College CG Institute of Management, affiliated with Limkokwing University of Creative Technology.

## Project Overview

The goal of this project is to build a user-friendly blogging platform that recommends personalized content to users. By integrating advanced techniques such as matrix factorization, the platform will predict user preferences based on their interaction history. This approach ensures that users receive tailored blog recommendations that match their interests and browsing behavior.

## Technologies Used

### Fullstack: Next.js


### Microservices: Python Microservices

 microservices drive the recommendation system. These microservices perform the following tasks:

- **Model Training**: Using matrix factorization and LSTM techniques to analyze user behavior and predict preferences.
- **Prediction**: Generating personalized blog recommendations based on user interactions and historical data.


## Project Goals

- Develop a user-friendly blogging platform that recommends relevant content based on user interests.
- Implement advanced algorithms for accurate prediction of user preferences.
- Enhance user engagement through personalized blog recommendations.

## Steps to Build Application

### Installation

Make sure Surprise library is installed in your Python environment:

```python
pip install scikit-surprise
pip install pandas
pip install numpy

```

## Imports and Components Explanation

### Blog Data Preprocessing Steps

- ### Importing Libraries

    ```python
    import pandas as _pd
    import numpy as _np
    ```
- ### Read CSV files
    ```python
    _authorData = _pd.read_csv('author_data.csv')
    _blogData = _pd.read_csv('blog_data.csv')
    _mediumBlogData = _pd.read_csv('medium_blog_data.csv')
    ```

- ### Display first few rows of each dataset
    ```python
    _authorData.head()
    _blogData.head()
    _mediumBlogData.head()
    ```
- ### Data Cleaning
    Drop unnecessary columns from _mediumBlogData
    ```python 
        _mediumBlogData.drop(["blog_title","blog_content","blog_img","blog_link","scrape_time"], axis=1, inplace=True)
        _mediumBlogData.shape
        _mediumBlogData.head()
    ```
- ### Final Data Integration
    Merge _dataWithAuthor with _blogData on 'blog_id'
```python
    _fullBlogData = _pd.merge(_dataWithAuthor, _blogData, on="blog_id")
    _fullBlogData.shape
    _fullBlogData.head()
```
- ### Feature Engineering 
    Create a new 'features' column combining 'topic', 'author_name', and 'ratings'
    ```python
    _fullBlogData["features"] = _fullBlogData["topic"] + ',' + _fullBlogData["author_name"] + ',' + _fullBlogData["ratings"].astype(str)
    _fullBlogData.shape
    _fullBlogData.head()
    ```

- ### Data Analysis
    - #### Number of ratings per user
        ```python
            ratings_per_user = _fullBlogData.groupby('userId').size().reset_index(name='num_ratings')
            print(ratings_per_user)
        ```

    - #### Number of ratings per blog per user
        ```python
            rating_counts = _fullBlogData.groupby(['userId', 'blog_id']).size().reset_index(name='ratings')
            print(rating_counts)
        ```

### Surprise and SVD Implementation
- ### Reader
    It is responsible for parsing a file or DataFrame to convert it into a format that Surprise can use.
    ```python
    from surprise import Reader
    reader = Reader(rating_scale=(0.5,5.0))
    ```
- ### Dataset
    It represents the entire dataset, typically loaded from a file or dataframe using `Reader`.
    ```python
    from surprise import Dataset
    data = Dataset.load_from_df(df[['userId', 'blog_id', 'rating']], reader)
    ```
- ### train_test_split()
    It is responsible for spiliting the dataset into training and testing sets. Splitting data into training and testing sets, we can evaluate how well the recommendation model performs on unseen data.
    ```python
    from surprise.model_selection import train_test_split
    trainset, testset = train_test_split(data, test_size=0.2)
    ```
    `test_size` refers to the proportion of the dataset that will be allocated fo the test set while splitting data into training ans testing sets.
    In above case splitting data into _Training_ `80%` and _Testing_ `20%`

- ### SVD
    SVD(singular Value Decomposition) is an matrix factorization technique commonly used in recommendation system to analyze and predict user-item interactions.

    - #### Matrix Representation
        In above case we have a user-blog matrix where each entry represents a user's rating with an item. Suppose we have matrix _R_ where _Rub_ denotes the rating given by user _u_ to blog _b_.
    - #### Objective

        The goal of SVD is to break down a user-item interaction matrix into smaller matrices that capture underlying patterns. This helps predict missing ratings (recommendations) accurately.

    - #### SVD Components

        - **U**: User-feature matrix where each row represents a user's association with latent factors.
        - **Σ**: Diagonal matrix containing singular values, which indicate the importance of each latent factor.
        - **V**: Item-feature matrix where each row represents an item's association with latent factors.

### Matrix Factorization

SVD decomposes the user-item matrix \( R \) into three matrices:
\[ R \approx U \Sigma V^T \]
This approximation helps represent \( R \) using fewer dimensions (latent factors).

### Prediction

To predict how user \( u \) would rate item \( i \):
\[ \hat{r}_{ui} = \langle U_u, V_i \rangle \]
- \( U_u \): Latent factor vector for user \( u \)
- \( V_i \): Latent factor vector for item \( i \)
- \( \langle \cdot, \cdot \rangle \): Dot product of vectors

### Algorithm (SVD) in Surprise

```python
   from surprise import SVD
    algo = SVD()
    algo.fit(trainset)
```
- **Training**: It learns latent factors \( U \) and \( V \) from training data (`trainset`). These factors are optimized to minimize the difference between predicted and actual ratings.


## Example Code

We have already trained an SVD model (`algo`) on a dataset (`trainset`), here's how we can recommend items for a specific user (`userId = 1`):

```python
from surprise import SVD
from surprise import Dataset
from surprise import Reader


# Recommend items for a specific user (userId = 1)
userId = 1
items_to_predict = _fullBlogData['blog_id'].unique() 
# Above my case _fullBlogData is my dataset
predictions = [algo.predict(userId, blog_id) for blog_id in items_to_predict]

# Sorting predictions by estimated rating (descending order)
predictions.sort(key=lambda x: x.est, reverse=True)

# Displaying top recommended items
top_n = 10  # Number of recommendations to display
print(f"Top {top_n} Recommendations for User {userId}:")
for i, prediction in enumerate(predictions[:top_n], 1):
    print(f"Rank {i}: Blog ID {prediction.iid} (Estimated Rating: {prediction.est})")
```

### Recommendation Based on Topic

```python
selected_topic = 'ai'

# Step 2: Generate predictions for items related to the selected topic
topic_related_items = _fullBlogData[_fullBlogData['topic'] == selected_topic]['blog_id'].unique()

predictions = []
for blog_id in topic_related_items:
    prediction = algo.predict(userId, blog_id)  
    predictions.append((blog_id, prediction.est))

# Step 3: Sort predictions based on estimated ratings (est) in descending order
predictions.sort(key=lambda x: x[1], reverse=True)

# Display top recommended items
top_n = 5 

print(f"Top {top_n} Recommendations for User {userId} based on topic '{selected_topic}':")

for i, (blog_id, estimated_rating) in enumerate(predictions[:top_n], 1):
    blog_info = _fullBlogData[_fullBlogData['blog_id'] == blog_id].iloc[0]
    print(f"Rank {i}: Blog ID {blog_id} (Topic: {blog_info['topic']}, Author: {blog_info['author_name']}, Estimated Rating: {estimated_rating})")
```
### Predicting User Rating for a Specific Blog

```python
userId3 = 3 
blog_id = 123  

prediction = algo.predict(userId3, blog_id)

# Display prediction result
print(f"Predicted Rating for User {userId3} on Blog {blog_id}: {prediction.est}")
```
### Showing Unrated Blogs Predicted to be Rated by a User


```python
# Get list of all blog IDs
all_blog_ids = _fullBlogData['blog_id'].unique()

# Get list of blog IDs rated by the user
rated_blog_ids = _fullBlogData[_fullBlogData['userId'] == userId]['blog_id'].unique()

# Filter out unrated blog IDs
unrated_blog_ids = list(set(all_blog_ids) - set(rated_blog_ids))

# Generate predictions for unrated blogs
predictions = []
for blog_id in unrated_blog_ids:
    prediction = algo.predict(userId, blog_id)
    predictions.append((blog_id, prediction.est))

# Sort predictions based on estimated ratings 
predictions.sort(key=lambda x: x[1], reverse=True)

# Display top unrated blogs predicted to be rated by the user
top_n = 5  
print(f"Top {top_n} Unrated Blogs Predicted to be Rated by User {userId}:")
for i, (blog_id, estimated_rating) in enumerate(predictions[:top_n], 1):
    blog_info = _fullBlogData[_fullBlogData['blog_id'] == blog_id].iloc[0] 
    print(f"Rank {i}: Blog ID {blog_id} (Topic: {blog_info['topic']}, Author: {blog_info['author_name']}, Estimated Rating: {estimated_rating})")

```
