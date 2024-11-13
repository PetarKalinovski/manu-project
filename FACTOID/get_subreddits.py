import pandas as pd
import numpy as np
from collections import Counter
import datetime

df = pd.read_csv('reddit_dataset.csv')

user_subreddit_counts = []

for i, row in df.iterrows():
    user_id = row['user_id']
    documents = eval(row['documents'])
    subreddit_counts = Counter([doc[3] for doc in documents])
    for subreddit, count in subreddit_counts.items():
        user_subreddit_counts.append({
            'user_id': user_id,
            'subreddit': subreddit,
            'post_count': count
        })
subreddit_df = pd.DataFrame(user_subreddit_counts)
subreddit_df['total_posts'] = subreddit_df.groupby('user_id')['post_count'].transform('sum')
subreddit_df['percentage'] = (subreddit_df['post_count'] / subreddit_df['total_posts']) * 100
most_active_subreddits = subreddit_df.loc[subreddit_df.groupby('user_id')['percentage'].idxmax()][['user_id', 'subreddit', 'percentage']]

subreddit_df.to_csv('subreddit_dataset.csv', index=False)
most_active_subreddits.to_csv('most_active_subreddits.csv', index=False)
