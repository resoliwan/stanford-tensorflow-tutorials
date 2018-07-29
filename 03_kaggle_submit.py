import pandas as pd
import numpy as np

test_df = pd.read_csv('./data/kaggle_mnist/test.csv')

sample_df = pd.read_csv('./data/kaggle_mnist/sample_submission.csv')
row = sample_df.shape[0]
sub_df = pd.DataFrame({'ImageId': sample_df.ImageId, 'Label': np.ones(row)})
sub_df.to_csv('./data/kaggle_mnist/submission.csv', index=False)
