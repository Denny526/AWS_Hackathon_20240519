#!/usr/bin/env python
# coding: utf-8

# In[2]:


import boto3
import pandas as pd
import numpy as np
from sagemaker import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# S3 bucket and object information--input
bucket_name = 'aws-toys-rong'
file_key = '每日能耗資料(20240518).csv'

# S3 bucket and file information--output
s3_file_key = 'anomaly_scores.csv'

# SageMaker endpoint name
endpoint_name = 'randomcutforest-2024-05-18-14-07-51-044'

# Initialize S3 client
s3_client = boto3.client('s3')


# In[3]:


def predict_data():
    # Initialize S3 client
    s3_client = boto3.client('s3')
    # Download the file from S3
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    data = pd.read_csv(obj['Body'])
    data = data.loc[:, ['Date', 'Target', 'Value (kWh)']]
    
    # 準備數據
    values = data['Value (kWh)'].values.reshape(-1, 1)
    
    # Initialize the SageMaker Predictor
    predictor = Predictor(
        endpoint_name=endpoint_name,
        serializer=CSVSerializer(),         # Serialize input data to CSV format
        deserializer=JSONDeserializer()     # Deserialize the output from JSON format
    )
    
    # Predict anomaly scores
    results = predictor.predict(values)

    # Extract the scores from the results
    anomaly_scores = [record['score'] for record in results['scores']]

    # Add the anomaly scores to the original dataframe
    data['anomaly_score'] = anomaly_scores
    # 将 DataFrame 保存为 CSV 文件
    csv_file_path = '/tmp/data_with_anomaly_scores.csv'
    data.to_csv(csv_file_path, index=False)
    

    # 初始化 S3 客户端
    s3_client = boto3.client('s3')

    # 上传 CSV 文件到 S3
    s3_client.upload_file(csv_file_path, bucket_name, s3_file_key)


# In[4]:


predict_data()

