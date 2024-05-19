#!/usr/bin/env python
# coding: utf-8

# In[27]:


import sagemaker
import boto3
import pandas as pd
import numpy as np
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput
from sagemaker.amazon.common import write_numpy_to_dense_tensor
from sagemaker import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer


# In[9]:


# 設定 SageMaker initial
session = sagemaker.Session()
role = get_execution_role()


# In[10]:


# 設定 S3 bucket 和文件名
bucket = 'aws-toys-rong'
data_key = '研華_瑞光大樓_每日能耗資料(20230101~20240510).csv'
data_location = f's3://{bucket}/{data_key}'


# In[11]:


# 從 S3 讀取數據
data = pd.read_csv(data_location,encoding='utf-8',)
data = data.loc[:, ['Date', 'Target', 'Value (kWh)']]


# In[12]:


data


# In[16]:


# 準備數據
values = data['Value (kWh)'].values.reshape(-1, 1)

# 將數據轉換為 RecordIO protobuf 格式並保存到本地文件
recordio_protobuf_file = 'data.protobuf'
with open(recordio_protobuf_file, 'wb') as f:
    write_numpy_to_dense_tensor(f, values)

# 上傳 RecordIO protobuf 文件到 S3
s3_data_path = session.upload_data(recordio_protobuf_file, bucket=bucket, key_prefix='sagemaker/rcf')


# In[ ]:


# 配置 RCF 模型
container = get_image_uri(boto3.Session().region_name, 'randomcutforest')
rcf = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=2,
    instance_type='ml.m4.xlarge',
    output_path=f's3://{session.default_bucket()}/output',
    sagemaker_session=session
)

# 設置 RCF 超參數
rcf.set_hyperparameters(
    num_samples_per_tree=256,
    num_trees=50,
    feature_dim=1
)

# 訓練 RCF 模型
rcf.fit({'train': TrainingInput(s3_data_path, distribution='ShardedByS3Key')})

# 部署模型
rcf_predictor = rcf.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)


# In[18]:


# 印出剛剛建立的Training job跟Endpoint，可以到UI介面上看有沒有出現
print('Training job name: {}'.format(rcf.latest_training_job.job_name))
print('Endpoint name: {}'.format(rcf_predictor.endpoint))

