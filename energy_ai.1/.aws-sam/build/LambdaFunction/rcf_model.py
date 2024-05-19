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

import os
import logging
# import pathlib
import uuid 
import csv

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def rcf():

    # 設定 SageMaker initial
    session = sagemaker.Session()
    role = get_execution_role()

    
    
    # 設定 S3 bucket 和文件名
    bucket = os.environ['DATA'] # 'aws-toys-rong'
    data_key = '研華_瑞光大樓_每日能耗資料(20230101~20240510).csv'
    data_location = f's3://{bucket}/{data_key}'
    
    # 從 S3 讀取數據
    data = pd.read_csv(data_location,encoding='utf-8',)
    data = data.loc[:, ['Date', 'Target', 'Value (kWh)']]
    
    
    # 準備數據
    values = data['Value (kWh)'].values.reshape(-1, 1)
    
    # 將數據轉換為 RecordIO protobuf 格式並保存到本地文件
    recordio_protobuf_file = 'data.protobuf'
    with open(recordio_protobuf_file, 'wb') as f:
        write_numpy_to_dense_tensor(f, values)
    
    # 上傳 RecordIO protobuf 文件到 S3
    s3_data_path = session.upload_data(recordio_protobuf_file, bucket=bucket, key_prefix='sagemaker/rcf')
    
    
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

    
    # 印出剛剛建立的Training job跟Endpoint，可以到UI介面上看有沒有出現
    print('Training job name: {}'.format(rcf.latest_training_job.job_name))
    print('Endpoint name: {}'.format(rcf_predictor.endpoint))
        
    # SageMaker endpoint name
    endpoint_name = 'randomcutforest-2024-05-18-14-07-51-044'
    
    # Initialize S3 client
    s3_client = boto3.client('s3')


    # # Download the file from S3
    # obj = s3_client.get_object(Bucket=bucket, Key=data_key)
    # data = pd.read_csv(obj['Body'])
    
    # data = data.loc[:, ['Date', 'Target', 'Value (kWh)']]
    
    
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
    data['id'] = str(uuid.uuid4())  # Generate a random UUID
    
    
    # 将 DataFrame 保存为 CSV 文件
    csv_file_path = '/tmp/data_with_anomaly_scores.csv'
    data.to_csv(csv_file_path, index=False)
    

    
    table_name = os.environ['SCORE_DATA']

    dynamodb = boto3.resource('dynamodb') 


    table = dynamodb.Table(table_name)
    
    # Parse CSV data
    # csv_data = event['body']
    csv_rows = data.splitlines()
    csv_reader = csv.reader(csv_rows)
    
    # Iterate over CSV rows and insert into DynamoDB
    for row in csv_reader:
        item = {
            'id': row['id'], 
            'date': row['Date'],
            'target': row['Target'],
            'value': row['Value (kWh)'],
            'anomaly_score': row['anomaly_score']
        }
        table.put_item(Item=item)
            
    # # Define the item to be inserted
    # item = {
    #     partition_key: message_id,
    #     'Subject': subject,
    #     'Message': message,
    #     # 'RecordTime': datetime.now()
    # }
    
    # # Insert the item into DynamoDB
    # table.put_item(Item=item)
    
        print(f'Job created successfully')


    
    
    
    # save_to_dynamodb(message_id, subject_text, message_text)
    
    # # S3 bucket and file information
    # bucket_name = os.environ['SCORE_DATA'] # 'aws-toys-rong'
    # s3_file_key = 'anomaly_scores.csv'
    
    # # 初始化 S3 客户端
    # s3_client = boto3.client('s3')
    
    # 上传 CSV 文件到 S3
    # s3_client.upload_file(csv_file_path, bucket_name, s3_file_key)
    
    
    
    # try:
        

        
    
    #     bucket_name = os.environ['SCORE_DATA']
    #     region = os.environ['REGION_NAME']
    #     s3_file_key = 'anomaly_scores.csv'
    #     # csv_file_path = '/tmp/anomaly_scores.csv'  # Ensure this file exists and is correct
    #     csv_file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "sample_file.txt")
    #     # s3_client.put_object(Bucket=bucket_name, Key=s3_file_key, Body=data)
        
    #     # # Create the S3 bucket
    #     logger.info(f'Creating bucket {bucket_name} in region {region}')
    #     # s3_client.create_bucket(
    #     #     Bucket=bucket_name,
    #     #     CreateBucketConfiguration={
    #     #         'LocationConstraint': region
    #     #     }
    #     # )
    #     # logger.info(f'Bucket {bucket_name} created successfully')
        
    #     # # Upload the file
    #     # logger.info(f'Uploading {csv_file_path} to bucket {bucket_name} with key {s3_file_key}')
    #     response = s3_client.upload_file(csv_file_path, bucket_name, s3_file_key)
    #     print(response)
    #     logger.info('Upload successful')
    #     print(f"DataFrame with anomaly scores has been uploaded to s3://{bucket_name}/{s3_file_key}")

    # except Exception as e:
    #     logger.error(f'Error: {e}')
    #     raise
    
    
    
    
