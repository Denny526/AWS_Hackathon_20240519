import boto3
import json
import csv
import io
import json
import os
  
# Environment variables setting
region = os.environ['REGION_NAME']
bedrock_model_id = os.environ['BEDROCK_MODEL']
table_name = os.environ['NOTI_TABLE']
partition_key = os.environ['PARTITION_KEY']

# Initialize S3 & SNS & DynamoDB client
s3 = boto3.client('s3', region)
sns_client = boto3.client('sns')
dynamodb = boto3.resource('dynamodb') 


def save_to_dynamodb(message_id, subject, message):
    
    table = dynamodb.Table(table_name)
    
    # Define the item to be inserted
    item = {
        partition_key: message_id,
        'Subject': subject,
        'Message': message,
    }
    
    # Insert the item into DynamoDB
    table.put_item(Item=item)
    
    print(f'Jobs {message_id} created successfully')


    
    
def lambda_handler(event, context):
    
    # Extract information from the event
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        event_name = record['eventName']
        
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        csv_content = response['Body'].read().decode('utf-8')
        
        csv_rows = csv.DictReader(io.StringIO(csv_content))
    
        for row in csv_rows:
            score = float(row['anomaly_score'])
            if score > 1:
                date = row['Date']
                Target = row['Target']
                value = row['Value (kWh)']
                prompt = f"As an energy management expert with over 30 years of experience, we're reaching out to you for assistance in analyzing anomalous data related to building energy consumption. Your insights are invaluable in resolving this issue.\n\n" \
                 f"Item: {Target}\n" \
                 f"Date: {date}\n" \
                 f"Energy Consumption Value: {value} kWh\n" \
                 f"Anomaly Score: {score}\n\n" \
                         f"Please provide the following:\n\n" \
                 f"1. Brief summary of the anomaly, including how the energy consumption value deviates from normal patterns.\n\n" \
                 f"2. Analysis of possible causes for the anomaly.\n" \
                 f"3. Steps for further investigation.\n" \
                 f"4. Recommended methods to address and resolve the issue.\n\n" \
                 f"Your suggestions should be detailed and specific to effectively address the anomaly."

                bedrock = boto3.client(
                    service_name='bedrock-runtime'
                )  
            
                input_data = {
                    "modelId": bedrock_model_id,  # "cohere.command-text-v14",
                    "contentType": "application/json",
                    "accept": "*/*",
                    "body": {
                        "prompt": prompt,
                        "max_tokens": 1000,
                        "temperature": 0.75,
                        "p": 0.01,
                        "k": 0,
                        "stop_sequences": [],
                        "return_likelihoods": "NONE"
                    }
                }
                
                response = bedrock.invoke_model(
                    body=json.dumps(input_data["body"]),
                    modelId=input_data["modelId"],
                    accept=input_data["accept"],
                    contentType=input_data["contentType"]
                )
            
                response_body = json.loads(response['body'].read())
                explanation = response_body['generations'][0]['text']
                # print(explanation)
        
                sns_topic_arn = os.environ['SNS_ARN']
                message_text = f"Dear Facility Manager, \n {explanation} \n Regards,\n Your AI Assistant"
                subject_text = "Facility update: Action required for anomalous data"
                try:
                    sent_message = sns_client.publish(TopicArn=sns_topic_arn, Message=message_text, Subject=subject_text)
                    if sent_message['ResponseMetadata']['HTTPStatusCode'] == 200:
                        print(sent_message)
                        print("Notification send successfully..!!!")
                        #return True
                        message_id = sent_message['MessageId']
                        save_to_dynamodb(message_id, subject_text, message_text)
                        print("Notification stored successfully..!!!")
                except Exception as e:
                    print("Error occured while publish notifications and error is : ", e)
                    return True
