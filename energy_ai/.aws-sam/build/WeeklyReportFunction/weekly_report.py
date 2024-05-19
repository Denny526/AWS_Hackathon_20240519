import boto3
import json
import csv
import io
import json
import os
  
# Environment variables setting
region = os.environ['REGION_NAME']
bedrock_model_id = os.environ['BEDROCK_MODEL']


# Initialize S3 & SNS & DynamoDB client
s3 = boto3.client('s3', region)
sns_client = boto3.client('sns')

def gen_weekly_report(event, context):
    
    # Extract information from the event
    bucket_name = os.environ['S3_BUCKET_NAME']
    object_key = os.environ['S3_OBJECT_KEY']
    
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    csv_content = response['Body'].read().decode('utf-8')
    
    # csv to json
    csv_rows = csv.DictReader(io.StringIO(csv_content))
    json_output = json.dumps([row for row in csv_rows])
    # print(json_output)

    prompt = f"""
    As an energy management expert with over 30 years of experience, we're reaching out to you for assistance in analyzing weekly data related to building energy consumption. Your insights are invaluable in resolving this issue.
    
    Here is the weekly energy consumption data in JSON format:
        
    {json_output}
    
    Please provide your advice with the following structure:
    1. A summary of the total energy consumption for the week, compared to the previous week and the 4-week average.
    2. Identification of any abnormal usage patterns, such as significant spikes or drops in consumption on particular days.
    3. Potential reasons for abnormal patterns, such as weather events, operational changes, faulty equipment, etc. Provide relevant recommendations to address any issues.
    4. Energy usage highlights, such as days with highest/lowest consumption, top consuming facilities, etc.
    5. Forecasts for energy needs in the upcoming week based on historical data and known factors.
    6. Your suggestions should be detailed and specific to effectively address the weekly report.
   
    Please generate the weekly energy usage report based on the data provided, following the guidelines outlined above. Keep the report concise, focused and insightful.
    """

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

    sns_topic_arn = os.environ['SNS_ARN']
    message_text = f"Dear Facility Manager, \n {explanation} \n Regards,\n Your AI Assistant"
    subject_text = "Facility update: Weekly report for energy data"
    try:
        sent_message = sns_client.publish(TopicArn=sns_topic_arn, Message=message_text, Subject=subject_text)
        if sent_message['ResponseMetadata']['HTTPStatusCode'] == 200:
            print(sent_message)
            print("Report send successfully..!!!")
            return True
    except Exception as e:
        print("Error occured while publish reports and error is : ", e)
        return True
