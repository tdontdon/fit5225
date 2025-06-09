# app.py
import boto3
import os
import json
import requests
import tempfile
from urllib.parse import unquote_plus 
import tagging_processor 

s3_client = boto3.client('s3')
sns = boto3.client("sns")
dynamodb_resource = boto3.resource('dynamodb')
TABLE_NAME = os.environ["TABLE_NAME"]
TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]

table = dynamodb_resource.Table(TABLE_NAME)

model_path = 'model.pt'

def lambda_handler(event, context):

    file_type = event['type']
    bucket = event['bucket']
    filename = event['filename']
    thumbnail_name = event['thumbnail_name']
    url = event['url']
    thumbnail_url = event['thumbnail_url']
    
    try:
        model_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': filename},
            ExpiresIn=300  # 5 minutes
        )

        response = requests.get(model_url)
        if response.status_code != 200:
            return {
            'statusCode': 500,
            'body': json.dumps("Failed to download image or video from presigned URL")
        }
        
        file_type_temp = filename.split(".")[-1]
        
        suffix = f".{file_type_temp}"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        # based on different file type to invoke different AI feature.
        if file_type == 'image':
            detected_tags, detected_counts = tagging_processor.extract_tags_from_image(temp_file_path, model_path)
        elif file_type =='video':
            detected_tags, detected_counts = tagging_processor.extract_tags_from_video(temp_file_path, model_path)

        lowercased_tags = {k.lower(): v for k, v in detected_counts.items()}
        item = {
            "url": url,
            "type": file_type,
            "filename": filename,
            "tags": lowercased_tags,
            "thumbnail-name": thumbnail_name,
            "thumbnail-url": thumbnail_url
        }

        print(item)
        response = table.put_item(Item=item)



        if TOPIC_ARN:
            birds_added = [pair.split(',', 1)[0].strip() for pair in lowercased_tags]
            sns.publish(
                TopicArn=TOPIC_ARN,
                Subject="New Bird Insert",
                Message=json.dumps({
                    "url": item['url'],
                    "tags": birds_added
                }, ensure_ascii=False),
                MessageAttributes={
                    "tag": {
                        "DataType": "String.Array",
                        "StringValue": json.dumps(birds_added)
                    }
                }
            )

        print(response)
        return {
                'statusCode': 200,
                'body': json.dumps(response)
            }


    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }


    
    
