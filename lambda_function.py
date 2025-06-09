import json
import cv2
import boto3
import numpy as np
import urllib.parse
import io
from decimal import Decimal

s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('BirdDatabase')
function_name = 'auto_tag'

thumbnail_scale = 0.5

def decimal_to_number(obj):
    if isinstance(obj, list):
        return [decimal_to_number(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: decimal_to_number(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    else:
        return obj



# create thumbnail and check the tags.
def lambda_handler(event, context):
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
        'Access-Control-Allow-Methods': 'OPTIONS,POST'
    }

    record = event['Records'][0]
    region = record['awsRegion']
    bucket = record['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(record['s3']['object']['key'], encoding='utf-8')

    name = key.split('.')[0]
    file_type = key.split('.')[1]
    type_within_database = None

    thumbnail_key = None
    thumbnail_url = None

    if file_type == "jpg" or file_type == "jpeg" or file_type == "png":
        type_within_database = "image"
        thumbnail_key = f"{name}-thumb.jpg"
        thumbnail_url = f"https://{bucket}.s3.{region}.amazonaws.com/{thumbnail_key}"

    elif file_type == "mp4" or file_type == "mov" or file_type == "avi" or file_type == "mkv":
        type_within_database = "video"
        thumbnail_key = 'null'
        thumbnail_url = 'null'

    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    
    response = table.get_item(
        Key={
                'url': f"https://{bucket}.s3.{region}.amazonaws.com/{key}",
                'type': type_within_database
            }
    )


    # if the data in dynamodb, it used the database's data
    if 'Item' in response:
        output = decimal_to_number(response['Item'])
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'message': 'Data already exists in the database.',
            'body': json.dumps(output)
        }

    if type_within_database == "image":
        response = s3_client.get_object(Bucket= bucket, Key = key)
        image_bytes = response['Body'].read()

        image_array = np.asanyarray(bytearray(image_bytes), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps('unable to decode image!')
            }

        resized_img = cv2.resize(img, (0, 0), fx=thumbnail_scale, fy=thumbnail_scale)

        _, buffer = cv2.imencode('.jpg', resized_img)
        resized_bytes = io.BytesIO(buffer)


        s3_client.put_object(
            Bucket=bucket,
            Key=thumbnail_key,
            Body=resized_bytes.getvalue(),
            ContentType='image/jpeg'        
        )


    obj = {
        'type': type_within_database,
        'bucket': bucket,
        'filename': key,
        'thumbnail_name': thumbnail_key,
        'url': url,
        'thumbnail_url': thumbnail_url
    }
    # tags = {
    #     "bird1": 1,
    #     "bird2": 2
    # }
    # item = {
    #     "url": url,
    #     "type": type_within_database,
    #     "filename": key,
    #     "tags": tags,
    #     "thumbnail-name": thumbnail_key,
    #     "thumbnail-url": thumbnail_url
    # }

    # table.put_item(Item=item)

    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse',  # default
        Payload = json.dumps(obj)
    )
    
    payload = json.load(response['Payload'])
    print(payload)
    return {
        'statusCode': 200,
        'headers': cors_headers,
        'body': json.dumps(payload)
    }
