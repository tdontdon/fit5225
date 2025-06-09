# Dockerfile
FROM public.ecr.aws/lambda/python:3.9

WORKDIR /var/task

RUN yum update -y && yum install -y glib2 mesa-libGL && yum clean all

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY tagging_processor.py .
COPY model.pt .

CMD [ "app.lambda_handler" ]