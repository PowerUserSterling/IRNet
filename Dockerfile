FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.6.0-cpu-py36-ubuntu16.04

RUN apt-get update

RUN apt-get install -y python-pip
RUN apt-get install -y libmysqlclient-dev

RUN pip install pattern \
  nltk==3.4 \
  numpy==1.14.0 \
  pytorch-pretrained-bert==0.5.1 \
  tqdm==4.31.1

WORKDIR /opt/ml/code/data
COPY IRNet/data/glove.42B.300d .
#RUN aws s3 cp s3://sagemaker-us-west-2-059252143357/glove.42B.300d.zip .
#RUN unzip glove.42B.300d.zip
#RUN rm glove.42B.300d.zip
