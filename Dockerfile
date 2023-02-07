FROM python:3.8

#otherwise there occurs an error because some package (probably node2vec) is using some old sklearn version that is not available anymore
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
 
RUN mkdir /app
COPY ./ /app
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN apt-get -y update && apt-get -y install graphviz
RUN pip install -r requirements.txt

  
CMD ["python", "index.py"]
