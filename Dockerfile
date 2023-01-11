FROM python:3.8
 
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install gunicorn
RUN pip install -r requirements.txt
RUN pip install ocpa==1.0.0
  
COPY ./ /app
  
CMD gunicorn --bind 0.0.0.0:8050 index:app