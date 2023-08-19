FROM python:3.10-slim

WORKDIR /usr/src/cloth

COPY ./requirements.txt ./

RUN pip install -r requirements.txt 

COPY . .

ENTRYPOINT ["python", "main.py"]