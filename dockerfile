# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/

FROM python:3.9-slim-bullseye

ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run the application:
COPY . . 
CMD ["python3", "app.py"]




# FROM ubuntu:16.04
# WORKDIR /usr/src/app

# RUN apt update -y
# RUN apt install python3 -y
# RUN apt install python3-pip -y

# COPY ./requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt


# COPY . .

# RUN python3 app.py

