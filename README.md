# ML_Flask_App


1. Activate the python virtual environment:

If you don't already have virtualenv, install it:
pip install virtualenv

if there isn't already a folder called venv
virtualenv venv 

Activate the virtual environment: 
source venv/bin/activate

2. Install requirements:
pip install -r requirements.txt

3. Start the app: 
python3 app.py



For the docker build: 

cd app


docker build -t name .
docker run -d name
