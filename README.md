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

4. Build & run via docker:
docker build -t mlflaskapp .
docker run -d mlflaskapp --network="host" --name='mlflaskapp'


Deployment  not through docker follow the same steps for development 

Terminate the process 
To check if process is running in the background
ps ax | grep test.py

To terminate the process
KILL PID