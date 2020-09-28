print(__file__)

import predictor as myapp

# This is just a simple wrapper for gunicorn to find your app.
# If you want to change the algorithm file, simply change "predictor" above to the
# new file.

# import model
# model.predict("data/sample_0.csv")

import sys
print(sys.version)
print(sys.executable)

app = myapp.app
