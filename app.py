import os

from flask import Flask, request, redirect
from flask import render_template

from models import get_requested_data, neural_network_model
import tensorflow as tf

UPLOAD_FOLDER = './temp'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "topSecret"

############# HOME PAGE ####################
# Simply displays the home page (group info)
@app.route('/')
def home_page():
    return render_template('home.html')
############################################


############# UPLOAD PAGE ##################
# Function for allowed file types (used below)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to handle the user upload
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return redirect(request.url)
        # Check that the file extension is allowed
        if file and allowed_file(file.filename):
            filename = 'tempCsv.csv'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Now the prediction function (neural network must be run on
            # the data (you can view it in the models.py file)
            neural_network_model()

            # Now we are sent to the dashboard
            return redirect('/dashboard')

    return render_template('upload.html')


# When the dashboard page is reached it renders the dashboard.html template
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard_func():

    # When first hitting the dashboard, the user is shown data for all departments
    if request.method == 'GET':
        departments, leaving, satis, proj, evalu, hours, time = get_requested_data('all')

        page_title = "Select a Department"

    # However, when they choose a department from the dropdown in the dashboard,
    # they are returned data for that specific department
    else:

        departments, leaving, satis, proj, evalu, hours, time = get_requested_data(request.form['selection'])

        page_title = request.form['selection']

    # Return all of the data. Jinja can pass this to the javascript in the HTML page
    return render_template('dashboard.html',
                           leaving_list=leaving,
                           dept_list=departments,
                           satis_data=satis,
                           proj_data=proj,
                           evalu_data=evalu,
                           hour_data=hours,
                           time_data=time,
                           page_title=page_title)

if __name__ == '__main__':
    app.run()
