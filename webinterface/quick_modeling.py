

# from webgui import _app
from flask import render_template
from flask import Blueprint

# Create a Blueprint for user-related routes
model_routes = Blueprint("model_routes", __name__, url_prefix="/quickmodel")



@model_routes.route('/')
def quickmodel():
    return render_template('quick_model.html')


