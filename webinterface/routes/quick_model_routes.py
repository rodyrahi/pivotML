from flask import Blueprint, render_template

quickmodel_routes = Blueprint("quickmodel_routes", __name__)

@quickmodel_routes.route('/quickmodel')
def quickmodel():
    return render_template('quick_model.html')