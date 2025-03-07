from flask import Blueprint, render_template
from data.database import *
from data.sql_database import *

target_routes = Blueprint("target_routes", __name__)


target_column =  get_project_value('1' , 'target_column')

@target_routes.route('/targets')
def targets():
        
        target_column = get_project_value('1' , 'target_column')
        

        if target_column == '':
            selected_target = 'No target selected'
        else:
            selected_target = target_column


        
            
        targets  = df.columns.tolist()
        return render_template('targets.html' , targets = targets , selected_target = selected_target)


@target_routes.route('/add_target/<target_column_selected>')
def add_target(target_column_selected):
    global target_column
    target_column = target_column_selected

    update_project_value('1' , 'target_column' , target_column)

    return '', 200
