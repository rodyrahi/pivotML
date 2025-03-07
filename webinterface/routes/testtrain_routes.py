from flask import Blueprint, render_template

testtrain_routes = Blueprint("testtrain_routes", __name__)

testtrainsplit = 0.4  # Default split

@testtrain_routes.route('/testtrain')
def testtrain():
    return render_template('testtrain.html', testtrainsplit=testtrainsplit)

@testtrain_routes.route('/update_split/<float:split_value>')
def update_split(split_value):
    global testtrainsplit
    testtrainsplit = split_value
    return '', 200
