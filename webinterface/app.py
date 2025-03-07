from flask import Flask
from routes.main_routes import main_routes
from routes.feature_routes import feature_routes
from routes.algorithm_routes import algorithm_routes
from routes.training_routes import training_routes
from routes.target_routes import target_routes
from routes.testtrain_routes import testtrain_routes
from quick_modeling import model_routes

# Initialize Flask app
app = Flask(__name__)

# Register blueprints
app.register_blueprint(main_routes)
app.register_blueprint(feature_routes)
app.register_blueprint(algorithm_routes)
app.register_blueprint(training_routes)
app.register_blueprint(target_routes)
app.register_blueprint(testtrain_routes)
app.register_blueprint(model_routes)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
