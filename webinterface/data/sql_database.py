import sqlite3
import json

# Database file
DB_FILE = "automl.db"

def init_db():
    """Initialize the SQLite database and create tables."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        csv_path TEXT NOT NULL,

        dataframe_path TEXT NOT NULL,
        features_used TEXT NOT NULL, -- Stored as a JSON string
        target_column TEXT NOT NULL,
        testtrainsplit REAL NOT NULL,
        algorithms_used TEXT NOT NULL, -- Stored as a JSON string
        metadata TEXT NOT NULL -- JSON metadata about model and dataset
                   
    )
    """)

    conn.commit()
    conn.close()

def save_project(user_id, csv_path,  dataframe_path, features_used, target_column, testtrainsplit, algorithms_used, metadata):
    
    init_db()
    
    """Save a project to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO projects (user_id, csv_path, dataframe_path, features_used, target_column, testtrainsplit, algorithms_used, metadata) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, 
        csv_path,
       
        dataframe_path,
        json.dumps(features_used),  # Convert list to JSON string
        target_column,
        testtrainsplit,
        json.dumps(algorithms_used),  # Convert list to JSON string
        json.dumps(metadata)  # Convert dict to JSON string
    ))

    conn.commit()
    conn.close()


def update_project_value(project_id, field_name, new_value):
    """Update a specific field for a project in the database."""
    init_db()
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Handle JSON serialization for specific fields
    if field_name in ['features_used', 'algorithms_used', 'metadata']:
        new_value = json.dumps(new_value)

    cursor.execute(f"""
    UPDATE projects 
    SET {field_name} = ?
    WHERE id = ?
    """, (new_value, project_id))

    conn.commit()
    conn.close()

def get_project_value(project_id, field_name):
    """Retrieve a specific field value for a project from the database."""
    init_db()
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(f"""
    SELECT {field_name}
    FROM projects
    WHERE id = ?
    """, (project_id,))
    
    value = cursor.fetchone()
    conn.close()

    # Handle JSON deserialization for specific fields
    if value and field_name in ['features_used', 'algorithms_used', 'metadata']:
        return json.loads(value[0])
    
    return value[0] if value else None





    
def get_projects():
    """Retrieve all projects from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM projects")
    projects = cursor.fetchall()

    conn.close()
    
    # Convert JSON fields back to Python objects
    project_list = []
    for project in projects:
        project_list.append({
            "id": project[0],
            "user_id": project[1],
            "csv_path": project[2],
            "dataframe": project[3],
            "features_used": json.loads(project[4]),
            "target_column": project[5],
            "testtrainsplit": project[6],
            "algorithms_used": json.loads(project[7]),
            "metadata": json.loads(project[8])

        })
    
    return project_list


# Initialize the database (run this once)
if __name__ == "__main__":
    init_db()
    print("Database initialized.")
