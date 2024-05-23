from flask import Flask, g, render_template, request, redirect, url_for
import sqlite3
from markupsafe import escape
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import time
import dill
from joblib import load
from sklearn.ensemble import RandomForestClassifier
import lime
import io
import pickle
import joblib
from PIL import Image


conn = sqlite3.connect('database.db')
cursor = conn.cursor()
cursor.execute("SELECT explainer FROM LimeExplainers WHERE id=?", (1,))
row = cursor.fetchone()
explainer_data = row[0]
try:
    explainer_lime = dill.loads(explainer_data)
except TypeError:
    print("Error while unpickling. Please check your environment.")
print(explainer_lime)



app = Flask(__name__)
app.config['DATABASE'] = 'database.db'
conn = sqlite3.connect('database.db')
cursor = conn.cursor()
cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS outcome_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_id INTEGER,
        outcome INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row  
    return db

def retrieve_dataframe_from_db(db_name, table_name):
    conn = sqlite3.connect(db_name)
    dataframe = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    conn.close()
    return dataframe

differences = retrieve_dataframe_from_db('database.db', 'differences')
individual = list(differences['individual'].values)

all_values = list(differences['sex_race_ethnicity'].values)
all_values = [int(x) for x in all_values if str(x) != 'nan']

race_sex = list(differences['sex_race'].values)
race_sex = [int(x) for x in race_sex if str(x) != 'nan']

race_ethnicity =list(differences['race_ethnicity'].values)
race_ethnicity = [int(x) for x in race_ethnicity if str(x) != 'nan']

sex_ethnicity =list(differences['sex_ethnicity'].values)
sex_ethnicity= [int(x) for x in sex_ethnicity if str(x) != 'nan']

sex =list(differences['sex'].values)
sex= [int(x) for x in sex if str(x) != 'nan']

ethnicity =list(differences['ethnicity'].values)
ethnicity= [int(x) for x in ethnicity if str(x) != 'nan']

race =list(differences['race'].values)
race= [int(x) for x in race if str(x) != 'nan']

# get all applicants

# fetch outcome data from the database
def get_outcome():
    db = get_db()
    cur = db.execute('SELECT * FROM outcomes')
    rows = cur.fetchall()
    return rows


@app.teardown_appcontext
def close_connection(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/validate')
def applicants():
    db = get_db()
    placeholders = ', '.join('?' for _ in individual)
    query = f'''
        SELECT applicants.*, outcomes.value
        FROM applicants
        LEFT JOIN outcomes ON applicants."index" = outcomes.id
        WHERE applicants."index" IN ({placeholders})
    '''
    cur = db.execute(query, individual)
    rows = cur.fetchall()

    col_names = [description[0] for description in cur.description]
    return render_template('validate.html', col_names=col_names, rows=rows)

@app.route('/groupstats')
def group_stats():
    db = get_db()
    placeholders = ', '.join('?' for _ in individual)
    query = f'''
        SELECT applicants.*, outcomes.value
        FROM applicants
        LEFT JOIN outcomes ON applicants."index" = outcomes.id
        WHERE applicants."index" IN ({placeholders})
    '''
    cur = db.execute(query, individual)
    rows = cur.fetchall()

    col_names = [description[0] for description in cur.description]
    return render_template('groupstats.html', col_names=col_names, rows=rows)

@app.route('/explain', methods=['GET', 'POST'])  
def explainability():
    bar_plot_url = None
    violin_plot_url = None
    if request.method == 'POST':
        max_display = int(request.form.get('max_display'))
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        cursor.execute("SELECT array, shape FROM ShapValues WHERE id=?", (1,))
        row = cursor.fetchone()
        retrieved_array_binary = row[0]
        retrieved_shape = tuple(map(int, row[1].strip('()').split(',')))
        retrieved_array = np.fromstring(retrieved_array_binary)
        retrieved_array = retrieved_array.reshape(retrieved_shape)
        column_names = retrieve_dataframe_from_db('database.db', 'applicantsShap').columns[2:]
        explanation = shap.Explanation(values=retrieved_array, feature_names=column_names)
        
        plt.figure() 
        shap.plots.bar(explanation, show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig('static/images/shap_plot_bar.png')

        plt.figure() 
        shap.plots.violin(explanation, show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig('static/images/shap_plot_violin.png')

        bar_plot_url = url_for('static', filename='images/shap_plot_bar.png', _external=True) + '?v=' + str(time.time())
        violin_plot_url = url_for('static', filename='images/shap_plot_violin.png', _external=True) + '?v=' + str(time.time())

    return render_template('explain.html', bar_plot_url=bar_plot_url, violin_plot_url=violin_plot_url)

@app.route('/subgroups', methods= ['GET', 'POST'])
def subgroups():
    if request.method == 'POST':
        db = get_db()
        selected_values = request.form.get('selectedValues')
        print(selected_values)
        if selected_values is None:
            return render_template('error.html', message="No values selected")
        selected_values_list = selected_values.split(',')
        print(selected_values_list)
        if selected_values_list == ['']:
            return render_template('error.html', message="No values selected")
        elif selected_values_list == ['race', 'sex', 'ethnicity']:
            print(all_values)
            placeholders = ', '.join('?' for _ in all_values)
            query = f'''
                SELECT applicants.*, outcomes.value
                FROM applicants
                LEFT JOIN outcomes ON applicants."index" = outcomes.id
                WHERE applicants."index" IN ({placeholders})
            '''
            cur = db.execute(query, all_values)
            rows = cur.fetchall()

            col_names = [description[0] for description in cur.description]
            return render_template('validate.html', col_names=col_names, rows=rows)
        elif selected_values_list == ['race', 'sex']:
            print(race_sex)
            placeholders = ', '.join('?' for _ in race_sex)
            query = f'''
                SELECT applicants.*, outcomes.value
                FROM applicants
                LEFT JOIN outcomes ON applicants."index" = outcomes.id
                WHERE applicants."index" IN ({placeholders})
            '''
            cur = db.execute(query, race_sex)
            rows = cur.fetchall()

            col_names = [description[0] for description in cur.description]
            return render_template('validate.html', col_names=col_names, rows=rows)
        elif selected_values_list == ['race', 'ethnicity']:
            print(race_ethnicity)
            placeholders = ', '.join('?' for _ in race_ethnicity)
            query = f'''
                SELECT applicants.*, outcomes.value
                FROM applicants
                LEFT JOIN outcomes ON applicants."index" = outcomes.id
                WHERE applicants."index" IN ({placeholders})
            '''
            cur = db.execute(query, race_ethnicity)
            rows = cur.fetchall()

            col_names = [description[0] for description in cur.description]
            return render_template('validate.html', col_names=col_names, rows=rows)
        elif selected_values_list == ['sex', 'ethnicity']:
            print(sex_ethnicity)
            placeholders = ', '.join('?' for _ in sex_ethnicity)
            query = f'''
                SELECT applicants.*, outcomes.value
                FROM applicants
                LEFT JOIN outcomes ON applicants."index" = outcomes.id
                WHERE applicants."index" IN ({placeholders})
            '''
            cur = db.execute(query, sex_ethnicity)
            rows = cur.fetchall()

            col_names = [description[0] for description in cur.description]
            return render_template('validate.html', col_names=col_names, rows=rows)
        elif selected_values_list == ['sex']:
            print(sex)
            placeholders = ', '.join('?' for _ in sex)
            query = f'''
                SELECT applicants.*, outcomes.value
                FROM applicants
                LEFT JOIN outcomes ON applicants."index" = outcomes.id
                WHERE applicants."index" IN ({placeholders})
            '''
            cur = db.execute(query, sex)
            rows = cur.fetchall()

            col_names = [description[0] for description in cur.description]
            return render_template('validate.html', col_names=col_names, rows=rows)
        elif selected_values_list == ['race']:
            print(race)
            placeholders = ', '.join('?' for _ in race)
            query = f'''
                SELECT applicants.*, outcomes.value
                FROM applicants
                LEFT JOIN outcomes ON applicants."index" = outcomes.id
                WHERE applicants."index" IN ({placeholders})
            '''
            cur = db.execute(query, race)
            rows = cur.fetchall()

            col_names = [description[0] for description in cur.description]
            return render_template('validate.html', col_names=col_names, rows=rows)
        elif selected_values_list == ['ethnicity']:
            print(ethnicity)
            placeholders = ', '.join('?' for _ in ethnicity)
            query = f'''
                SELECT applicants.*, outcomes.value
                FROM applicants
                LEFT JOIN outcomes ON applicants."index" = outcomes.id
                WHERE applicants."index" IN ({placeholders})
            '''
            cur = db.execute(query, ethnicity)
            rows = cur.fetchall()

            col_names = [description[0] for description in cur.description]
            return render_template('validate.html', col_names=col_names, rows=rows)
    else:
        # Render the page with no data if the method is GET
        return render_template('subgroups.html')

#for display for outcome edit
@app.route('/edit_outcome/<int:applicant_id>', methods=['GET'])
def edit_outcome(applicant_id):
    db = get_db()
    if request.method == 'POST':
        outcome = request.form['outcome']
        db.execute('''
            INSERT INTO outcomes (id, value)
            VALUES (?, ?)
        ''', (applicant_id, outcome))       
        
        #db.execute('DELETE FROM applicants WHERE id = ?', (applicant_id,))

        db.commit()
        return redirect(url_for('applicants'))
    
    cur = db.execute('SELECT id, value FROM  outcomes WHERE id = ?', (applicant_id,))
    outcome = cur.fetchone()
    if outcome is None:
        outcome = {'id': applicant_id, 'value': ''}

    return render_template('edit_outcome.html', outcome=outcome)

@app.route('/explanation_lime/<int:applicant_id>', methods=['GET', 'POST'])
def lime_outcome(applicant_id):
    print('hi')
    if request.method == 'POST':
        max_display_lime = int(request.form.get('max_display_lime'))
        print('hi2')
        db = get_db()
        
        cur = db.execute('SELECT model FROM LimeExplainers WHERE id = ?', (1,))
        row = cur.fetchone()

        model_blob = row[0]

       

        rf_model = joblib.load(io.BytesIO(model_blob))
        print('hello')
        cur.execute('SELECT * FROM applicantsShap WHERE "index"=?', (applicant_id,))
        row = cur.fetchone()
    

        cur = db.execute("PRAGMA table_info(applicantsShap)")
        columns = [column[1] for column in cur.fetchall()]
        columns = columns[2:]
        

        db.close()
        row_values = list(row)  # Convert the row to a list
        #row_values = np.array(row_values)  
        np_row = row_values[2:]
        np_row_df = pd.DataFrame([np_row], columns=columns)

        print(max_display_lime)

        exp = explainer_lime.explain_instance(np_row_df.iloc[0], rf_model.predict_proba, num_features=max_display_lime)
        fig = plt.figure(figsize=(50, 50))
        exp.as_pyplot_figure()
        fig.savefig('static/images/explanation.png')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Create an image file from the BytesIO object
        image = Image.open(buf)
        image.save('static/images/explanation.png')

        explanation_url_lime = url_for('static', filename='images/explanation.png', _external=True) + '?v=' + str(time.time())

        return render_template('explanation_lime.html', explanation_url_lime=explanation_url_lime)

@app.route('/groupstats', methods= ['GET', 'POST'])
def groupstats():
    if request.method == 'POST':
        selected_values = request.form.get('selectedValues')
        print(selected_values)
        if selected_values == '':
            return render_template('error.html', message="No values selected")
        elif selected_values == 'sex':
            group_split_url = url_for('static', filename='images/sex_split.png', _external=True) + '?v=' + str(time.time())
            group_stats_url = url_for('static', filename='images/sex_stats.png', _external=True) + '?v=' + str(time.time())
            return render_template('groupstats.html', group_split_url=group_split_url, group_stats_url=group_stats_url)
        elif selected_values == 'race':
            group_split_url = url_for('static', filename='images/race_split.png', _external=True) + '?v=' + str(time.time())
            group_stats_url = url_for('static', filename='images/race_stats.png', _external=True) + '?v=' + str(time.time())
            return render_template('groupstats.html', group_split_url=group_split_url, group_stats_url=group_stats_url)
        elif selected_values == 'ethnicity':
            group_split_url = url_for('static', filename='images/ethnicity_split.png', _external=True) + '?v=' + str(time.time())
            group_stats_url = url_for('static', filename='images/ethnicity_stats.png', _external=True) + '?v=' + str(time.time())
            return render_template('groupstats.html', group_split_url=group_split_url, group_stats_url=group_stats_url)
    else:
        # Render the page with no data if the method is GET
        return render_template('groupstats.html')

#form submission for outcome update
@app.route('/edit_outcome/<int:applicant_id>', methods=['POST'])
def update_outcome(applicant_id):
    outcome = request.form['outcome']
    db = get_db()
    db.execute('''
        INSERT INTO outcomes (id, value)
        VALUES (?, ?)
        ON CONFLICT(id) DO UPDATE SET value=excluded.value
    ''', (applicant_id, outcome))

    # Log the change
    db.execute('''
        INSERT INTO outcome_logs (applicant_id, outcome, timestamp)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    ''', (applicant_id, outcome))


    db.commit()
    return redirect(url_for('applicants'))


# @app.route("/")
# def index():
#     return 'Index Page'

# @app.route("/explain")
# def explain():
#     return 'Explainability'

# @app.route('/user/<username>')
# def show_user_profile(username):
    # show the user profile for that user
    # return f'User {escape(username)}'

if __name__ == '__main__':
    app.run(debug=True, port=5000)