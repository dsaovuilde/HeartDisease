import dash
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import dash_daq as daq
import math

app = dash.Dash(__name__, )
server = app.server

# download CSV file from gitHub
url = 'https://raw.githubusercontent.com/dsaovuilde/HeartDisease/master/Data/Heart.csv'
heart = pd.read_csv(url)

# train_test split to split the dataframe into test data and train data for prediction models
y = pd.DataFrame(heart['target'])
X = heart.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)

rf = RandomForestClassifier(n_estimators=733, min_samples_split=5, min_samples_leaf=4, max_features='auto',
                            max_depth=80, bootstrap=True, random_state=7896543)
rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
dt = pd.DataFrame(X_test)
dt['predicted'] = prediction
dt['actual'] = y_test
cm = confusion_matrix(y_test, prediction)

# calculate sensitivity and specificity
true_negatives = cm[0][0]
true_positives = cm[1][1]
false_positives = cm[0][1]
false_negatives = cm[1][0]
sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)


def plot_confusion_matrix():
    labels = ['False', 'True']
    data = go.Heatmap(z=cm, y=labels, x=labels, colorscale='tealgrn')
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": labels[i],
                    "y": labels[j],
                    "font": {"color": "black"},
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                }
            )
    layout = {
        "title": 'confusion matrix',
        "xaxis": {"title": "Actual value"},
        "yaxis": {"title": "Predicted value"},
        "annotations": annotations
    }
    fig = go.Figure(data=data, layout=layout)
    return fig


cols = list(heart.columns.values)
cols.remove('target')
lr = LogisticRegression(random_state=0).fit(X, y)

heatmap = go.Heatmap(
    x=list(heart.columns.values),
    y=list(heart.columns.values),
    z=heart.corr(),
    type='heatmap',
    colorscale='Viridis',
)

data = [heatmap]
for col in heart.columns:
    num = heart[col].max() - heart[col].min()

# fig1 is a bar graph of logistic regression coefficients
fig1 = go.Figure([go.Bar(x=cols, y=lr.coef_[0])])
fig1.update_layout(title='coefficients')
fig1.update_layout(xaxis_title='Intercept = .12508204')

# fig2 is heatmap
fig2 = go.Figure(data=data,
                 layout=go.Layout(
                     title=go.layout.Title(text='Heatmap')
                 )
                 )

# fig3 is a scatter chart depcition heart disease as a function of age and maximum heart rate
fig3 = go.Figure()

fig3.add_trace(
    go.Scatter(x=heart.age[heart.target == 1], y=heart.thalach[heart.target == 1], mode='markers', name='Disease'))
fig3.add_trace(
    go.Scatter(x=heart.age[heart.target == 0], y=heart.thalach[heart.target == 0], mode='markers', name='No Disease'))
fig3.update_layout(title='Heart Disease as a function of Age and Maximum Heart rate', xaxis_title='Age',
                   yaxis_title='Max Heart Rate')

# 4 is a scatter chart showing heart disease as a function of type of chest pain and blood pressure
fig4 = go.Figure()
fig4.add_trace(
    go.Scatter(x=heart.cp[heart.target == 1], y=heart.trestbps[heart.target == 1], mode='markers', name='Disease'))
fig4.add_trace(
    go.Scatter(x=heart.cp[heart.target == 0], y=heart.trestbps[heart.target == 0], mode='markers', name='No Disease'))
fig4.update_layout(title='Heart Disease as a function of Chest Pain and Blood Pressure', xaxis_title='Chest Pain',
                   yaxis_title='Blood Pressure')

app.layout = html.Div([
    html.Div([
        html.H1('Descriptive Model', style={'textAlign': 'center'}),
        html.Div([
            dcc.Graph(
                figure=(fig2)
            )
        ], style={'display': 'inline-block'}),
        html.Div([
            dcc.Graph(figure=fig3),
        ], style={'display': 'inline-block'}),
        html.Div([
            dcc.Graph(figure=fig4),
        ], style={'display': 'inline-block'}),
        html.Div([
            html.H2('Legend'),
            html.P('age - age'),
            html.P('sex - sex(1=male, 0=female)'),
            html.P('cp - chest pain(1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 0 = asymptomatic)'),
            html.P('trestbps - resting blood pressure'),
            html.P('chol - cholesterol'),
            html.P('fbs - fasting blood sugar'),
            html.P('restcg - resting electrocardiographic results'),
            html.P('thalach - maximum heart rate achieved'),
            html.P('exang - exercise induced angina'),
            html.P('oldpeak - ST depression induced by exerices relative to rest'),
            html.P('slope - slope of the peak exercise ST segment'),
            html.P('ca - number of major vessels colored by flouroscopy'),
            html.P('thal - normal/fixed/reversable defect'),
            html.P('num - narrowing'),
            html.P('target - presence of heart disease')
        ], style={'display': 'inline-block'}),
html.Div([
            dcc.Graph(figure=fig1),
        ], style={'display': 'inline-block'}),
    ]),
    html.Div([
        html.H1('Odds of Heart Disease'),
        html.P('Select a variable and increase or decrease the slider to see the effect of incremental changes on the odds of having heart disease.'),
        html.P('NOTE: Blood pressure has a negative correlation in this dataset likely due to the number of patients with high BP who are otherwise healthy thus no disease.'),
        html.Div([
            dcc.RadioItems(
                id='feature',
                options=[{'label': i, 'value': i} for i in ['Age', 'Blood Pressure', 'Max Heart Rate']],
                value='Age',
                labelStyle={'display': 'inline-block'}
            ),
            daq.Slider(
                id='slider',
                min=29,
                max=220,
                handleLabel={'showCurrentValue': True, 'label': "VALUE"},
                value=heart['age'].min()
            ),
            html.Div(id='slider_output'),
        ]),
        html.Div([
            html.H1('Predictive Model', style={'textAlign': 'center'}),
            html.Div([dash_table.DataTable(
                id='table',
                columns=[{'name': i, 'id': i, 'type': 'numeric'} for i in dt.columns],
                data=pd.DataFrame(X_test).to_dict('records'),
                editable=True,
                filter_action='native',
                sort_action='native',
                sort_mode='multi',
                row_selectable='multi',
                selected_columns=[],
                selected_rows=[],
                style_table={
                    'height': '300px',
                    'overflowY': 'scroll'
                }
            ),
                html.P('Table can be sorted and filtered. Use arrows next to columns to sort. Type in filter boxes to filter.'), ]),
            html.Div(dcc.Graph(figure=plot_confusion_matrix())),
            html.P(u'Overall Accuracy : {}'.format(accuracy)),
            html.P(u'Sensitivity: {}'.format(sensitivity)),
            html.P(u'Specificity: {}'.format(specificity))
        ])
    ])
])

@app.callback(
    dash.dependencies.Output('table', 'style_data_conditional'),
    [dash.dependencies.Input('table', 'selected_columns')]
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

@app.callback(
    dash.dependencies.Output('slider_output', 'children'),
    [dash.dependencies.Input('slider', 'value'),
     dash.dependencies.Input('feature', 'value')])
def update_output(slider, radio):
    if radio == 'Age':
        coef = -.01266125
        dif = (int(slider) - 29)
        odds_ratio = math.exp(dif * coef)
        if (odds_ratio > 1):
            percent = (odds_ratio * 100) - 100
            return 'Odds of heart disease has increased by {}%'.format(percent)
        else:
            percent = 100 - (odds_ratio * 100)
            return 'Odds of heart disease has decreased by {}%'.format(percent)
    if radio == 'Blood Pressure':
        coef = -.01468438
        dif = (int(slider) - heart.trestbps.min())
        odds_ratio = math.exp(dif * coef)
        if (odds_ratio > 1):
            percent = (odds_ratio * 100) - 100
            return 'Odds of heart disease has increased by {}%'.format(percent)
        else:
            percent = 100 - (odds_ratio * 100)
            return 'Odds of heart disease has decreased by {}%'.format(percent)
    if radio == 'Max Heart Rate':
        coef = .02946909
        dif = (int(slider) - heart.thalach.min())
        odds_ratio = math.exp(dif * coef)
        if (odds_ratio > 1):
            percent = (odds_ratio * 100) - 100
            return 'Odds of heart disease has increased by {}%'.format(percent)
        else:
            percent = 100 - (odds_ratio * 100)
            return 'Odds of heart disease has decreased by {}%'.format(percent)


print(accuracy)
print(sensitivity)
print(specificity)


if __name__ == '__main__':
    app.run_server(debug=True)
