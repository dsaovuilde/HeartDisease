import dash
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(heart.drop('target', axis=1), y, test_size=0.2, random_state=1)
X_test2['target'] = y_test2
rf = RandomForestClassifier(n_estimators=644, min_samples_split=10, min_samples_leaf=1, max_features='sqrt',
                            max_depth=40, bootstrap=False, random_state=4)
rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

y_test2['predicted'] = prediction
X_test2['predicted'] = prediction

# calculate sensitivity and specificity
tp_filter = (y_test2['predicted'] == 1) & (y_test2['target'] == 1)
true_positives = len(y_test[tp_filter])
tn_filter = (y_test2['predicted'] == 0) & (y_test2['target'] == 0)
true_negatives = len(y_test[tn_filter])
fp_filter = (y_test2['predicted'] == 1) & (y_test2['target'] == 0)
false_positives = len(y_test[fp_filter])
fn_filter = (y_test2['predicted'] == 0) & (y_test2['target'] == 1)
false_negatives = len(y_test[fn_filter])
sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)


def plot_confusion_matrix():
    cm = confusion_matrix(y_test, prediction)
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


# Graph showing presence of chest pain and odds of heart disease
cp1 = heart[heart['cp'] == 1]
cp1_pos = cp1[cp1['target'] == 1].shape[0]
cp2 = heart[heart['cp'] == 2]
cp2_pos = cp2[cp2['target'] == 1].shape[0]
cp3 = heart[heart['cp'] == 3]
cp3_pos = cp3[cp3['target'] == 1].shape[0]
cp0 = heart[heart['cp'] == 0]
cp0_pos = cp0[cp0['target'] == 1].shape[0]

odds = [cp0_pos / cp0.shape[0], cp1_pos / cp1.shape[0], cp2_pos / cp2.shape[0], cp3_pos / cp3.shape[0]]

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
    ]),
    html.Div([
        dcc.Graph(figure=fig1),
    ]),
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
            columns=[{'name': i, 'id': i} for i in X_test2.columns],
            data=pd.DataFrame(X_test2).to_dict('records'),
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
            html.P(
                'Table can be sorted and filtered. To filter search for strings rather than numbers. (\'6.2\' instead of 6.2)'), ]),
        html.Div(dcc.Graph(figure=plot_confusion_matrix())),
        html.P(u'Overall Accuracy : {}'.format(accuracy)),
        html.P(u'Sensitivity: {}'.format(sensitivity)),
        html.P(u'Specificity: {}'.format(specificity))
    ])
])


@app.callback(
    dash.dependencies.Output('slider_output', 'children'),
    [dash.dependencies.Input('slider', 'value'),
     dash.dependencies.Input('feature', 'value')])
def update_output(slider, radio):
    if radio == 'Age':
        coef = -.05395622
        dif = (int(slider) - 29) / float(heart.age.std())
        odds_ratio = math.exp(dif * coef)
        if (odds_ratio > 1):
            percent = (odds_ratio * 100) - 100
            return 'Odds of heart disease has increased by {}%'.format(percent)
        else:
            percent = 100 - (odds_ratio * 100)
            return 'Odds of heart disease has decreased by {}%'.format(percent)
    if radio == 'Blood Pressure':
        coef = -.3189122
        dif = (int(slider) - heart.trestbps.min()) / float(heart.trestbps.std())
        odds_ratio = math.exp(dif * coef)
        if (odds_ratio > 1):
            percent = (odds_ratio * 100) - 100
            return 'Odds of heart disease has increased by {}%'.format(percent)
        else:
            percent = 100 - (odds_ratio * 100)
            return 'Odds of heart disease has decreased by {}%'.format(percent)
    if radio == 'Max Heart Rate':
        coef = .5045519
        dif = (int(slider) - heart.thalach.min()) / float(heart.thalach.std())
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
