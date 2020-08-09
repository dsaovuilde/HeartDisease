import dash
import pandas as pd
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

USERNAME_PASSWORD_PAIRS = {'test': 'test'}
app = dash.Dash(__name__, suppress_callback_exceptions=True)
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
server = app.server

# download CSV file from gitHub
url = 'https://raw.githubusercontent.com/dsaovuilde/HeartDisease/master/Data/Heart.csv'
heart = pd.read_csv(url)


#train_test split to split the dataframe into test data and train data for prediction models
y = pd.DataFrame(heart['target'])
X = heart.drop('target', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(heart.drop('target',axis=1),y, test_size=0.2, random_state=1)
X_test2['target']= y_test2
rf = RandomForestClassifier(n_estimators=644, min_samples_split=10, min_samples_leaf=1, max_features='sqrt', max_depth=40, bootstrap=False, random_state=4)
rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
accuracy = accuracy_score(y_test, prediction)


y_test2['predicted'] = prediction
X_test2['predicted'] = prediction

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
cp1 = heart[heart['cp']==1]
cp1_pos = cp1[cp1['target']==1].shape[0]
cp2 = heart[heart['cp']==2]
cp2_pos = cp2[cp2['target']==1].shape[0]
cp3 = heart[heart['cp']==3]
cp3_pos = cp3[cp3['target']==1].shape[0]
cp0 = heart[heart['cp']==0]
cp0_pos = cp0[cp0['target']==1].shape[0]

odds = [cp0_pos/cp0.shape[0], cp1_pos/cp1.shape[0], cp2_pos/cp2.shape[0], cp3_pos/cp3.shape[0]]

cols = list(heart.columns.values)
cols.remove('target')
lr = LogisticRegression(random_state=0).fit(X,y)


heatmap = go.Heatmap(
   x = list(heart.columns.values),
   y = list(heart.columns.values),
   z = heart.corr(),
   type = 'heatmap',
   colorscale = 'Viridis',
)

data = [heatmap]
for col in heart.columns:
    num = heart[col].max() - heart[col].min()


fig1 = go.Figure([go.Bar(x=cols, y=lr.coef_[0])])
fig1.update_layout(title='coefficients')
fig1.update_layout(xaxis_title='Intercept = .12508204')

fig2 = go.Figure(data=data,
                            layout=go.Layout(
        title=go.layout.Title(text='Heatmap')
                            )
                        )

fig3= go.Figure(go.Bar(
            x=odds,
            y=['asymptomatic','typical angina','atypical angina','non-anginal pain'],
            orientation='h'
        ))
fig3.update_layout(
            title='Odds By Type of Chest Pain',
            xaxis_title='Odds of Heart Disease',
            yaxis_title='Type of Chest Pain'
        )


app.layout = html.Div([
    html.Div([
        html.H1('Descriptive Model', style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(
            figure=(fig2)
        )
    ], style={'display': 'inline-block'}),
        html.Div([
        dcc.Graph(figure=fig1),
    ], style={'display': 'inline-block'}),
        html.Div([
        dcc.Graph(figure=fig3),
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
    html.P('Table can be sorted and filtered. To filter search for strings rather than numbers. (\'6.2\' instead of 6.2)'),]),
    html.Div(dcc.Graph(figure=plot_confusion_matrix())),
    html.P(u'Overall Accuracy : {}'.format(accuracy)),
    html.P(u'Sensitivity: {}'.format(sensitivity)),
    html.P(u'Specificity: {}'.format(specificity))
])
])


@app.callback(Output('table', 'children'),
              [Input('table', 'selected_columns')]
              )
def update_table(selected_columns):
    return [{
        'if': {'column_id': i},
        'background_color': '#D2F3FF'
    } for i in selected_columns]


if __name__ == '__main__':
    app.run_server(debug=True)

