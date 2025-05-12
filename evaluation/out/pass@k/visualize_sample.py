import os
import json
import plotly.graph_objects as go
import plotly.express as px

if __name__ == '__main__':
    tasks = ['T1-T2-T4-rq1' ,'T4-T2-T1', 'T2-T4-T1', 'T2-T1-T4', 'T4-T1-T2', 'T1-T4-T2', 'T1-T2-T4']
    scores = [41.57, 40.75, 40.75, 38.5, 39.21, 39.21, 38.5]

    fig = go.Figure(px.bar(y=scores, x=tasks, orientation='v', text_auto=True, color=["red", "green", "green", "blue", "blue", "magenta", "magenta"]))
    fig.show()

