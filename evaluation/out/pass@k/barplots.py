import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

rq3_pass1 = 'correct path to bigcode-evaluation-harness/out/pass@k/rq-3-pass@1.csv'
rq1_pass1 = 'correct path to bigcode-evaluation-harness/out/pass@k/rq-1-pass1_table.csv'

apr_pass1_starcoder = 28.66
apr_pass10_starcoder = 39.11
apr_pass1_granite = 16.16
apr_pass10_granite = 19.84

rq1_data = pd.read_csv(rq1_pass1)
rq3_data = pd.read_csv(rq3_pass1)

models = ['starcoder2-3b', 'granite-3b-code-base']
methods = ['dare-ties', 'ties', 'weight-averaging']

def type1_plots():
    # all rq3 vs. ap
    # group by constituent tasks and plot a bar plot for each group
    grouped = rq3_data.groupby('constituent_tasks')

    for group_name, group_df in grouped:
        for model in models:
            if model == 'starcoder2-3b':
                continue

            for method in methods:                
                target_column = model + '-' + method
                group_df['value'] = group_df[target_column].apply(lambda x: float(x.split('%')[0]))

                string_to_num = {s: i for i, s in enumerate(group_df['tasks'])}
                color_values = [string_to_num[s] for s in group_df['tasks']]

                # find the corresponding task in the rq1 data
                rq1_task = rq1_data.loc[rq1_data['tasks'] == group_name]
                rq1_x = rq1_task['tasks'].values[0]
                rq1_y = rq1_task[target_column].values[0]

                fig = go.Figure(go.Bar(x=group_df['tasks'], y=group_df['value'],
                            marker_color=list(reversed(color_values)),
                            marker=dict(
                                colorscale='Viridis',
                        )))

                fig.add_trace(go.Bar(x=[rq1_x + ' (RQ1)'], y=[float(rq1_y.split('%')[0])],
                            name='equal-weight merging',
                            marker_color='red',
                            showlegend=False,
                            # text=rq1_y,
                            # textposition='outside',
                        ))

                fig.add_trace(go.Bar(x=['APR'], y=[apr_pass1_granite],
                            name='APR',
                            marker_color='blue',
                            showlegend=False,
                            # text=str(apr_pass1_starcoder),
                            # textposition='outside',
                        ))

                threshold_value = None
                if model == 'starcoder2-3b':
                    threshold_value = apr_pass1_starcoder
                elif model == 'granite-3b-code-base':
                    threshold_value = apr_pass1_granite
                fig.add_hline(y=threshold_value, line=dict(color='blue', width=2, dash='dash'))
                fig.add_hline(y=float(rq1_y.split('%')[0]), line=dict(color='red', width=2, dash='dash'))
                
                if len(group_df['tasks'].iloc[0].split('-')) == 3:
                    fig.update_layout(
                        title=f"{group_name.replace('-', ',')} with {method}",
                        font=dict(
                            size=16  # Set your desired font size here
                        ),
                        xaxis_title='Tasks',
                        yaxis_title='Pass@1 (%)',
                        showlegend=False,   
                        bargap=0.5,
                        width=400,
                    )
                else:
                    fig.update_layout(
                        title=f"{group_name.replace('-', ',')} with {method}",
                        font=dict(
                            size=18  # Set your desired font size here
                        ),
                        xaxis_title='Tasks',
                        yaxis_title='Pass@1 (%)',
                        showlegend=False,   
                        bargap=0,
                        bargroupgap=0,
                        width=800,
                    )
                
                fig.update_xaxes(tickangle=90)
                fig.show()

def type2_plots():
    # rq3 vs. rq1 vs. apr
    # select the best performing adapter of each group for rq3 and compare with rq1
    grouped = rq3_data.groupby('constituent_tasks')
    
    for group_name, group_df in grouped:
        for model in models:
            if model == 'granite-3b-code-base':
                continue

            for method in methods:
                target_column = model + '-' + method
                group_df['value'] = group_df[target_column].apply(lambda x: float(x.split('%')[0]))
                
                # Select the best performing adapter for each group
                best_performing_adapter = group_df.loc[group_df['value'].idxmax()]
                print(best_performing_adapter)

                string_to_num = {s: i for i, s in enumerate(group_df['tasks'])}
                color_values = [string_to_num[s] for s in group_df['tasks']]



type1_plots()