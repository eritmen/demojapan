from dash import Dash, Input, Output, State, html, dcc, dash_table, callback
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import json
import re

from src.helpers import build_quick_stats_panel, build_top_panel, build_chart_panel, generate_metric_row
from src.dataset import possible_parameter_df
from src.globals import TABS


line_color = "#1975FA"
pio.templates.default = "plotly_dark"


def _field(id, title):
    return dbc.Row(
        [
            dbc.Label(title, html_for=id, width=2),
            dbc.Col(
                dbc.Input(type="text", id=id, placeholder=title),
                width=8,
            ),
        ],
        className="mb-3",
    )

def _dropdown(id, title, options):
    return dbc.Row(
        [
            dbc.Label(title, html_for=id, width=2),
            dbc.Col(
                dcc.Dropdown(
                    id=id,
                    options=options,
                    placeholder=title,
                ),
                width=8,
            ),
        ],
        className="mb-3",
    )

def _checklist(id, title, options):
    return dbc.Row(
        [
            dbc.Label(title, html_for=id, width=2),
            dbc.Col(
                dcc.Checklist(
                    id=id,
                    options=options,
                    labelStyle={'display': 'inline-block', 'margin-right': "5px"}
                ),
                width=8,
            ),
        ],
        className="mb-3",
    )


class Tab():
    def __init__(self, config):
        self.config = config
        self.title = config.title

    def init_data(self, data):
        pass
    
    def render(self):
        raise NotImplementedError
    
    @classmethod
    def type(cls):
        return "base"
    
    @classmethod
    def type(cls):
        return "base"
    
    @classmethod
    def create_new(cls):
        pass

    @classmethod
    def return_state_ids(cls):
        return []
    
    @classmethod
    def render_settings(cls):
        return ()
    
    @classmethod
    def settings_callbacks(cls):
        return None


class GraphTab(Tab):
    def __init__(self, config):
        super().__init__(config)

    def init_data(self, data):
        # TODO: hard-coding this for now to prevent live queries
        df = pd.read_pickle("data/kale_cached.pkl")
        df['timestamp'] = df.timestamp.astype('object').astype('M8[ns]')
        df['timestamp'] = df['timestamp']  + np.timedelta64(51, 'Y') + np.timedelta64(7, 'M')
        self.df = df.sort_values(self.config.timestamp_col)
        # self.column_list = ["factoryId", "lineId", "machineId", "parameter"]
        self.params = possible_parameter_df(df, self.config.param_cols)
        
    @classmethod
    def type(cls):
        return "graph"

    @classmethod
    def render_settings(cls):
        id = f"{cls.type()}-tab"
        with open('data/kale_seramik_table_schemas.json') as f:  #TODO: get this from the database rather than file
            table_schema = json.load(f)
        tables = list(table_schema.keys())
        form = dbc.Form([
            _field(f"input-{id}-1", "Tab title"),
            _dropdown(f"input-{id}-2", "Table", options=tables),
            _dropdown(f"input-{id}-3", "Timestamp column", options=["col1", "col2"]),
            _dropdown(f"input-{id}-4", "Value column", options=["col1", "col2"]),
            _checklist(f"input-{id}-5", "Parameter columns", options=["Param1", "Param2", "Param3", "Param4", "Param5"]),
            _field(f"input-{id}-6", "Parameter picker regex"),
        ])
        return form
    
    @classmethod
    def settings_callbacks(cls):
        with open('data/kale_seramik_table_schemas.json') as f:  #TODO: get this from the database rather than file
            table_schema = json.load(f)
        def dropdown_callback(table):
            options = [row["name"] for row in table_schema[table]]
            return (options, options, options)
        id = f"{cls.type()}-tab"
        return (
            [
                Output(f"input-{id}-3", "options"),
                Output(f"input-{id}-4", "options"),
                Output(f"input-{id}-5", "options"),
            ],
            [Input(f"input-{id}-2", "value")],
            dropdown_callback,
        )

    @classmethod
    def return_state_ids(cls):
        return [f"input-{cls.type()}-tab-{i+1}" for i in range(6)]
    
    @classmethod
    def create_new(cls, set_btn, *inputs):
        title, table_name, timestamp_col, value_col, param_cols, param_regex = inputs
        #TODO write a proper config with these
        config = OmegaConf.create({
            "type": cls.type(),
            "title": title,
            "table": table_name,
            "timestamp_col": timestamp_col,
            "value_col": value_col,
            "param_cols": param_cols,
            "param_regex": param_regex,
        })
        return cls(config)

    def render(self, stopped_interval):
        parameter_names = []
        data = []
        for param_list in self.params:
            param_name = "_".join(param_list)
            if self.config.param_regex and not re.search(self.config.param_regex, param_name, flags=re.IGNORECASE):
                continue
            query = " & ".join([f'{name}=="{value}"' for name, value in zip(self.config.param_cols, param_list)])
            df_chart = self.df.query(query).copy()
            try:
                df_chart = df_chart.astype({'value': 'float'})
            except ValueError:
                continue
            df_chart[df_chart['value'].notnull()]
            if len(df_chart) < 5:
                continue
            parameter_names.append(param_list)
            data.append(df_chart)

        return (
            html.Div(
                id="graph-rows",
                children=[self.generate_metric_row_helper(stopped_interval, params, df) for params, df in zip(parameter_names, data)],
            ),
            stopped_interval,
        )
    
    def create_figure(self, column_values, df_chart):
        parameter_id = "_".join(column_values)
        layout = dict(
            showlegend=False,
            uirevision=True,
            margin=dict(l=0, r=0, t=4, b=4, pad=0),
            xaxis=dict(
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=True,
            ),
            yaxis=dict(
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",   
            title="",
            xaxis_title="",
            yaxis_title="",
        )
        
        fig = px.scatter(
            df_chart,
            x='timestamp',
            y='value',
            marginal_y="violin",
        )
        fig.update_traces(marker=dict(size=10, color=line_color), selector=dict(mode='markers'))
        fig.update_layout(**layout)
        fig.add_trace(go.Scatter(x=df_chart['timestamp'].values, y=df_chart['value'].values,
                                 mode='lines', line=dict(color=line_color, width=3)))
        min_limit = np.min(df_chart['value'].values)
        max_limit = np.max(df_chart['value'].values)
        spread = max_limit - min_limit
        min_limit = min_limit + 0.15 * spread
        max_limit = max_limit - 0.15 * spread
        y_min =min_limit*np.ones(len(df_chart))
        y_max =max_limit*np.ones(len(df_chart))
        
        # fig.add_trace(go.Scatter(x=df_chart['timestamp'].values, y=y_min, mode='lines', line=dict(color='red')))
        # fig.add_trace(go.Scatter(x=df_chart['timestamp'].values, y=y_max, mode='lines', line=dict(color='red')))
        #fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(**layout)
        return fig

    
    def generate_metric_row_helper(self, stopped_interval, parameter_names, data):
        suffix_row = "_row"
        suffix_button_id = "_button"
        suffix_sparkline_graph = "_sparkline_graph"
        suffix_count = "_count"
        suffix_ooc_n = "_OOC_number"
        suffix_ooc_g = "_OOC_graph"
        suffix_indicator = "_indicator"

        item = "_".join(parameter_names)
        div_id = item + suffix_row
        button_id = item + suffix_button_id
        sparkline_graph_id = item + suffix_sparkline_graph
        count_id = item + suffix_count
        ooc_percentage_id = item + suffix_ooc_n
        ooc_graph_id = item + suffix_ooc_g
        indicator_id = item + suffix_indicator

        min_limit = np.min(data['value'].values)
        max_limit = np.max(data['value'].values)
        spread = max_limit - min_limit
        percent = 100 * (data['value'].values[-1] - min_limit) / spread if spread > 0 else 0.5

        return generate_metric_row(
            div_id,
            None,
            {
                "id": item,
                "className": "metric-row-button-text",
                "children": html.Button(
                    id=button_id,
                    className="metric-row-button",
                    children=item,
                    title="Click to visualize live SPC chart",
                    n_clicks=0,
                ),
            },
            {"id": count_id, "children": str(len(data))},
            {
                "id": item + "_sparkline",
                "children": dcc.Graph(
                    id=sparkline_graph_id,
                    style={"width": "100%", "height": "95%"},
                    config={
                        "staticPlot": False,
                        "editable": False,
                        "displayModeBar": False,
                    },
                    figure=self.create_figure(parameter_names, data),
                ),
            },
            {"id": ooc_percentage_id, "children": f"{percent:.2f}%"},
            {
                "id": ooc_graph_id + "_container",
                "children": daq.GraduatedBar(
                    id=ooc_graph_id,
                    color={
                        "gradient":True,
                        "ranges": {
                            "#92e0d3": [0, 3],
                            "#f4d44d ": [3, 6],
                            "#f45060": [6, 10],
                        }
                    },
                    showCurrentValue=False,
                    max=10,
                    value=percent/10,
                    className="graduatedBar",
                ),
            },
            {
                "id": item + "_pf",
                "children": daq.Indicator(
                    id=indicator_id, value=True, color="#91dfd2", size=12
                ),
            },
        )
    
    @classmethod
    def type(cls):
        return "graph"


class MetricTab(Tab):
    def __init__(self, config):
        super().__init__(config)

    def init_data(self, data):
        self.df = data
        self.params = list(self.df)

    def render(self, stopped_interval):
        return (
            html.Div(
                id="status-container",
                children=[
                    build_quick_stats_panel(len(self.params)),
                    html.Div(
                        id="graphs-container",
                        children=[
                            build_top_panel(stopped_interval, self.params, self.df),
                            build_chart_panel(self.params),
                        ],
                    ),
                ],
            ),
            stopped_interval,
        )
    
    @classmethod
    def type(cls):
        return "metric"
    
    @classmethod
    def render_settings(cls):
        id = f"{cls.type()}-tab"
        return dbc.Form([
            _field(f"input-{id}-1", "Tab title"),
            _field(f"input-{id}-2", "Table name"),
            _field(f"input-{id}-3", "Metric name"),
            _field(f"input-{id}-4", "SQL Expression"),
        ])
    
    @classmethod
    def return_state_ids(cls):
        return [f"input-{cls.type()}-tab-{i+1}" for i in range(4)]
    
    @classmethod
    def create_new(cls, set_btn, metrics_tab_1, metrics_tab_2, metrics_tab_3, metrics_tab_4, **args):
        title, table_name, metric_name, sql = metrics_tab_1, metrics_tab_2, metrics_tab_3, metrics_tab_4
        #TODO write a proper config with these
        config = OmegaConf.create({"type": cls.type(), "title": title})
        return cls(config)
    

class EnergyTab(GraphTab):
    def __init__(self, config):
        super().__init__(config)
        
    @classmethod
    def type(cls):
        return "energy"

    def init_data(self, data):
        # TODO: hard-coding this for now to prevent live queries
        df = pd.read_csv('data/Brulor_efficiency_v3.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.iloc[-1000:,:]
        messages = ['Reduce parameter 4 by %5', 'Check parameter 2', 'Fan is not working properly',
            'Reduce parameter 6 by %10', 'Check paramter 2', 'Increase parameter 1 by 5']
        type_list =[]
        size_list = []
        message_list = []
        for i in range(len(df)):
            if df['True_class'].iloc[i] > 1.2* df['Predicted_class'].iloc[i]:
                type_list.append('Anomaly')
                size_list.append((df['True_class'].iloc[i])/
                                 df['Predicted_class'].iloc[i])
                ind = np.random.randint(len(messages))
                message_list.append(messages[ind])
            else:
                type_list.append('Normal')
                size_list.append(0.2)
                message_list.append('-')
        
        df['Type'] = type_list
        df['Size'] = size_list
        df['Messages'] = message_list
        self.df = df
        self.params = list(df)

    def fill_gauge(self, text, percentage):
        return daq.Gauge(
            size=240,
            max=100,
            min=0,
            #label = text,
            units="%",
            color='red', #{"gradient":True,"ranges":{"blue":[0,10],"red":[10,100]}},
            value = percentage,
            showCurrentValue=True,
            style = {
                    "align": "center",
                    "display": "flex",
                    "marginTop": "0%",
                    "marginBottom": "0%",
                    'font':'12'
                },
        )

    def create_scatter(self):
        df = self.df
        fig = px.scatter(df, x="Predicted_class", y="True_class", color="Type",
                 size='Size', hover_data= ['Messages'], color_discrete_sequence=[line_color, '#C1121F'], height = 750)
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
                
        fig.update_xaxes(range=[3.5, 4.25])
        fig.update_yaxes(range=[3.5, 5.5])
        if df['Predicted_class'].min() < df['True_class'].min():
            min_value = df['Predicted_class'].min() 
        else:
            min_value = df['True_class'].min()
        if df['Predicted_class'].max() > df['True_class'].max():
            max_value = df['Predicted_class'].max() + 0.1
        else:
            max_value = df['Predicted_class'].max() + 0.1
        fig.add_trace(
            go.Scatter(
                x=np.arange(min_value, max_value, 0.1),
                y=np.arange(min_value, max_value, 0.1),
                name="Ideal Process",
                line_color = '#FDF0D5',
                mode='lines',
                line = dict(shape = 'linear', dash = 'dash')
            )
        )
        fig.update_layout(
            title="Energy Diagnostics Chart",
            xaxis_title="Values to Be (cubic meter)",
            yaxis_title="Actual Values (cubic meter)",
            legend_title="Consumption Types",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            #line_color = "#f4d44d",
        )
        return fig

    def create_time_plot(self, start_date, end_date):
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        df_line = self.df[self.df['timestamp'].between(start_date.tz_localize('utc'), end_date.tz_localize('utc'))]
        fig_line = px.line(df_line, x='timestamp', y='True_class',
              title="Time-series Anomaly Chart",height = 650)
        fig_line.update_traces(line_color=line_color, line_width=3)
        
        df_line_high = df_line[df_line['Type'] == 'Anomaly']
        fig_line.add_trace(go.Scatter(x=df_line_high['timestamp'].values, y=df_line_high['True_class'].values,
                    mode='markers', name='Anomalies', line=dict(color="red", width=5)))
        fig_line.update_layout(
            #title="Plot Title",
            xaxis_title="Time",
            yaxis_title="Energy Consumption (cubic meter)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                yanchor="bottom",
                y=0.05,
                xanchor="right",
                x=0.99
            )
        )
        return fig_line

    def create_metrics(self):
        df = self.df
        anomaly_ratio = int((len(df[df['Type'] == 'Anomaly'])/len(df))*100)
        fig_anomaly_ratio = self.fill_gauge('Anomaly Ratio', anomaly_ratio)
        extra_consumption = int((df['True_class'].sum()-df['Predicted_class'].sum())
                                /df['Predicted_class'].sum()*100)
        fig_anomaly_con= self.fill_gauge('Saveable Energy Consumption', extra_consumption)
        amount_saved = int((df['True_class'].sum()-df['Predicted_class'].sum())*10/12)
        return fig_anomaly_ratio, fig_anomaly_con, amount_saved
    
    def render(self, stopped_interval):
        fig_anomaly_ratio, fig_anomaly_con, amount_saved = self.create_metrics()
        energy_text = dcc.Markdown(f"### {amount_saved}$ can be saved per day.")
        return (
            html.Div(
                id="energy-container", children=[
                    html.Div(children=[
                        html.Div(
                            children=[dcc.Graph(id="energy-scatter-graph", figure=self.create_scatter())],
                            className = 'twelve columns',
                            style={'display': 'inline-block'}
                        ),
                    html.Div([  
                        html.Div(className='row', children=[
                                html.Div(
                                    children=[
                                        html.H4("Anomaly Percentage"),
                                        html.Div(id="anomaly-ratio-gauge", children=[fig_anomaly_ratio],
                                                 style={'display': 'inline-block', 'justify':"center", 'align':"center"}),
                                        #html.Small("Saveable Energy"),
                                        #html.Div(id="consumption-ratio-gauge",children=fig_anomaly_con),
                                        #html.Div(id='text-energy-saving', children=energy_text),
                                    ],                                 
                                    className='four columns',
                                    style={'display': 'inline-block', 'justify':"center", 'align':"center"},
                                ),
                                html.Div(
                                    children=[
                                        #html.Small("Anomaly Percentage"),
                                        #html.Div(id="anomaly-ratio-gauge", children=fig_anomaly_ratio),
                                        html.H4("Saveable Energy"),
                                        html.Div(id="consumption-ratio-gauge",children=[fig_anomaly_con],
                                                 style={'display': 'inline-block', 'justify':"center", 'align':"center"}),
                                    ],                                 
                                    className='four columns',
                                    style={'display': 'inline-block', 'justify':"center", 'align':"center"},
                                ),
                                html.Div(id='text-energy-saving', children=
                                         [html.Br(),html.Br(),html.Br(),
                                          html.Br(),html.Br(),
                                          html.Br(),energy_text],
                                         className='four columns',style={"color": "red"}),

                            ]),
                        ])],
                        style={'textAlign': 'center'}
                    ),
                    html.Div(children=[
                        dcc.Graph(id="energy-line-graph", figure=self.create_time_plot("2022-07-22", "2022-07-23")),
                    ]),
                ],
            ),
            stopped_interval,
        )
class AddTab(Tab):
    def __init__(self, config):
        super().__init__(config)

    def init_data(self, data):
        pass

    def render(self, stopped_interval):
        form = html.Div(
            className="container px-4",
            children=[
                dbc.Row([
                    dbc.Col(
                        [
                            dbc.Label("Select Dashboard", html_for="tab-select-dropdown"),
                            dcc.Dropdown(
                                    id="tab-select-dropdown",
                                    options=[{"label": " ", "value": "none"}] + list(
                                        {"label": param.upper(), "value": param.lower()} for param in CONTENT_TAB_TYPES.keys()
                                    ),
                                    value="none",
                                ),
                        ],
                        width=7,
                        className="p-3",
                    ),
                    dbc.Col(
                        [
                            dbc.Label(" ", html_for="tab-creator-btn", style={'white-space': 'pre'}),
                            html.Button("CREATE", id="tab-creator-btn"),
                        ],
                        width=3,
                        className="p-3",
                    ),
                ],
                className="g-3",
                ), 
            html.Div(id="tab-creator-panel"),
            ]
        )
        return form, stopped_interval

    @classmethod
    def type(cls):
        return "add"


class SettingTab(Tab):
    def __init__(self, config):
        super().__init__(config)

    def init_data(self, data):
        pass

    def render(self, stopped_interval):
        return (
            html.Div([
                html.Br(),
                dcc.Upload(
                    id='upload-json-file',
                    children=html.Div([
                        html.A('Upload Credential File')
                    ])
                    ),
                html.Div(id='output-table-names'),
                html.Br(),
                html.Button("Generate New Tab", id="add-tab-button"),
            ]),
            stopped_interval,
        )
    
    @classmethod
    def type(cls):
        return "setting"


#################  Helpers  ###############################
def build_tabs(tabs):
    children = [dcc.Tab(
            id=f"Content-{key}",
            label=tab.title,
            value=key,
            className="custom-tab",
            selected_className="custom-tab--selected",
    ) for key, tab in tabs.items() if key not in ["tab+", "tab-s"]]
    children += [
        dcc.Tab(
            id=f"Add-tab",
            label="➕",
            value=f"tab+",
            className="custom-tab",
            selected_className="custom-tab--selected",
        ),
        dcc.Tab(
            id=f"Settings-tab",
            label="⚙",
            value=f"tab-s",
            className="custom-tab",
            selected_className="custom-tab--selected",
        ),
    ]
    return children
    

CONTENT_TAB_TYPES = {cls.type(): cls for cls in [GraphTab, MetricTab, EnergyTab]}

TAB_TYPES = CONTENT_TAB_TYPES.copy()
TAB_TYPES.update({
    AddTab.type(): AddTab,
    SettingTab.type(): SettingTab,
})
