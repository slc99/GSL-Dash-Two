from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import csv
import dataretrieval.nwis as nwis
from datetime import date
import numpy as np
from dateutil.relativedelta import relativedelta
import plotly.express as px
from dash_extensions import BeforeAfter
import plotly.graph_objects as go

class DataSchema:
    streamflow_after_consumption = 'streamflow_after_consumption'
    groundwater = 'groundwater'
    evap_rate = 'evap_rate'
    percip = 'percip'
    total_consumptive_use = 'total_consumptive_use'
    streamflow_before_consumption = 'streamflow_before_consumption'
    agriculture_consumption = 'agriculture_consumption'
    municipal_consumption = 'municipal_consumption'
    mineral_consumption =  'mineral_consumption'
    impounded_wetland_consumption = 'impounded_wetland_consumption'
    reservoir_consumption =  'reservoir_consumption'

class Policy:

    all_policies = []
    slider_policies = []

    def __init__(self, title: str, description: str, affected: str, 
        affect_type: str, delta: float, initial_cost = 0, cost_per_year = 0, 
        slider = '', slider_message = '',):
        
        # assert affect_type is 'proportion' or 'absolute', f'Affect type: {affect_type}. Must be proportion or absolute.'
        # assert affected in DataSchema.__dict__, f'Affected: {affected} not in {DataSchema.__dict__}'

        if affect_type == 'proportion':
            assert delta >= 0, f'Delta: {delta}. Delta must be above 0 '

        self.title = title
        self.description = description
        self.affected = affected
        self.affect_type = affect_type
        self.delta = delta
        self.initial_cost = initial_cost
        self.cost_per_year = cost_per_year
        self.slider = slider
        self.slider_message = slider_message
        self.first_thirty_cost_in_millions = int((initial_cost + (cost_per_year * 30))/1000000)
        self.checklist_label = html.Span(children=[html.Strong(self.title), html.Br() ,self.description, html.Br()])
        self.id_name = title.replace(' ','-').lower()

        self.checklist_component = dcc.Checklist(
                id=self.id_name + '-checklist',
                options = [{'label': self.checklist_label, 'value': self.title}],
                value = [],
                labelStyle={'display': 'block','text-indent': '-1.25em'},
                style={'position':'relative','left': '1em'}
            )
        self.slider_component = html.Span(
                id=self.id_name + '-display',
                children = [
                    html.I(self.slider_message,style={'position':'relative','left': '1em'}),
                    html.Br(),
                    dcc.Slider(
                    id = self.id_name + '-slider',
                    min = 0,
                    max = self.first_thirty_cost_in_millions,
                    value = 0,
                    marks = {
                        0: '$0', 
                        int(self.first_thirty_cost_in_millions/2): f'${self.first_thirty_cost_in_millions/2} million', 
                        int(self.first_thirty_cost_in_millions): {'label':f'${self.first_thirty_cost_in_millions} million', 'style':{'right':'-200px'}}
                    },
                    tooltip={'placement':'bottom'}), 
                ], 
            style={'display': 'none'}
        )

        Policy.all_policies.append(self)

        if self.slider:
            Policy.slider_policies.append(self)


    def __repr__(self):
        return f'{self.title}, {self.description}, {self.affected}, {self.affect_type}, {self.delta}'

    @classmethod
    def instantiate_from_csv(cls, path: str):
        with open(path,'r') as f:
            reader = csv.DictReader(f)
            policies = list(reader)
        for policy in policies:
            Policy(
                title = policy.get('Title'),
                description = policy.get('Description'),
                affected = policy.get('Affected Variable'),
                affect_type = policy.get('Affect Type'),
                delta = float(policy.get('Delta per Year')),
                initial_cost = float(policy.get('Cost to Implement')),
                cost_per_year = float(policy.get('Cost per Year')),
                slider = bool(policy.get('Slider')),
                slider_message = policy.get('Slider Message'),
            )

def LoadMonthlyStats(path: str) -> pd.DataFrame:
    '''
    Loads monthly statistics as a dataframe
    '''
    data = pd.read_csv(path,index_col=0,)
    data.rename(columns={
        'Streamflow': DataSchema.streamflow_after_consumption,
        'Groundwater': DataSchema.groundwater,
        'Evap (km3/km2)': DataSchema.evap_rate,
        'Percipitation (mm)': DataSchema.percip, 
        'Consumptive use': DataSchema.total_consumptive_use,
        'Streamflow without consumption': DataSchema.streamflow_before_consumption,
        'agriculture consumption': DataSchema.agriculture_consumption,
        'municipal consumption': DataSchema.municipal_consumption,
        'reservoir consumption': DataSchema.reservoir_consumption,
        'mineral consumption': DataSchema.mineral_consumption,
        'wetland consumption': DataSchema.impounded_wetland_consumption,
    },inplace=True)
    return data

def AdjustMonthlyStats(policies: list, monthly_stats: pd.DataFrame) -> pd.DataFrame:
    '''
    Adjusts the monthly variables by the given policies
    '''
    adjusted_monthly = monthly_stats.copy(deep=True)  

    for policy in policies:
        if policy.affect_type == 'proportion':
            adjusted_monthly[policy.affected] *= policy.delta
    for policy in policies:
        if policy.affect_type == 'absolute':
            adjusted_monthly[policy.affected] += policy.delta
            adjusted_monthly[policy.affected].clip(0,inplace=True)
    
    adjusted_monthly[DataSchema.total_consumptive_use] = (
        adjusted_monthly[DataSchema.agriculture_consumption] 
        + adjusted_monthly[DataSchema.municipal_consumption]
        + adjusted_monthly[DataSchema.reservoir_consumption]
        + adjusted_monthly[DataSchema.mineral_consumption]
        + adjusted_monthly[DataSchema.impounded_wetland_consumption]
    )

    for policy in policies:
        if policy.affected == DataSchema.total_consumptive_use:
            if policy.affect_type == 'proportion':
                adjusted_monthly[policy.affected] *= policy.delta
            if policy.affect_type == 'absolute':
                adjusted_monthly[policy.affected] += policy.delta
                adjusted_monthly[policy.affected].clip(0,inplace=True)

    adjusted_monthly[DataSchema.streamflow_after_consumption] = (
        adjusted_monthly[DataSchema.streamflow_before_consumption] - adjusted_monthly[DataSchema.total_consumptive_use])
    
    return adjusted_monthly

def GetUSGSSiteData(site_num: str, start_date: str, end_date: str, service='dv') -> pd.DataFrame:
    '''
    Uses dataretrieval.nwis to load the record of the given site
    '''
    df = nwis.get_record(sites=site_num, service=service ,start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

def CreateLakeData(path: str, bath_df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function loads the historic data from the saved cvs and then updates with new data if required
    '''
    today = date.today().strftime("%Y-%m-%d")

    saved_data = pd.read_csv(path, index_col=0)
    last_saved_day = saved_data.at[len(saved_data)-1, 'datetime']

    if saved_data.at[len(saved_data)-1, 'datetime'].split('-')[1] == today.split('-')[1]:
        return saved_data

    today = date.today().strftime("%Y-%m-%d")
    FOOT_TO_METER = 0.3048
    
    new_data = GetUSGSSiteData('10010024',last_saved_day,today)

    new_data['YYYY-MM'] = new_data['datetime'].dt.year.astype(str) + '-' + new_data['datetime'].dt.month.astype(str) #the better way
    new_data = new_data.groupby(new_data['YYYY-MM']).mean(numeric_only=True)
    new_data = new_data.iloc[1:] 

    new_data.rename(columns={'62614_Mean':'Elevation'},inplace=True)
    new_data['Elevation'] *= FOOT_TO_METER
    new_data.reset_index(inplace=True)
    new_data['datetime'] = pd.to_datetime(new_data['YYYY-MM'])

    for index, row in new_data.iterrows():
        round_e = round(new_data.at[index,'Elevation'], 2)
        new_data.at[index,'Surface Area'] = bath_df.at[round_e,'Surface Area']
        new_data.at[index,'Volume'] = bath_df.at[round_e,'Volume']

    combined = pd.concat([saved_data,new_data],ignore_index=True)
    combined.reset_index(inplace=True,drop=True)

    return combined

def GSLPredictor(years_forward: int, monthly_stats: pd.DataFrame,
                    bath_df: pd.DataFrame, lake_df: pd.DataFrame, start_date='',
                    weather=False):
    
    MAX_VOLUME = 39.896064
    
    if len(start_date) == 0:
        yyyy_mm_date = date.today().strftime("%Y-%m")  
    else:
        yyyy_mm_date = start_date
        
    cur_date = pd.to_datetime(yyyy_mm_date)

    months_forward = 12 * years_forward
    
    elevation = round(lake_df.set_index('YYYY-MM').at[yyyy_mm_date,'Elevation'], 2)
    predictions = pd.DataFrame(columns=['datetime','Elevation Prediction','YYYY-MM'])
    
    weather_factor = 0
    lr_weather_counter = 0
    
    for i in range(months_forward):
        surface_area = bath_df.at[elevation,'Surface Area']
        volume = bath_df.at[elevation,'Volume']
        month_index = (i%12)
        
        lost_evap = surface_area * monthly_stats.at[month_index,DataSchema.evap_rate]
        gained_rain = surface_area * (monthly_stats.at[month_index,DataSchema.percip]/1000000)
        gained_stream = monthly_stats.at[month_index,DataSchema.streamflow_after_consumption]
        
        if weather:
            if lr_weather_counter == 0:
                lr_weather_counter = np.random.randint(60,120)
                lr_weather_factor = np.random.normal(scale=0.15)
                
            if month_index == 0:
                weather_factor = np.random.normal(scale=0.35)
            
            lr_weather_counter -= 1
            
            gained_rain *= 1 + (weather_factor + lr_weather_factor)
            gained_stream *= 1 + (weather_factor +  lr_weather_factor)
        
        GAINED_GROUND_WATER_MONTHLY = 0.00775
        net = gained_rain + gained_stream + GAINED_GROUND_WATER_MONTHLY - lost_evap
        
        volume += net
        
        if volume >= MAX_VOLUME:
            elevation = bath_df.index[(bath_df['Volume']-MAX_VOLUME).abs().argsort()][:1][0]
        else:
            elevation = bath_df.index[(bath_df['Volume']-volume).abs().argsort()][:1][0]
            
        cur_date += relativedelta(months=1)
        
        predictions.loc[len(predictions.index)] = [cur_date,elevation,cur_date.strftime("%Y-%m")]
        
    return predictions

def CreateLineGraph(prediction: pd.DataFrame, lr_average_elevaton: float, df_lake: pd.DataFrame, units: str, rolling=60) -> px.scatter:

    mean_elevation_without_humans = 1282.3
    METERS_TO_FEET = 3.28084

    avg_elevation = round(df_lake['Elevation'].mean(),2)
    combined = pd.concat([df_lake,prediction],ignore_index=True)

    temp = pd.concat([combined.iloc[len(df_lake)-rolling:len(df_lake)]['Elevation'],combined.iloc[len(df_lake):]['Elevation Prediction']])
    combined['Elevation Prediction'] = temp

    if units == 'imperial':
        elevation_unit = 'ft'
        combined['Elevation Prediction'] *= METERS_TO_FEET
        lr_average_elevaton *= METERS_TO_FEET
        combined['Elevation'] *= METERS_TO_FEET
        mean_elevation_without_humans *= METERS_TO_FEET
        avg_elevation *= METERS_TO_FEET
    else:
        elevation_unit = 'm'

    colors = ['blue','red']

    combined['datetime'] = pd.to_datetime(combined['datetime'])

    fig = px.scatter(combined, y=['Elevation','Elevation Prediction'],
                        x='datetime', trendline='rolling',
                        trendline_options=dict(window=rolling),
                        color_discrete_sequence=colors,
                        labels = {
                            'value':f'Lake Elevation ({elevation_unit})',
                            'datetime':'Year'
                        },
                    )

    start_date = "1870-01-01"
    end_date = combined.at[len(combined)-1, "datetime"]
    fig.update_xaxes(type='date', range=[start_date, end_date])

    #only show trendlines
    fig.data = [t for t in fig.data if t.mode == 'lines']

    #assign label positions
    if (0 < (mean_elevation_without_humans - lr_average_elevaton) < 0.4) or (0 < (df_lake['Elevation'].mean() - lr_average_elevaton) < 0.4):
        lr_pos = 'bottom left'
        human_pos = 'top left'
    elif 0 < (lr_average_elevaton - df_lake['Elevation'].mean()) < 0.4:
        lr_pos = 'top left'
        human_pos = 'bottom left'
    else:
        lr_pos = 'top left'
        human_pos = 'top left'

    fig.add_hline(y=mean_elevation_without_humans, line_dash='dot',
                    annotation_text = f'Average Natural Level, {mean_elevation_without_humans}{elevation_unit}',
                    annotation_position = 'top left',
                    annotation_font_size = 10,
                    annotation_font_color = 'black',
                    )

    fig.add_hline(y=avg_elevation, line_dash='dot',
                annotation_text = f'Average Since 1847, {avg_elevation}{elevation_unit}',
                annotation_position = human_pos,
                annotation_font_size = 10,
                annotation_font_color = colors[0],
                line_color = colors[0],
                )

    fig.add_hline(y=lr_average_elevaton, 
                    line_dash='dot',
                    annotation_text = f'Long-term Policy Average, {lr_average_elevaton}{elevation_unit}',
                    annotation_position = lr_pos,
                    annotation_font_size = 10,
                    annotation_font_color = colors[1],
                    line_color = colors[1],
            )

    return fig

def WrittenEffects(lr_average_elevation: float, df_lake: pd.DataFrame, bath_df: pd.DataFrame) -> html.Div:
    NATURAL_ELEVATION_MEAN = 1282.30
    NATURAL_SA_MEAN = bath_df.at[NATURAL_ELEVATION_MEAN, 'Surface Area']
    NATURAL_VOLUME_MEAN = bath_df[NATURAL_ELEVATION_MEAN, 'Volume']
    HUMAN_ELEVATION_MEAN = round(df_lake['Elevation'].mean(), 2)

    delta_elevation_percent = 100 * ((lr_average_elevation - NATURAL_ELEVATION_MEAN) / NATURAL_ELEVATION_MEAN)
    delta_sa_percent = 100 * ((bath_df.at[lr_average_elevation, 'Surface Area'] - NATURAL_SA_MEAN) / NATURAL_SA_MEAN)
    delta_volume_percent = 100 * ((bath_df.at[lr_average_elevation, 'Volume'] - NATURAL_VOLUME_MEAN) / NATURAL_VOLUME_MEAN)


    if delta_elevation_percent >= 0:
        color = 'green'
        elevation_descriptor = 'higher'
        volume_descriptor = 'more'
    else:
        color = 'red'
        descriptor = 'lower'
        volume_descriptor = 'less'
    
    return html.Div(
        id = 'written-effects',
        children=[
            html.H3('Based on the selected policy choices, in the long term, the lake will be:'),
            html.Ul([
                html.Li(f'fds',style={'color':color})
            ])
        ]
    )

def BuyBackEffect(millions_spent: int) -> float:
    '''
    This function takes the amount of money spent on water rights buy backs and calculates the total amount of water saving it would create, in km3/yr
    '''
    pass

def RetrieveImage(lr_average_elevation: float) -> html.Img:
    closest_half_meter = round(lr_average_elevation * 2) / 2
    image_path = f'/assets/gsl_{closest_half_meter}.png'
    return html.Img(src=image_path)

def CreateSankeyDiagram(units: str) -> go.Figure:

    labels = ['Streamflow','Bear River','Weber River','Jordan River','Groundwater','Direct Percipitation','Total Water In','Mineral Extraction','Evaporation',
        'Lake Water Lost','Total Water Out','Other Streams']

    ACRE_FEET_PER_KM3 = 810714
    
    # Data is average from 1963 to 2022. Mineral extraction data is from 1994 to 2014. Other stream data is assumed to be linearly dependent upon the other three streams
    # lame these are hard coded in but what can you do
    km3_per_year_values = [1.59, 0.377, 0.52, 2.63, 0.09, 1.47, 0.19, 4.25, 4.19, 0.25, 0.14]

    if units == 'metric':
        volume_unit = 'km3/yr'
        values = km3_per_year_values
    else:
        volume_unit = 'AF/yr'
        values = [x * ACRE_FEET_PER_KM3 for x in km3_per_year_values]

    fig = go.Figure(
        data=go.Sankey(
            arrangement = "snap",
            valuesuffix = volume_unit,
            node = dict(
                label = labels,
                x = [0.25, 0, 0, 0, 0.25, 0.25, 0.5, 1, 1, 0.5, 0.75, 0],
                y = [0.5] * 10,
                # y = [1, 1, 0.9, 0.8, 0.8, 0.9, 1, 0.9, 1, 0.9, 1, 0.7],
                hovertemplate = "%{label}"
            ),
            link = dict(
                source = [1, 2, 3, 0, 4, 5, 10, 10, 6, 9, 11],
                target = [0, 0, 0, 6, 6, 6, 7, 8, 10, 10, 0],
                value = values,
            ),
        )
    )
    return fig

def CreateConsumptiveUseSunburst(unit: str) -> px.pie:
    
    ACRE_FEET_PER_KM3 = 810714

    consumptive_average = pd.read_pickle('data/pie_chart_pickle.pkl')

    if unit == 'imperial':
        volume_unit = 'acre feet'
        consumptive_average['Consumption (km3/yr)'] *= ACRE_FEET_PER_KM3
    else:
        volume_unit = 'km3'

    consumptive_average.rename(columns={'Consumption (km3/yr)': f'Consumption ({volume_unit}/yr)'},inplace=True)

    pie_chart = px.pie(
            consumptive_average, 
            values=f'Consumption ({volume_unit}/yr)',
            names='Consumption category',
            hover_name='Consumption category',
            hole=0.5,
            )
    pie_chart.update_traces(textinfo='percent+label',hovertemplate='<b>%{label}</b><br>Consumption: %{value:.3f}' +f' {volume_unit}/yr')
    
    return pie_chart

Policy.instantiate_from_csv('data/policies.csv')
monthly_stats = LoadMonthlyStats('data/monthly_stats.csv')
bath = pd.read_csv('data/bath_df.csv', index_col=0)
lake = CreateLakeData('data/lake_df.csv', bath)

slider_policy_list = []
for policy in Policy.slider_policies:
    slider_policy_list.append(policy.checklist_component)
    slider_policy_list.append(policy.slider_component)


app = Dash(__name__)
# server = app.server


app.layout = html.Div([
    html.Div(
        id='before-after-parent',
        children=[
            BeforeAfter(before={'src':'assets/gsl_before.jpg'}, after={'src':'assets/gsl_after.jpg'}, hover=False, id='before-after-image'),
            html.I('Click and drag to see differences in water level from 1986 to 2022')
        ]
    ),
    dcc.Graph(id='sankey-diagram'),
    dcc.Graph(id='consumptive-use-sunburst'),
    html.Div(children=[html.H2('Policy Options',style={'margin-bottom':'0em'}),html.A('Sources',href='#about-the-policies')],style={'margin-bottom':'1em'}),
    html.Div(
        id='policy-sliders',
        children = slider_policy_list
    ),
    html.Div(
        id='policy-checklist-container',
        children = [
            dcc.Checklist(
                id='policy-checklist',
                options=[{'label': x.checklist_label, 'value': x.title} for x in Policy.all_policies if not x.slider],
                value=[],
                labelStyle={'display': 'block','text-indent': '-1.25em'},
                style={'position':'relative','left': '1em'}
            ),
        ]
    ),
    html.Div(id='enviroment',
        children=[
            html.H2('Enviroment Variables'),
            dcc.Checklist(id='weather-checklist',
                options = [{
                    'label': html.Span(children=[
                        html.Strong('Cosmetic Weather'), 
                        html.Br() ,
                        'Activate RANDOMLY generated weather. Does not change predicted long term average.',
                        html.Br()
                        ]), 
                    'value': True
                    }],
                value = [],
                labelStyle={'display': 'block','text-indent': '-1.25em'},
                style={'position':'relative','left': '1em'}
            ),
            html.Br(),
            html.Strong('Adjust direct rainfall'),
            dcc.Slider(
                id = 'rain-slider',
                min = -100,
                max = 100,
                value = 0,
                marks = {
                    -100: '100% less rain (No rain)',
                    -50: '50% less rain',
                    0: 'No change',
                    50: '50% more rain',
                    100: {'label':'100% more rain (Double rain)', 'style':{'right':'-200px'}}
                },
                tooltip={
                    'placement':'bottom'
                }
            ),
            html.Strong('Adjust human water consumption'),
            dcc.Slider(
                id = 'human-consumption-slider',
                min = -100,
                max = 100,
                value = 0,
                marks = {
                    -100: '100% less consumption (No humans)',
                    -50: '50% less consumption',
                    0: 'No change',
                    50: '50% more consumption',
                    100: {'label':'100% more consumption (Double consumption)', 'style':{'right':'-200px'}},
                },
                tooltip={
                    'placement':'bottom'
                }
            ),
            html.Strong('Adjust pre-consumption streamflow'),
            dcc.Slider(
                id = 'streamflow-slider',
                min = -100,
                max = 100,
                value = 0,
                marks = {
                    -100: '100% less streamflow (No streamflow)',
                    -50: '50% less streamflow',
                    0: 'No change',
                    50: '50% more streamflow',
                    100: {'label':'100% more streamflow (Double streamflow)', 'style':{'right':'-200px',}},
                },
                tooltip={
                    'placement':'bottom'
                }
            ),
            html.Span(
                children = [
                html.Strong('How many years in the future to predict'),
                dcc.Slider(
                    id = 'years-forward-slider',
                    min = 20,
                    max = 100,
                    value = 30,
                    step = 1,
                    marks = {
                        20: '20 years forward',
                        40: '40 years forward',
                        60: '60 years forward',
                        80: '80 years forward',
                        100: {'label':'100 years forward', 'style':{'right':'-200px'}},
                    },
                    tooltip={'placement':'bottom'},
                ),
                ],
            style={'display': 'none'}
            ),

        ]
    ),

    dcc.Dropdown(id = 'unit-dropdown',options = [{'label':'Metric','value':'metric'},{'label':'Imperial','value':'imperial'},],value = 'imperial'),
    html.Button(id='run-model-button', n_clicks=0, children='Run Model'),
    html.Button(id='reset-model-button', n_clicks=0, children='Reset Selection'),
    html.Div(
        id='line-graph',
        children=[
            html.H2('Graph of predicted elevation'),
            dcc.Graph(id='output-graph'),
        ]
    ),

    html.Div(id='predicted-image',
        children = [
            html.H2('Predicted surface area'),
            html.Div(id='lake-predicted-image', style={'max-width': '500px'})
        ]
    ),
    html.H2('Predicted effects based on policy choices:'),
    html.Div(
        id='about-the-policies',
        children=[
        html.P('Here is some policy text!')
        ]
    )
])


@app.callback(
    Output('consumptive-use-sunburst','figure'),
    Output('sankey-diagram', 'figure'),
    Input('unit-dropdown','value'),
)
def DrawSunburstAndSankey(unit):
    print(unit)
    return CreateConsumptiveUseSunburst(unit), CreateSankeyDiagram(unit)

@app.callback(
    Output('output-graph','figure'),
    Output('lake-predicted-image','children'),
    Input('run-model-button', 'n_clicks'),
    State('policy-checklist','value'),
    State('rain-slider','value'),
    State('human-consumption-slider','value'),
    State('years-forward-slider','value'),
    State('streamflow-slider','value'),
    State('weather-checklist','value'),
    State('unit-dropdown','value'),
    State(Policy.slider_policies[0].id_name + '-slider','value'),
    State(Policy.slider_policies[1].id_name + '-slider','value'),
    State(Policy.slider_policies[2].id_name + '-slider','value'),
    State(Policy.slider_policies[3].id_name + '-slider','value'),
    State(Policy.slider_policies[4].id_name + '-slider','value'),
    State(Policy.slider_policies[5].id_name + '-slider','value'),
    State(Policy.slider_policies[6].id_name + '-slider','value'),
    State(Policy.slider_policies[7].id_name + '-slider','value'),
    State(Policy.slider_policies[8].id_name + '-slider','value'),
    # prevent_initial_call=True
)
def Modeling(_: int, checklist_policies: list, rain_delta: int, consumption_delta: int, years_forward: int, streamflow_delta: int, weather: list, units: str,
    slider_0: float, slider_1: float, slider_2: float, slider_3: float, slider_4: float, slider_5: float, slider_6: float, slider_7: float, slider_8: float,) -> list:

    #this is a list of policy objects that are applied to the 'monthly_stats' dataframe
    applied_policies = []

    policy_slider_values = [slider_0,slider_1,slider_2,slider_3,slider_4,slider_5,slider_6,slider_7,slider_8]
    i=0
    for selected_cost in policy_slider_values:
        max_consumption_change = Policy.slider_policies[i].delta
        max_yearly_cost = Policy.slider_policies[i].first_thirty_cost_in_millions
        # this equation figures out the monthly change to consumption given the selected investment amount
        selected_consumption_change_monthly = (selected_cost * (max_consumption_change / max_yearly_cost)) /12
        applied_policies.append(
            Policy(
                f'slider {i}',
                'n/a',
                Policy.slider_policies[i].affected,
                Policy.slider_policies[i].affect_type,
                delta = -selected_consumption_change_monthly
            )
        )
        i += 1

    # the dash component passes 'weather' as a list
    if len(weather) > 0:
        weather = True
    else:
        weather = False

    #adds checklist policies (without a slider) to the policies to apply
    for policy in Policy.all_policies:
        if policy.title in checklist_policies:
            applied_policies.append(policy)


    rain_delta = (rain_delta+100) / 100
    applied_policies.append(Policy('rain-slider','n/a',DataSchema.percip,'proportion',delta=rain_delta))

    consumption_delta = (consumption_delta + 100) / 100
    applied_policies.append(Policy('consumption-slider','n/a',DataSchema.total_consumptive_use,'proportion',delta=consumption_delta))

    streamflow_delta = (streamflow_delta + 100) / 100
    applied_policies.append(Policy('streamflow-slider','n/a',DataSchema.streamflow_before_consumption,'proportion',delta=streamflow_delta))

    #adjust monthly stats based on the selected policies
    adjusted_monthly_stats = AdjustMonthlyStats(applied_policies, monthly_stats)

    #run the model based on the adjusted monthly stats
    prediction = GSLPredictor(years_forward, adjusted_monthly_stats, bath, lake, weather=weather)

    #long run average is based on the average of the last year. If weather is selected, the model is ran again without weather to find the true lr average
    MONTHS_BEFORE_LONG_RUN_AVG = 108
    if not weather:
        # lr_average_elevation = round(prediction['Elevation Prediction'].loc[MONTHS_BEFORE_LONG_RUN_AVG:].mean(),2)
        lr_average_elevation = round(prediction.tail(12)['Elevation Prediction'].mean(), 2)
    else:
        temp_prediction_weather = GSLPredictor(years_forward, adjusted_monthly_stats, bath, lake, weather=False)
        # lr_average_elevation = round(temp_prediction_weather['Elevation Prediction'].loc[MONTHS_BEFORE_LONG_RUN_AVG:].mean(),2)
        lr_average_elevation = round(temp_prediction_weather.tail(12)['Elevation Prediction'].mean(), 2)

    # units cosmetically change graph
    line_graph = CreateLineGraph(prediction, lr_average_elevation, lake, units)

    lake_picture = RetrieveImage(lr_average_elevation)

    return line_graph, lake_picture


@app.callback(
    Output('policy-checklist','value'),
    Output('rain-slider','value'),
    Output('human-consumption-slider','value'),
    Output('years-forward-slider','value'),
    Output('streamflow-slider','value'),
    Output('weather-checklist','value'),
    Input('reset-model-button', 'n_clicks')
)
def ResetButton(_:int):
    return [], 0, 0, 30, 0, []

# Below are the dash call back for policy sliders. Each callback displays or hides the slider based on the adjoining checkbox. It also resets it to zero when hidden
@app.callback(
    Output(Policy.slider_policies[0].id_name + '-display', 'style'),
    Output(Policy.slider_policies[0].id_name + '-slider', 'value'),
    Input(Policy.slider_policies[0].id_name + '-checklist', 'value'),
)
def DisplayWaterBuyback(selected):
    if len(selected) == 0:
        return {'display': 'none'}, 0
    else:
        return {'display': 'block'}, 0
@app.callback(
    Output(Policy.slider_policies[1].id_name + '-display', 'style'),
    Output(Policy.slider_policies[1].id_name + '-slider', 'value'),
    Input(Policy.slider_policies[1].id_name + '-checklist', 'value'),
)
def DisplayWaterBuyback(selected):
    if len(selected) == 0:
        return {'display': 'none'}, 0
    else:
        return {'display': 'block'}, 0
@app.callback(
    Output(Policy.slider_policies[2].id_name + '-display', 'style'),
    Output(Policy.slider_policies[2].id_name + '-slider', 'value'),
    Input(Policy.slider_policies[2].id_name + '-checklist', 'value'),
)
def DisplayWaterBuyback(selected):
    if len(selected) == 0:
        return {'display': 'none'}, 0
    else:
        return {'display': 'block'}, 0
@app.callback(
    Output(Policy.slider_policies[3].id_name + '-display', 'style'),
    Output(Policy.slider_policies[3].id_name + '-slider', 'value'),
    Input(Policy.slider_policies[3].id_name + '-checklist', 'value'),
)
def DisplayWaterBuyback(selected):
    if len(selected) == 0:
        return {'display': 'none'}, 0
    else:
        return {'display': 'block'}, 0
@app.callback(
    Output(Policy.slider_policies[4].id_name + '-display', 'style'),
    Output(Policy.slider_policies[4].id_name + '-slider', 'value'),
    Input(Policy.slider_policies[4].id_name + '-checklist', 'value'),
)
def DisplayWaterBuyback(selected):
    if len(selected) == 0:
        return {'display': 'none'}, 0
    else:
        return {'display': 'block'}, 0
@app.callback(
    Output(Policy.slider_policies[5].id_name + '-display', 'style'),
    Output(Policy.slider_policies[5].id_name + '-slider', 'value'),
    Input(Policy.slider_policies[5].id_name + '-checklist', 'value'),
)
def DisplayWaterBuyback(selected):
    if len(selected) == 0:
        return {'display': 'none'}, 0
    else:
        return {'display': 'block'}, 0
@app.callback(
    Output(Policy.slider_policies[6].id_name + '-display', 'style'),
    Output(Policy.slider_policies[6].id_name + '-slider', 'value'),
    Input(Policy.slider_policies[6].id_name + '-checklist', 'value'),
)
def DisplayWaterBuyback(selected):
    if len(selected) == 0:
        return {'display': 'none'}, 0
    else:
        return {'display': 'block'}, 0
@app.callback(
    Output(Policy.slider_policies[7].id_name + '-display', 'style'),
    Output(Policy.slider_policies[7].id_name + '-slider', 'value'),
    Input(Policy.slider_policies[7].id_name + '-checklist', 'value'),
)
def DisplayWaterBuyback(selected):
    if len(selected) == 0:
        return {'display': 'none'}, 0
    else:
        return {'display': 'block'}, 0
@app.callback(
    Output(Policy.slider_policies[8].id_name + '-display', 'style'),
    Output(Policy.slider_policies[8].id_name + '-slider', 'value'),
    Input(Policy.slider_policies[8].id_name + '-checklist', 'value'),
)
def DisplayWaterBuyback(selected):
    if len(selected) == 0:
        return {'display': 'none'}, 0
    else:
        return {'display': 'block'}, 0

if __name__ == '__main__':
    app.run_server(debug=True)
