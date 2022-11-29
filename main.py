from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import csv
import dataretrieval.nwis as nwis
from datetime import date
import numpy as np
from dateutil.relativedelta import relativedelta
import plotly.express as px

class DataSchema:
    streamflow_after_consumption = 'streamflow_after_consumption'
    groundwater = 'groundwater'
    evap_rate = 'evap_rate'
    percip = 'percip'
    total_consumptive_use = 'total_consumptive_use'
    streamflow_before_consumption = 'streamflow_before_consumption'

class Policy:

    all_policies = []

    def __init__(self, title: str, description: str, affected: str, affect_type: str, delta: float):
        
        assert affect_type == ('proportion' or 'absolute'), f'Affect type: {affect_type}. Must be proportion or absolute.'
        # assert affected in DataSchema.__dict__, f'Affected: {affected} not in {DataSchema.__dict__}'

        if affect_type == 'proportion':
            assert delta >= 0, f'Delta: {delta}. Delta must be above 0 '

        self.title = title
        self.description = description
        self.affected = affected
        self.affect_type = affect_type
        self.delta = delta
        self.checklist_label = html.Span(children=[html.Strong(self.title), html.Br() ,self.description, html.Br()])

        Policy.all_policies.append(self)

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
                delta = float(policy.get('Delta')),
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
    },inplace=True)
    return data

def AdjustMonthlyStats(policies: list[Policy], monthly_stats: pd.DataFrame) -> pd.DataFrame:
    '''
    Adjusts the monthly streamflow by the given policies
    '''
    adjusted_monthly = monthly_stats.copy(deep=True)  

    for policy in policies:
        if policy.affect_type == 'proportion':
            adjusted_monthly[policy.affected] *= policy.delta
    for policy in policies:
        if policy.affect_type == 'absolute':
            adjusted_monthly[policy.affected] += policy.delta
    
    adjusted_monthly[DataSchema.total_consumptive_use].clip(0,inplace=True)

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

def YearMonth(row):
    '''
    Function used for adding YYYY-MM column to lake dataframe. Really, not a great function 
    '''
    month_string = str(row['datetime'].month)
    if len(month_string) == 1:
        month_string = '0' + month_string
        
    return str(row['datetime'].year)+ '-' + month_string

def LoadBathData(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

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

    new_data = new_data['datetime'].dt.year.astype(str) + '-' + new_data['datetime'].dt.month.astype(str) #the better way
    #new_data['YYYY-MM'] = new_data.apply(lambda row: YearMonth(row), axis=1)
    new_data = new_data.groupby(new_data['YYYY-MM']).mean()
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
                lr_weather_factor = np.random.normal(scale=0.2)
                
            if month_index == 0:
                weather_factor = np.random.normal(scale=0.5)
            
            lr_weather_counter -= 1
            
            gained_rain *= 1 + (weather_factor + lr_weather_factor)
            gained_stream *= 1 + (weather_factor +  lr_weather_factor)
        
        gained_ground = 0.00775
        net = gained_rain + gained_stream + gained_ground - lost_evap
        
        volume += net
        
        if volume >= MAX_VOLUME:
            elevation = bath_df.index[(bath_df['Volume']-MAX_VOLUME).abs().argsort()][:1][0]
        else:
            elevation = bath_df.index[(bath_df['Volume']-volume).abs().argsort()][:1][0]
            
        cur_date += relativedelta(months=1)
        
        predictions.loc[len(predictions.index)] = [cur_date,elevation,cur_date.strftime("%Y-%m")]
        
    return predictions

def CreateLineGraph(prediction: pd.DataFrame, lr_average_elevaton: float, df_lake: pd.DataFrame, rolling=60) -> px.scatter:

    combined = pd.concat([df_lake,prediction],ignore_index=True)

    temp = pd.concat([combined.iloc[len(df_lake)-rolling:len(df_lake)]['Elevation'],combined.iloc[len(df_lake):]['Elevation Prediction']])
    combined['Elevation Prediction'] = temp

    colors = ['blue','red']

    combined['datetime'] = pd.to_datetime(combined['datetime'])

    fig = px.scatter(combined, y=['Elevation','Elevation Prediction'],
                        x='datetime', trendline='rolling',
                        trendline_options=dict(window=rolling),
                        color_discrete_sequence=colors,
                        labels = {
                            'value':'Lake Elevation (m)',
                            'datetime':'Year'
                        },
                    )

    # start_date = "1950-01-01"
    # end_date = combined.at[len(combined)-1, "datetime"]
    # fig.update_xaxes(type='date', range=[start_date, end_date])

    #only show trendlines
    fig.data = [t for t in fig.data if t.mode == 'lines']

    #assign label positions
    if (0 < (1282.3 - lr_average_elevaton) < 0.4) or (0 < (df_lake['Elevation'].mean() - lr_average_elevaton) < 0.4):
        lr_pos = 'bottom left'
        human_pos = 'top left'
    elif 0 < (lr_average_elevaton - df_lake['Elevation'].mean()) < 0.4:
        lr_pos = 'top left'
        human_pos = 'bottom left'
    else:
        lr_pos = 'top left'
        human_pos = 'top left'

    fig.add_hline(y=1282.3, line_dash='dot',
                    annotation_text = 'Average Natural Level, 1282.3m',
                    annotation_position = 'top left',
                    annotation_font_size = 10,
                    annotation_font_color = 'black',
                    )
    avg_elevation = round(df_lake['Elevation'].mean(),2)
    fig.add_hline(y=avg_elevation, line_dash='dot',
                annotation_text = f'Average Since 1847, {avg_elevation}m',
                annotation_position = human_pos,
                annotation_font_size = 10,
                annotation_font_color = colors[0],
                line_color = colors[0],
                )

    fig.add_hline(y=lr_average_elevaton, 
                    line_dash='dot',
                    annotation_text = f'Long-term Policy Average, {lr_average_elevaton}m',
                    annotation_position = lr_pos,
                    annotation_font_size = 10,
                    annotation_font_color = colors[1],
                    line_color = colors[1],
            )

    return fig

def WrittenEffects(lr_average_elevation: float, df_lake: pd.DataFrame, bath_df: pd.DataFrame) -> html.Div:
    delta_elevation = round(df_lake['Elevation'].mean() - lr_average_elevation, 2)
    delta_surface_area = round(df_lake['Surface Area'].mean() - bath_df.at[lr_average_elevation, 'Surface Area'], 2)
    delta_volume = round(df_lake['Volume'].mean() - bath_df.at[lr_average_elevation, 'Volume'], )
    if delta_elevation >= 0:
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
            html.Br(),
            html.Ul([
                html.Li(f'',style={'color':color})
            ])
        ]
    )

Policy.instantiate_from_csv('data/policies.csv')
monthly_stats = LoadMonthlyStats('data/monthly_stats.csv')
bath = LoadBathData('data/bath_df.csv')
lake = CreateLakeData('data/lake_df.csv', bath)


app = Dash(__name__)


app.layout = html.Div([
    dcc.Checklist(
        id='policy-checklist',
        options=[{'label': x.checklist_label, 'value': x.title} for x in Policy.all_policies],
        value=[],
        labelStyle={
            'display': 'block',
            'text-indent': '-1.25em'
        },
        style={
            'position':'relative',
            'left': '1em'
        }
        ),
    html.Button(id='run-model-button', n_clicks=0, children='Run Model'),
    html.H2('Graph of predicted elevation'),
    dcc.Graph(
        id='output-graph'
    ),
    html.H2('Predicted surface area'),
    html.Div(id='lake-predicted-image'),
    html.H2('Predicted effects based on policy choices:'),

])
@app.callback(
    Output('output-graph','figure'),
    Output('lake-predicted-image','children'),
    Input('run-model-button', 'n_clicks'),
    State('policy-checklist','value')
)
def modeling(_: int, selected_policies) -> list[str]:
    weather=False
    applied_policies = []
    for policy in Policy.all_policies:
        if policy.title in selected_policies:
            applied_policies.append(policy)

    adjusted_monthly_stats = AdjustMonthlyStats(applied_policies, monthly_stats)
    prediction = GSLPredictor(20, adjusted_monthly_stats, bath, lake)

    MONTHS_BEFORE_LONG_RUN_AVG = 108
    if not weather:
        lr_average_elevation = round(prediction['Elevation Prediction'].loc[MONTHS_BEFORE_LONG_RUN_AVG:].mean(),2)
    else:
        pass

    line_graph = CreateLineGraph(prediction, lr_average_elevation, lake)

    closest_meter = int(lr_average_elevation)
    image_path = f'/assets/gsl_{closest_meter}.png'
    lake_picture = html.Img(src=image_path)

    return line_graph, lake_picture
    
    # return px.scatter(prediction, x='datetime', y='Elevation Prediction')

if __name__ == '__main__':
    app.run_server(debug=True)