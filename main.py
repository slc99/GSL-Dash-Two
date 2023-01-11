from dash import Dash, dcc, html, dash_table
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
import pickle

class DataSchema:
    streamflow_after_consumption = 'streamflow_after_consumption'
    groundwater = 'groundwater'
    evap_km3_per_km2 = 'evap_km3_per_km2'
    percip_km3_per_km2 = 'percip_km3_per_km2'
    total_consumptive_use = 'total_consumptive_use'
    streamflow_before_consumption = 'streamflow_before_consumption'
    agriculture_consumption = 'agriculture_consumption'
    municipal_consumption = 'municipal_consumption'
    mineral_consumption =  'mineral_consumption'
    impounded_wetland_consumption = 'impounded_wetland_consumption'
    reservoir_consumption =  'reservoir_consumption'
    water_imports = 'water_imports'
    utah_population_millions = 'utah_population_millions'
    avg_elevation = 'avg_elevation'
    avg_temp_c = 'avg_temp_c'
    avg_sa = 'avg_sa'
    avg_volume = 'avg_volume'

class Policy:

    all_policies = []
    slider_policies = []

    def __init__(self, title: str, description: str, affected: str, 
        affect_type: str, delta: float, initial_cost = 0, cost_per_year = 0, 
        slider = '', slider_message = '', selected_cost = 0):

        self.title = title
        self.description = description
        self.affected = affected
        self.affect_type = affect_type
        self.delta = delta
        self.initial_cost = initial_cost
        self.cost_per_year = cost_per_year
        self.slider = slider
        self.slider_message = slider_message
        if self.initial_cost > 0 and self.cost_per_year > 0:
            # change this to implement sliders where there is an initial and yearly cost. 
            # may need a callback from the years forward slider
            pass
        if cost_per_year > 0:
            self.max_to_invest = cost_per_year
        else:
            self.max_to_invest = initial_cost
        self.checklist_label = html.Span(children=[html.Strong(self.title), html.Br() ,self.description, html.Br()])
        self.id_name = title.replace(' ','-').lower()
        self.selected_cost = selected_cost

        Policy.all_policies.append(self)
        if self.slider:
            Policy.slider_policies.append(self)
    
    def __repr__(self):
        return f'{self.title}, {self.description}, {self.affected}, {self.affect_type}, {self.delta}'

    def cost_for_years(self, years: int) -> float:
        if self.selected_cost != 0:
            if self.cost_per_year > 0:
                return self.initial_cost + (self.selected_cost * years)
            return self.selected_cost
        return self.initial_cost + (self.cost_per_year * years)

    def create_checklist_component(self):
        id_name = self.title.replace(' ','-').lower()
        return dcc.Checklist(
            id=id_name + '-checklist',
            options = [{'label': self.checklist_label, 'value': self.title}],
            value = [],
            labelStyle={'display': 'block','text-indent': '-1.25em'},
            style={'position':'relative','left': '1em'}
        )
    
    def create_slider_component(self):
        return html.Span(
            id=self.id_name + '-display',
            children = [
                html.I(self.slider_message,style={'position':'relative','left': '1em'}),
                html.Br(),
                dcc.Slider(
                id = self.id_name + '-slider',
                min = 0,
                max = self.max_to_invest,
                value = 0,
                marks = {
                    0: '$0', 
                    int(self.max_to_invest/2): Effect.format_cost_to_print(self.max_to_invest/2), 
                    int(self.max_to_invest): {'label':Effect.format_cost_to_print(self.max_to_invest), 'style':{'right':'-200px'}}
                },
                tooltip={'placement':'bottom'}), 
            ], 
        style={'display': 'none'}
        )   

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

class Effect:
    all_effects = []

    def __init__(self, description: str, lower_threshold: float, upper_threshold: float, cost_equation: str, units: str):
        self.description = description
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.cost_equation = cost_equation
        self.units = units
        Effect.all_effects.append(self)

    def cost_function(self, elevation_m: float, bath: pd.DataFrame) -> float:
        if self.cost_equation == '':
            return 0
        MAX_SURFACE_AREA = 5574.65
        # exposed_km2 is used in the cost equations that are evaluated and returned
        exposed_km2 = MAX_SURFACE_AREA - bath.at[round(elevation_m,2), 'Surface Area']
        return eval(self.cost_equation.split('=')[1])
    
    def filled_description(self, elevation: float, bath: pd.DataFrame) -> str:
        cost = self.format_cost_to_print(self.cost_function(elevation, bath))
        return self.description.format(cost=cost)
    
    @staticmethod
    def format_cost_to_print(cost: float) -> str:
        if cost > 1000000000:
            word = 'billion'
            cost /= 1000000000
            return f'${cost:.1f} {word}'
        elif cost > 1000000:
            word = 'million'
            cost /= 1000000            
            return f'${cost:.1f} {word}'
        elif cost > 1000:
            return f'${round(cost/1000)},000'
        else:
            return f'${cost:.0f}'
    
    @classmethod
    def instantiate_from_csv(cls, path:str):
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            effects = list(reader)
        for effect in effects:
            Effect(
                description = effect.get('effect_description'),
                lower_threshold = float(effect.get('lower_m')),
                upper_threshold = float(effect.get('upper_m')),
                cost_equation = effect.get('cost_equation'),
                units = effect.get('units')
            )

def FormatNumberToPrint(cost: float) -> str:
    if cost > 1000000000:
        word = 'billion'
        cost /= 1000000000
        return f'{cost:.1f} {word}'
    elif cost > 1000000:
        word = 'million'
        cost /= 1000000            
        return f'{cost:.1f} {word}'
    elif cost > 1000:
        return f'{round(cost/1000)},000'
    else:
        return f'{cost:.00f}'

def LoadYearlyStats(path: str) -> dict:
    '''
    Loads yearly statsitics as a dictionary
    '''
    data = {}
    with open(path,'rb') as handle:
        unconverted_data = pickle.load(handle)

    data[DataSchema.streamflow_after_consumption] = unconverted_data['streamflow_after_consumption']
    data[DataSchema.groundwater] = unconverted_data['groundwater']
    data[DataSchema.percip_km3_per_km2] = unconverted_data['percip_km3_per_km2']
    data[DataSchema.evap_km3_per_km2] = unconverted_data['evap_km3_per_km2']
    data[DataSchema.streamflow_before_consumption] = unconverted_data['streamflow_before_consumption']
    data[DataSchema.avg_temp_c] = unconverted_data['avg_temp_c']
    data[DataSchema.municipal_consumption] = unconverted_data['municipal_consumption']
    data[DataSchema.reservoir_consumption] = unconverted_data['reservoir_consumption']
    data[DataSchema.mineral_consumption] = unconverted_data['mineral_consumption']
    data[DataSchema.impounded_wetland_consumption] = unconverted_data['wetland_consumption']
    data[DataSchema.water_imports] = unconverted_data['water_imports']
    data[DataSchema.total_consumptive_use] = unconverted_data['net_consumption']
    data[DataSchema.agriculture_consumption] = unconverted_data['agriculture_consumption']

    return data

def AdjustYearlyStats(policies: list, yearly_stats: dict) -> dict:
    '''
    Adjusts the yearly variables by the given policies
    '''
    adjusted_yearly = yearly_stats.copy()

    for policy in policies:
        if policy.affect_type == 'absolute':
            adjusted_yearly[policy.affected] += policy.delta
            if adjusted_yearly[policy.affected] < 0:
                adjusted_yearly[policy.affected] = 0
    for policy in policies:
        if policy.affect_type == 'proportion':
            new_adjusted = adjusted_yearly[policy.affected] * policy.delta
            policy.delta_absolute = adjusted_yearly[policy.affected] - new_adjusted
            adjusted_yearly[policy.affected] = new_adjusted
    
    adjusted_yearly[DataSchema.total_consumptive_use] = (
        adjusted_yearly[DataSchema.agriculture_consumption] 
        + adjusted_yearly[DataSchema.municipal_consumption]
        + adjusted_yearly[DataSchema.reservoir_consumption]
        + adjusted_yearly[DataSchema.mineral_consumption]
        + adjusted_yearly[DataSchema.impounded_wetland_consumption]
    )

    for policy in policies:
        if policy.affected == DataSchema.total_consumptive_use:
            if policy.affect_type == 'absolute':
                adjusted_yearly[policy.affected] += policy.delta
                #clip to zero
                if adjusted_yearly[policy.affected] < 0:
                    adjusted_yearly[policy.affected] = 0
            if policy.affect_type == 'proportion':
                new_adjusted = adjusted_yearly[policy.affected] * policy.delta
                policy.delta_absolute = adjusted_yearly[policy.affected] - new_adjusted
                adjusted_yearly[policy.affected] = new_adjusted

    # DELETE ME!!!!
    # adjusted_yearly[DataSchema.total_consumptive_use] = 1.31
    # ABOVE THIS

    adjusted_yearly[DataSchema.streamflow_after_consumption] = (adjusted_yearly[DataSchema.streamflow_before_consumption] 
                                                                - adjusted_yearly[DataSchema.total_consumptive_use])

    
    return adjusted_yearly

def GSLPredictor(years_forward: int, yearly_stats: dict,
                    bath_df: pd.DataFrame, lake_df: pd.DataFrame):
    '''
    Predicts the yearly elevation of the Great Salt Lake given the information passed
    '''
    
    MAX_VOLUME = 39.896064
        
    cur_date = pd.to_datetime(date.today().strftime("%Y"))
    start_year = int(str(cur_date).split('-')[0])
    
    elevation = round(lake_df.at[start_year,DataSchema.avg_elevation], 2)
    predictions = pd.DataFrame(columns=['datetime','Elevation Prediction','YYYY'])
    
    for i in range(years_forward):
        surface_area = bath_df.at[elevation,'Surface Area']
        volume = bath_df.at[elevation,'Volume']
        
        lost_evap = surface_area * yearly_stats[DataSchema.evap_km3_per_km2]
        gained_rain = surface_area * yearly_stats[DataSchema.percip_km3_per_km2]
        gained_stream = yearly_stats[DataSchema.streamflow_after_consumption]
        net = gained_rain + gained_stream + yearly_stats[DataSchema.groundwater] - lost_evap
        
        volume += net
        
        if volume >= MAX_VOLUME:
            elevation = bath_df.index[(bath_df['Volume']-MAX_VOLUME).abs().argsort()][:1][0]
        else:
            elevation = bath_df.index[(bath_df['Volume']-volume).abs().argsort()][:1][0]
            
        cur_date += relativedelta(years=1)
        predictions.loc[len(predictions.index)] = [cur_date,elevation,cur_date.strftime("%Y")]
        
    return predictions

def CreateLineGraph(prediction: pd.DataFrame, lr_average_elevaton: float, lake: pd.DataFrame, units: str, rolling=5) -> px.scatter:
    '''
    Creates a line graph of the past elevation and predicted elevation.
    '''
    prediction['YYYY'] = prediction['YYYY'].astype('int64')
    adj_pred = prediction.set_index('YYYY')

    MEAN_ELEVATION_BEFORE_1847 = 1282.3
    METERS_TO_FEET = 3.28084
    historic_avg_elevation = round(lake[DataSchema.avg_elevation].mean(),2)

    combined = pd.concat([lake[[DataSchema.avg_elevation]],adj_pred])
    temp = pd.concat([combined.iloc[len(lake)-rolling:len(lake)][DataSchema.avg_elevation],combined.iloc[len(lake):]['Elevation Prediction']])
    combined['Elevation Prediction'] = temp
    combined['Year'] = combined.index

    if units == 'imperial':
        elevation_unit = 'ft'
        combined['Elevation Prediction'] *= METERS_TO_FEET
        lr_average_elevaton *= METERS_TO_FEET
        combined[DataSchema.avg_elevation] *= METERS_TO_FEET
        MEAN_ELEVATION_BEFORE_1847 *= METERS_TO_FEET
        historic_avg_elevation *= METERS_TO_FEET
    else:
        elevation_unit = 'm'

    colors = ['blue','red']

    fig = px.scatter(
        combined, 
        y=[DataSchema.avg_elevation,'Elevation Prediction'],
        x='Year',
        trendline='rolling',
        trendline_options=dict(window=rolling),
        color_discrete_sequence=colors,
        labels = {
            'value':f'Lake Elevation ({elevation_unit})',
            # 'datetime':'Year'
        },
    )

    start_date = 1870
    end_date = combined[-1:].index[0]
    fig.update_layout(
        xaxis_range=[start_date,end_date],
        margin={
            'l':20,
            'r':20,
            'b':20,
            't':20 
        }
    )

    #only show trendlines:
    fig.data = [t for t in fig.data if t.mode == 'lines']

    lr_pos = 'top left'
    human_pos = 'top left'

    fig.add_hline(y=MEAN_ELEVATION_BEFORE_1847, line_dash='dot',
                    annotation_text = f'Average Natural Level, {MEAN_ELEVATION_BEFORE_1847:.2f}{elevation_unit}',
                    annotation_position = 'top left',
                    annotation_font_size = 10,
                    annotation_font_color = 'black',
    )

    fig.add_hline(y=historic_avg_elevation, line_dash='dot',
                    annotation_text = f'Average Since 1847, {historic_avg_elevation:.2f}{elevation_unit}',
                    annotation_position = human_pos,
                    annotation_font_size = 10,
                    annotation_font_color = colors[0],
                    line_color = colors[0],
    )

    fig.add_hline(y=lr_average_elevaton, 
                    line_dash='dot',
                    annotation_text = f'Long-run Policy Average, {lr_average_elevaton:.2f}{elevation_unit}',
                    annotation_position = lr_pos,
                    annotation_font_size = 10,
                    annotation_font_color = colors[1],
                    line_color = colors[1],
    )

    return fig

def RetrieveImage(elevation: float) -> html.Img:
    ''' Given an a lake elevation, retrieves the matching picture of the lake '''
    closest_half_meter = round(elevation * 2) / 2
    image_path = f'/assets/gsl_{closest_half_meter}.png'
    return html.Img(id='lake-image',src=image_path)

def CreateSankeyDiagram() -> go.Figure:

    path = 'data/sankey_dict.pkl'
    with open(path,'rb') as handle:
        sankey_dict = pickle.load(handle)

    link_value_list = list(sankey_dict.values())
    link_value_list.remove(sankey_dict['Total Water Out'])

    volume_unit = 'km3'

    fig = go.Figure(
        data = go.Sankey(
            arrangement = "snap",
            valuesuffix = volume_unit,
            node = dict(
                label = list(sankey_dict.keys()),
                x = [0.25, 0, 0, 0, 0.25, 0.25, 0, 0.5, 1, 1, 0.7, 0.5],
                y = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                hovertemplate = "%{label}"
            ),
            link = dict(
                source = [0, 1, 2, 3, 4, 5, 6, 7, 10, 10, 11],
                target = [7, 0, 0, 0, 7, 7, 0, 10, 8, 9, 10],
                value = link_value_list,            
            )
        ),
    )
    fig.update_layout(title='Yearly flow of water in and out of the Great Salt Lake (2003-2013)')
    return fig

def CreateConsumptiveUseSunburst() -> px.pie:
    consumptive_average = pd.read_pickle('data/pie_chart_df.pkl')
    volume_unit = 'km3'

    consumptive_average.rename(columns={'Consumption (km3/yr)': f'Consumption ({volume_unit}/yr)'},inplace=True)

    pie_chart = px.pie(
        consumptive_average, 
        values=f'Consumption ({volume_unit}/yr)',
        names='Consumption Category',
        hover_name='Consumption Category',
        hole=0.5,
        title='Average yearly water consumption in the Great Salt Lake Basin (2003-2013)',
    )
    pie_chart.update_traces(textinfo='percent+label',hovertemplate='<b>%{label}</b><br>Consumption: %{value:.2f}' + f' {volume_unit}/yr')
    pie_chart.update_traces(textposition='outside')

    return pie_chart

def CreateWrittenEffects(lr_elevation: float, applied_policies: pd.DataFrame, bath: pd.DataFrame, unit_designation: str, years_forward: int) -> html.Div:

    MI2_PER_KM2 = 0.386102
    AVERAGE_M_SINCE_1847 = 1282.30
    ACRE_FEET_PER_KM3 = 810714
    FEET_PER_METER = 3.28084
    ELEVATION_M_PER_KM3_CONSUMPTION = 1.948
    M3_PER_KM3 = 1000000
    ACRE_FEET_PER_M3 = ACRE_FEET_PER_KM3 / M3_PER_KM3

    historical_average_sa = bath.at[AVERAGE_M_SINCE_1847, 'Surface Area']
    predicted_average_sa = bath.at[lr_elevation, 'Surface Area']
    change_in_sa = predicted_average_sa - historical_average_sa
    percent_change_sa = 100 * change_in_sa / historical_average_sa

    
    if unit_designation == 'imperial':
        volume_unit = 'AF'
        volume_unit_effectivness = 'AF'
        elevation_unit = 'ft'
        area_unit = 'mi2'
        area_unit_words = 'square miles'
        change_in_sa *= MI2_PER_KM2
    else:
        volume_unit = 'km3'
        volume_unit_effectivness = 'm3'
        elevation_unit = 'm'
        area_unit = 'km2'
        area_unit_words = 'square kilometers'
    
    # now we form the policy output table
    policies_df = pd.DataFrame(
        columns = [
            'Policy',
            f'Cost over {years_forward} years (millions)',
            f'Yearly Water Savings ({volume_unit})',
            f'Approximate Effect on Elevation ({elevation_unit})',
            f'Cost effectivness ({volume_unit_effectivness}/million $)'
        ]
    )

    # add applied policies to the dataframe by row
    for policy in applied_policies:

        POLICIES_TO_NOT_DISPLAY = ['Rain Adjustment Slider','Consumption Ajustment Slider','Temperature Adjustment Slider','Streamflow Adjustment Slider']
        if policy.title in POLICIES_TO_NOT_DISPLAY:
            continue

        policy_title = policy.title
        policy_cost_millions = policy.cost_for_years(years_forward) / 1000000
        policy_delta = -policy.delta
        policy_effect_on_elevation = policy_delta * ELEVATION_M_PER_KM3_CONSUMPTION
        if policy_cost_millions != 0:
            policy_cost_effectiveness = M3_PER_KM3 * policy_delta / policy_cost_millions
        else:
            policy_cost_effectiveness = None

        policy_features = [policy_title, policy_cost_millions, policy_delta, policy_effect_on_elevation, policy_cost_effectiveness]
        policies_df.loc[len(policies_df)] = policy_features
    
    # add the 'total' row
    policies_df.loc['Total: '] = policies_df.sum(numeric_only=True)
    policies_df.at['Total: ','Policy'] = 'Total: '
    policies_df.at['Total: ',f'Cost effectivness ({volume_unit_effectivness}/million $)'] = (
        M3_PER_KM3 
        * policies_df.at['Total: ',f'Yearly Water Savings ({volume_unit})']                                                                 
        / policies_df.at['Total: ',f'Cost over {years_forward} years (millions)']
    )
    
    # convert for units after all the processing is done
    if unit_designation == 'imperial':
        policies_df[f'Yearly Water Savings ({volume_unit})'] *= ACRE_FEET_PER_KM3
        policies_df[f'Approximate Effect on Elevation ({elevation_unit})'] *= FEET_PER_METER
        policies_df[f'Cost effectivness ({volume_unit_effectivness}/million $)'] *= ACRE_FEET_PER_M3

    total_water_savings = policies_df.iloc[-1][f'Yearly Water Savings ({volume_unit})']
    total_policy_cost = policies_df.at['Total: ', f'Cost over {years_forward} years (millions)'] * 1000000
    
    if np.isnan(total_water_savings):
        total_water_savings = 0
        total_policy_cost = 0

    policies_df = policies_df.round(2)
    policy_table = dash_table.DataTable(policies_df.to_dict('records'),[{"name": i, "id": i} for i in policies_df.columns])

    #now to create the list of written effects
    if change_in_sa > 0:
        words = ['less', 'a decrease','increase']
    else:
        words = ['more','an increase','decrease']

    written_number ={
        20: 'twenty',
        30: 'thirty',
        40: 'fourty',
        50: 'fifty',
        60: 'sixty',
        70: 'seventy',
        80: 'eighty',
        90: 'ninety',
        100: 'one hundred'
    }

    li_list = []

    yearly_lake_level_cost = 0
    for effect in Effect.all_effects:
        if effect.lower_threshold < lr_elevation < effect.upper_threshold:
           li_list.append(html.Li(effect.filled_description(lr_elevation,bath)))
           yearly_lake_level_cost += effect.cost_function(lr_elevation, bath)

    policy_cost_minus_lake_costs = total_policy_cost - (years_forward * yearly_lake_level_cost)

    if policy_cost_minus_lake_costs < 0:
        summary_words = ['effects of the lake\'s level','selected policies']
    else:
        summary_words = ['selected policies', 'effects of the lake\'s elevation']

    summary_sentence = html.Li(children=[
        html.Strong(f'''Overall, the {summary_words[0]} will cost ${FormatNumberToPrint(abs(policy_cost_minus_lake_costs))} more then the cost of the {summary_words[1]}
        over the next {written_number[years_forward]} years. '''),
        'This ignores all non-monetary effects. See below for further discussion.'
    ])
    total_lake_level_cost = html.Li(children=[
        'The effects of the lake\'s predicted level will cost ',
        html.Strong(f'${FormatNumberToPrint(years_forward * yearly_lake_level_cost)} '),
        f'over the next {written_number[years_forward]} years, equivalent to ${FormatNumberToPrint(yearly_lake_level_cost)} yearly.',
    ])
    written_policy_effects = html.Li(children=[
        f'The selected policies will {words[2]} water going into the lake by {total_water_savings:.2f} {volume_unit} per year and cost ',
        html.Strong(f'${FormatNumberToPrint(total_policy_cost)}'),
        f'over then next {written_number[years_forward]} years.'
    ])
    surface_area_effect = html.Li(children=[
        html.Strong(f'{FormatNumberToPrint(abs(change_in_sa))} {area_unit_words} of lakebed will be exposed'),
        f', {words[1]} of {-percent_change_sa:.0f}% compared to pre-settler levels.',
    ])

    if change_in_sa < 0:
        li_list.insert(0, surface_area_effect)
    if total_water_savings != 0:
        li_list.insert(0, written_policy_effects)
    if yearly_lake_level_cost != 0:
        li_list.insert(0, total_lake_level_cost)
    li_list.insert(0, summary_sentence)

    return html.Div(
        id='written-lake-effects-output', 
        children=[
            html.Ul(children=li_list),
            html.Button(id='show-policy-table-button', n_clicks=0, children='Show a detailed breakdown of selected policies'),
            html.Br(),
            html.Div(id='policy-table',children=policy_table, style={'display': 'none',}),
        ]      
    )

def NewLakeStats(lr_elevation: float, bath: pd.DataFrame, unit_designation: str) -> html.Div:
    MI2_PER_KM2 = 0.386102
    AVERAGE_M_SINCE_1847 = 1282.30
    ACRE_FEET_PER_KM3 = 810714
    FEET_PER_METER = 3.28084
    M3_PER_KM3 = 1000000
    LOWEST_POINT_M = 1270

    predicted_sa = bath.at[lr_elevation, 'Surface Area']
    predicted_volume = bath.at[lr_elevation, 'Volume']
    predicted_depth = lr_elevation - LOWEST_POINT_M
    percent_sa = 100 * predicted_sa / bath.at[AVERAGE_M_SINCE_1847, 'Surface Area']
    percent_volume = 100 * predicted_volume / bath.at[AVERAGE_M_SINCE_1847, 'Volume']
    percent_elevation = 100 * predicted_depth / (AVERAGE_M_SINCE_1847 - LOWEST_POINT_M)

    if unit_designation == 'imperial':
        volume_unit = 'AF'
        elevation_unit = 'ft'
        area_unit = 'mi2'
        predicted_sa *= MI2_PER_KM2
        predicted_depth *= FEET_PER_METER
        predicted_volume *= ACRE_FEET_PER_KM3
    else:
        volume_unit = 'km3'
        elevation_unit = 'm'
        area_unit = 'km2'

    output = html.Ul(
        children=[
            html.Li(
                children=[
                    'Predicted maximum depth: ',
                    html.Strong(f'{predicted_depth:.1f}{elevation_unit}'),
                    f', {percent_elevation:.0f}% of the average since 1847.'
                ]
            ),
            html.Li(
                children=[
                    'Predicted volume: ',
                    html.Strong(f'{predicted_volume:.1f}{volume_unit}'),
                    f', {percent_volume:.0f}% of the average since 1847.'
                ]
            ),
            html.Li(
                children=[
                    'Predicted surface area: ',
                    html.Strong(f'{predicted_sa:.0f}{area_unit}'),
                    f', {percent_sa:.0f}% of the average since 1847.'
                ]
            ),
        ]
    )

    return html.Div(output)

def TempToEvap(temperature_delta: float, unit: str) -> Policy:
    '''Based on linear regression'''
    C_PER_F = 5/9
    EVAP_CHANGE_PER_C = 0.00010 #should be 0.0001
    AVG_TEMP_2018_THROUGH_2022 = 11.67
    INTERCEPT = -0.000075

    if unit == 'imperial':
        temperature_delta *= C_PER_F

    evap_rate = INTERCEPT + ((temperature_delta + AVG_TEMP_2018_THROUGH_2022) * EVAP_CHANGE_PER_C)

    return Policy(
        title='Temperature Adjustment Slider',
        description='Temperature effects the lake\'s rate of evaporation',
        affected=DataSchema.evap_km3_per_km2,
        affect_type='absolute',
        delta=(evap_rate)
    )

def ParseMarkedownText(text_file_path: str, mathjax: bool=False) -> dcc.Markdown:
    with open(text_file_path, 'r') as f:
        content = f.read()
    return dcc.Markdown(content, mathjax=mathjax)

def QAndASection(yearly_lake: pd.DataFrame) -> html.Section:
    '''This section is seperated because it is ugly. I do not think that is a good reason'''
    labels_dict = {
        'utah_population_millions': 'Utah Population (thousands)', #whoopsie
        'net_consumption': 'Total Water Consumption (km3/yr)',
        'avg_temp_c': 'Average Yearly Temperature (C)',
        'evap_km3_per_km2': 'Evaporation Rate (km3/km2/yr)',
        'y':'Expected Evaporation(km3/yr)',
        'Elevation':'Elevation (m)',
        'streamflow_after_consumption':'Total streamflow entering the lake (km3)',
        'datetime':'Year',
        'avg_elevation':'Average Elevation (m)',
    }
    forty_year_pop_graph = px.scatter(
        yearly_lake.tail(40), 
        x='utah_population_millions', 
        y='net_consumption', 
        trendline='ols',
        labels=labels_dict,
        title='Relationship between the population of Utah and water consumption, 1973-2013'
    )
    evap_temp_graph = px.scatter(
        yearly_lake.drop(range(1987,1992)), 
        x='avg_temp_c', 
        y='evap_km3_per_km2', 
        trendline='ols',
        labels=labels_dict,
        title='Average yearly temperature and evaporation rate, 1964-2013 (excluding 1987-1992)'
    )
    AVG_EVAP_RATE_2018_TO_2022 = 0.001153
    evap_elevation_graph = px.line(
        bath,
        y = bath['Surface Area'] * AVG_EVAP_RATE_2018_TO_2022,
        labels=labels_dict,
        title='Relationship between Great Salt Lake elevation and expected yearly evaporation'
    )
    streamflow_year_graph = px.line(
        yearly_lake.loc[1960:], 
        y='streamflow_after_consumption',
        labels=labels_dict,
        title='Streamflow entering the Great Salt Lake (1964-2021)'
    )
    elevation_trend = px.scatter(
        yearly_lake, 
        y='avg_elevation', 
        trendline='ols',
        labels=labels_dict,
        title='Average elevation of the Great Salt Lake (1847-2023)'
    )
    return html.Section(
        id='q-and-a',
        children=[
            html.Strong('Q: Does the model account for Utah\'s growing population?'),
            html.P('''While counterintuitive, the population of Utah has been uncorrelated with the total rate of water consumption for the last forty years. 
            This can be seen in the below graph, which shows the most recent forty years of water consumption data plotted against the population of Utah.'''),
            html.Div(
                className='writeup-charts',
                children=[
                    dcc.Graph(figure=forty_year_pop_graph),
                ]
            ),
            html.P('''This counterintuitive result can largely be explained by a few phenomena. First, as the population of Utah has increased, 
            farmland has been replaced by concrete. As cities grow into their surrounding countryside, land that was devoted to growing crops is replaced 
            by parking lots, homes, and lawns. As a whole, urban areas consume significantly less water per acre than farms do. In addition, efforts to reduce 
            water usage have helped keep overall water consumption flat. '''),
            html.P('''Because there is no statistical relationship between population and water consumption (since the 1970\'s), the prediction is not affected by the 
            growing population.'''),
            html.Strong('Q: How is climate change represented in the model?'),
            html.P('''Climate change (as controlled through the \'adjust average temperature\' slider)  is integrated by adjusting the evaporation rate of the lake. 
            As seen below, the evaporation rate is correlated with the average yearly temperature to a statistically significant degree (p < 0.01). By default, 
            temperature is set to +1.8 C, as this is the predicted warming by 2050 in Utah. '''),
            html.Div(
                className='writeup-charts',     
                children=[
                    dcc.Graph(figure=evap_temp_graph),
                    html.I('''Date range was based on data availability. The years 1987 through 1992 were excluded because the West Desert Pumping Project and 
                    it\'s subsequent backflow are unaccounted for.''')
                ]
            ),
            html.P('''The effect of climate change on precipitation and streamflow is not automatically integrated into the model. However, both can be controlled 
            through the provided sliders.'''),
            html.Strong('Q: Isn\'t the lake expected to disappear within five years? Why does the model not show this?'),
            dcc.Markdown('''Despite a rush of [misleading](https://www.cnn.com/2023/01/06/us/great-salt-lake-disappearing-drought-climate/index.html) 
            [headlines](https://www.washingtonpost.com/climate-environment/2023/01/06/great-salt-lake-utah-drying-up/) to the contrary, the lake 
            is not expected to disappear in five years. The [official briefing](https://pws.byu.edu/GSL%20report%202023) states that “if this \[the average since 2020\] 
            rate of water loss continues, the lake would be on track to disappear in the next five years”. This is a statement of the current rate of decline, 
            not a prediction of the future decline. The lake disappearing in five years would make a bad prediction for two reasons. First, nearly all water that 
            leaves the lake is through evaporation (the exception being water drained from the lake for mineral extraction). Because of this, the surface area is 
            directly related to the rate of water loss. As the water level declines, so does surface area. This in turn reduces the amount of water that is lost per 
            year, stabilizing the lake level at a new, lower level. This is illustrated in the below graph:'''),
            html.Div(className='writeup-charts',children=[dcc.Graph(figure=evap_elevation_graph)]),
            html.P('''Second, the sample size of two years is not large enough to base a prediction on. The below graph shows how widely the rate of streamflow varies 
            year-to-year. With a standard deviation of 1.6km3, a prediction based on two years of data is unreliable.'''),
            html.Div(
                className='writeup-charts',
                children=[
                    dcc.Graph(figure=streamflow_year_graph),
                ]
            ),
            html.Strong('Q: What is the difference between water consumption and water use?'),
            html.P('''Throughout this project, I focus on \'consumptive\' water use. Consumptive use is defined by the US Geographical Survey (USGS) as 
            "the part of water withdrawn that is evaporated, transpired, incorporated into products or crops, consumed by humans or livestock, or otherwise 
            not available for immediate use". Non-consumptive use of water  is eventually returned back to the water system. In the Great Salt Lake basin, the 
            lake ultimately receives all non-consumed water. '''),
            html.Strong('Q: Why is the predicted elevation flat?'),
            html.P('''The model can only predict a long-run average. The model does not have the ability to predict weather cycles. Historically, the lake has 
            fluctuated along a clear, downward sloping trend, pictured below:'''),
            html.Div(
                className='writeup-charts',
                children=[
                    dcc.Graph(figure=elevation_trend),
                    html.I('On average, the Great Salt Lake has trended towards losing 1.6cm (0.6in) every year since 1847')
                ]
            ),
            html.P('''Only considering a long-run average has a few disadvantages. The most important is that it underestimates the risk of ecosystem collapse and 
            other effects that involve specific thresholds. If the long-run average lake level is slightly above a threshold, then the model will not predict the event 
            happening.. However, the threshold will be crossed when the lake level decreases due to natural weather variations. To account for this, the predicted costs 
            are not built based on a threshold system, but rather a linear or quadratic relationship between expected cost and lake level. See below for more detail.'''),
            html.Strong('Q: Do you have any other cool graphs?'),
            html.P('''Of course. Here is a Sankey diagram that is helpful to understand the underlying dynamics of the Great Salt Lake. The diagram displays how water 
            enters and leaves the lake during of an average year.'''),
            dcc.Graph(className='writeup-charts',figure=CreateSankeyDiagram()),
            html.P('''Throughout an average year, humans consume 1.85km3 (1.5 million acre-feet) of water in the Great Salt Lake Basin. A breakdown of how that 
            water is consumed is displayed below. '''),
            dcc.Graph(className='writeup-charts',figure=CreateConsumptiveUseSunburst()),
            html.P('''A few notes on the above categories: Agricultural consumption is by far the largest consumer of water in the basin. Evaporative mineral ponds 
            include a handful of companies that pull water from the lake for the purpose of removing minerals. The most prominent of these companies is US Magnesium, 
            formerly known as MagCorp. Municipal consumption includes all water used by households and industrial buildings, including lawn irrigation and data centers. 
            Evaporation from impounded wetlands is the amount of water that evaporates from the surface of wetlands that are artificially maintained through a series of 
            levies near river entrances to the Great Salt Lake. Finally, reservoir evaporation is the water that evaporates from the surface of reservoirs. Because this 
            chart is based on consumptive water use, it may look different then other water use charts you may have seen.''')
        ]
    )

Policy.instantiate_from_csv('data/policies.csv')
Effect.instantiate_from_csv('data/elevation_effects.csv')
yearly_stats = LoadYearlyStats('data/yearly_stats.pkl')
yearly_stats[DataSchema.evap_km3_per_km2] = 0
bath = pd.read_pickle('data/bath_pickle.pkl')
full_lake = pd.read_pickle('data/yearly_lake.pkl')
lake = full_lake[[DataSchema.avg_elevation,DataSchema.avg_sa,DataSchema.avg_volume]]


slider_policy_list = []
for policy in Policy.slider_policies:
    slider_policy_list.append(policy.create_checklist_component())
    slider_policy_list.append(policy.create_slider_component())

app = Dash(__name__)
app.title = 'GSL Policy Dashboard'
server = app.server


app.layout = html.Div([
    html.Link(rel="preconnect", href="https://fonts.googleapis.com"),
    html.Link(href="https://fonts.googleapis.com/css2?family=Lato:wght@400;900&display=swap", rel="stylesheet"),
    html.Div(
        id='page-topper',
        children=[
            html.Img(className='top-image',src='assets/top_bar_image.png'),
            html.H1('Great Salt Lake Policy Dashboard', id='main-title'),
            html.H3(html.I('How policy decisions of today will effect the lake of tomorrow')),
            html.Img(className='top-image',src='assets/top_bar_image.png'),
        ]
    ),
    html.Div(
        id='nav-bar',
        children=[
            html.Div('GSL Dashboard'),
            html.Ul(
                children=[
                    html.Li(html.A('Inputs',href='#model-input-title')),
                    html.Li(html.A('Output',href='#model-output-title')),
                    html.Li(html.A('Q&A',href='#q-and-a-title')),
                    html.Li(html.A('Model Writeup',href='#model-writeup-title')),
                    html.Li(html.A('About Me',href='#about-me-title')),
                ]   
            ),
        ]
    ),
    html.Div(
        id='opening-blurb', 
        className='center-column-content',
        children=[ParseMarkedownText('data/markdown_text/opening_blurb.txt')],
    ),
    html.Div(
        id='model-options',
        className='center-column-content',
        children=[
            html.H2('Model Inputs', id='model-input-title'),
            html.Hr(),
            html.H3('Policy Options'),
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
            html.Div(
                id='sandbox-sliders',
                children=[
                    html.H3('Component Options'),
                    html.Br(),
                    html.Strong('Adjust direct percipitation onto the lake\'s surface'),
                    dcc.Slider(
                        id = 'rain-slider',
                        min = -100,
                        max = 100,
                        value = 1,
                        marks = {
                            -100: '100% less (No percip.)', 
                            -50: '50% less',
                            0: {'label':'Long-run average, 5-year average', 'style':{'max-width': '120px'}},
                            50: '50% more',
                            100: {'label':'100% more (Double percip.)', 'style':{'right':'-200px'}}
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
                            -100: '100% less (No water consumption)',
                            -50: '50% less',
                            0: 'Recent average',
                            50: '50% more',
                            100: {'label':'100% more (Double consumption)', 'style':{'right':'-200px'}},
                        },
                        tooltip={
                            'placement':'bottom'
                        }
                    ),
                    html.Strong('Adjust river flow before human use'),
                    dcc.Slider(
                        id = 'streamflow-slider',
                        min = -100,
                        max = 100,
                        marks = {
                            -100: '100% less (No river flow)',
                            -50: '50% less',
                            -16: {'label':'5-year average', 'style':{'max-width': '20px', 'margin-left':'-10px'}},
                            0: {'label':'Long-run average', 'style':{'max-width': '60px'}},
                            50: '50% more',
                            100: {'label':'100% more (Double river flow)', 'style':{'right':'-200px',}},
                        },
                        value = -16,
                        tooltip={
                            'placement':'bottom'
                        }
                    ),
                    html.Span(
                        id = 'temperature-slider-parent',
                        children = [
                            html.Strong('Adjust average temperature'),
                            dcc.Slider(
                                id = 'temperature-slider',
                                min = -5,
                                max = 5,
                                value = 3.2, # I do not why this works, but this actually sets a value of 1.8
                                step = 0.1,
                                tooltip={'placement':'bottom'},
                            ),
                        ],
                        style={'display': 'block'}
                    ),
                    html.Span(
                        children = [
                        html.Strong('Years into the future to predict'),
                            dcc.Slider(
                                id = 'years-forward-slider',
                                min = 20,
                                max = 100,
                                value = 40,
                                step = 10,
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
                    style={'display': 'block'}
                    ),
                ]
            ),
            dcc.Dropdown(id='unit-dropdown',options = [{'label':'Metric','value':'metric'},{'label':'Imperial','value':'imperial'},],value = 'metric'),
            html.Br(),
            html.Button(id='run-model-button', n_clicks=0, children='Run Model'),
            html.H2('Model Output', id='model-output-title'),
             html.Hr(),
        ]
    ),

    html.Div(
        id='model-output',
        children=[
            html.Div(
                id='line-graph',
                className='output-box',
                children=[
                    html.H3('Great Salt Lake\'s historic and predicted elevation (five-year rolling average):', className='output-title'),
                    dcc.Loading(dcc.Graph(id='output-graph')),
                ]
            ),
            html.Div(
                id='right-column',
                children=[
                    html.Div(
                        id='predicted-image',
                        className='output-box',
                        children = [
                            html.H3('Predicted surface area:',className='output-title'),
                            dcc.Loading(html.Div(id='lake-predicted-image')),
                            html.I('The black line indicates the shoreline of the average natural lake level of 1282m (4207ft).'),
                        ]
                    ),
                    html.Div(
                        id='new-lake-stats-container',
                        className='output-box',
                        children = [
                            html.H3('New lake statistics:', className='output-title'),
                            dcc.Loading(html.Div(id='new-lake-stats')),
                        ]
                    ),
                ]
            ),
            html.Div(
                id='written-effects-container',
                className='output-box',
                children = [
                    html.H3('Predicted effects based on policy choices:', className='output-title'),
                    dcc.Loading(html.Div(id='written-effects')),
                ]
            ),
        ]
    ),
    html.Div(
        id='writeup-under-the-model',
        className='center-column-content',
        children=[
            html.H2('Q&A (and some fun graphs)', id='q-and-a-title'),
            html.Hr(),
            QAndASection(full_lake),
            html.H2('Writeup',id='model-writeup-title'),
            html.Hr(),
            ParseMarkedownText('data/markdown_text/writeup.txt',mathjax=True),
            # a seperate div is required because of the special hanging line indent.
            html.Div(ParseMarkedownText('data/markdown_text/works-cited.txt'),id='works-cited'), 
            html.P('Lake needs more water'),
            html.H2('About Me',id='about-me-title'),
            html.Hr(),
            ParseMarkedownText('data/markdown_text/about_me.txt'),
        ]
    ),
])


@app.callback(
    Output('temperature-slider','min'),
    Output('temperature-slider','max'),
    Output('temperature-slider','marks'),
    Output('temperature-slider','value'),
    Input('unit-dropdown','value'),
    State('temperature-slider','value')
)
def AdjustDisplayedUnits(unit: str, cur_temp_value: float):
    F_PER_C = 9/5
    if unit == 'imperial':
        min = -9
        max = 9
        marks = {
            -9: '9\u00B0 f cooler',
            -4.5: '-4.5\u00B0 f',
            0: '5-year average',
            3.2: {'label':'Predicted warming by 2050', 'style':{'max-width': '120px', 'margin-left':'-20px'}},
            4.5: '+4.5\u00B0 f',
            9: {'label':'9\u00B0 f warmer', 'style':{'right':'-200px'}}
        }
        value = cur_temp_value * F_PER_C
    else:
        min = -5
        max = 5
        marks = {
            -5: '5\u00B0 C cooler',
            -2.5: '-2.5\u00B0 C',
            0: '5-year average', 
            1.8: {'label':'Predicted warming by 2050', 'style':{'max-width': '100px', 'margin-left':'-20px'}}, 
            # 2.5:'2.5\u00B0 C warmer', 
            2.5: '+2.5\u00B0 C',
            5: {'label':'5\u00B0 C warmer', 'style':{'right':'-200px'}}}
        value = cur_temp_value / F_PER_C
    
    return min, max, marks, value

@app.callback(
    Output('output-graph','figure'),
    Output('lake-predicted-image','children'),
    Output('written-effects','children'),
    Output('new-lake-stats','children'),
    Input('run-model-button', 'n_clicks'),
    State('policy-checklist','value'),
    State('rain-slider','value'),
    State('human-consumption-slider','value'),
    State('years-forward-slider','value'),
    State('streamflow-slider','value'),
    State('temperature-slider','value'),
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
def Modeling(_: int, checklist_policies: list, rain_delta: int, consumption_delta: int, years_forward: int, streamflow_delta: int, temperature_delta: float,
    units: str, slider_0: float, slider_1: float, slider_2: float, slider_3: float, slider_4: float, slider_5: float, slider_6: float, 
    slider_7: float, slider_8: float,) -> list:

    #this is a list of policy objects that are applied to the 'monthly_stats' dataframe
    applied_policies = []

    policy_slider_values = [slider_0, slider_1, slider_2, slider_3, slider_4, slider_5, slider_6, slider_7, slider_8]
    i = -1
    for cost in policy_slider_values:
        i += 1
        if cost == 0:
            continue
        portion_of_effect = cost / Policy.slider_policies[i].max_to_invest
        effective_delta = portion_of_effect * Policy.slider_policies[i].delta
        applied_policies.append(
            Policy(
                title = Policy.slider_policies[i].title,
                description = 'n/a',
                affected = Policy.slider_policies[i].affected,
                affect_type = Policy.slider_policies[i].affect_type,
                delta = -effective_delta,
                initial_cost = Policy.slider_policies[i].initial_cost,
                cost_per_year = Policy.slider_policies[i].cost_per_year,
                selected_cost = cost,
            )
        )

    #adds checklist policies (without a slider) to the policies to apply
    for policy in Policy.all_policies:
        if policy.title in checklist_policies:
            policy.selected_cost = policy.max_to_invest
            applied_policies.append(policy)


    rain_delta = (rain_delta + 100) / 100
    consumption_delta = (consumption_delta + 100) / 100
    streamflow_delta = (streamflow_delta + 100) / 100
    applied_policies.append(Policy('Rain Adjustment Slider','n/a',DataSchema.percip_km3_per_km2,'proportion',delta=rain_delta))
    applied_policies.append(Policy('Consumption Ajustment Slider','n/a',DataSchema.total_consumptive_use,'proportion',delta=consumption_delta))
    applied_policies.append(Policy('Streamflow Adjustment Slider','n/a',DataSchema.streamflow_before_consumption,'proportion',delta=streamflow_delta))
    applied_policies.append(TempToEvap(temperature_delta, units))


    adjusted_yearly_stats = AdjustYearlyStats(applied_policies, yearly_stats)

    prediction = GSLPredictor(years_forward, adjusted_yearly_stats, bath, lake)
    lr_average_elevation = round(prediction.iloc[-1]['Elevation Prediction'], 2)


    line_graph = CreateLineGraph(prediction, lr_average_elevation, lake, units)
    lake_picture = RetrieveImage(lr_average_elevation)
    written_effects = CreateWrittenEffects(lr_average_elevation, applied_policies, bath, units, years_forward)
    new_lake_stats = NewLakeStats(lr_average_elevation, bath, units)

    return line_graph, lake_picture, written_effects, new_lake_stats


@app.callback(
    Output('policy-table','style'),
    Input('show-policy-table-button','n_clicks')
)
def DisplayPolicyTable(n_clicks:int):
    if n_clicks % 2 == 0:
        return {'display': 'none'}
    return {'display': 'block'}


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
    app.run_server(debug=False)
