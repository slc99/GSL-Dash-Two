from dash import Dash, html
from . import policy_checklist


def create_layout(app: Dash) -> html.Div: 
    return html.Div( className='app-div',
        children=[
            html.Section(
                className='title-section',
                children=[
                    html.H1('Great Salt Lake Simulator',className='main-title'),
                    html.P('Visualize how the policy of today will effect the lake of tomorrow',className='subtitle'),
                    html.Hr(),
                    html.P('Liam Connor',className='author-name'),
                    html.P('Published Novermber 21, 2022',className='published-date'),
                    html.Hr()
                ]
            ),
            html.Section( 
                className='background-info',
                children=[
                    html.H2('Background'),
                    html.P(
                        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Sed odio morbi quis commodo. Neque viverra justo nec ultrices dui sapien. Augue neque gravida in fermentum et sollicitudin ac. Egestas egestas fringilla phasellus faucibus scelerisque eleifend donec pretium vulputate. Suscipit tellus mauris a diam maecenas sed enim ut sem. Pretium quam vulputate dignissim suspendisse in est ante in nibh. Amet mauris commodo quis imperdiet massa. In massa tempor nec feugiat nisl. Ultricies lacus sed turpis tincidunt id aliquet risus feugiat in. Egestas integer eget aliquet nibh."
                    ),
                ]
                
            ),
            html.Section( 
                className='the-model',
                children=[
                    html.H2('The Model'),
                    html.Div(
                        className='modules',
                        children=[
                            html.Div(
                                className='module',
                                children=[
                                    html.H3('Policies'),
                                    html.Div(
                                        className="dropdown-container",
                                        children=[
                                            policy_checklist.render(app)
                                        ]
                                   ),
                                ]
                            ),
                            html.Div(
                                className='module',
                                children=[
                                    html.H3('Long Run Lake Look'),
                                    #Picture of background here and brought in images of the lake
                                ]
                            ),
                            html.Div(
                                className='module',
                                children=[
                                    html.H3('Water Height'),
                                    #Graph of water level here
                                ]
                            ),
                            html.Div(
                                className='module',
                                children=[
                                    html.H3('Estimated Effects'),
                                    #List of effects here
                                ]
                            ),
                            html.Div(
                                className='module',
                                children=[
                                    html.H3('Implications'),
                                    #List of implications here with changing colors
                                ]
                            ),
                        ]
                    )
                ]
            ),
            html.Section(
                className='model-details',
                children=[
                    html.H2('Model Details'),
                    html.P(
                        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Sed odio morbi quis commodo. Neque viverra justo nec ultrices dui sapien. Augue neque gravida in fermentum et sollicitudin ac. Egestas egestas fringilla phasellus faucibus scelerisque eleifend donec pretium vulputate. Suscipit tellus mauris a diam maecenas sed enim ut sem. Pretium quam vulputate dignissim suspendisse in est ante in nibh. Amet mauris commodo quis imperdiet massa. In massa tempor nec feugiat nisl. Ultricies lacus sed turpis tincidunt id aliquet risus feugiat in. Egestas integer eget aliquet nibh."
                    ),
                ]
            ),
            html.Section(
                className='data-and-sources',
                children=[
                    html.H2('Data and Sources'),
                ]
            )
            # html.H1(app.title),
            # html.Hr(), #divider
            # html.Div(
            #     className="dropdown-container",
            #     children=[
            #         nation_dropdown.render(app)
            #     ]
            # ),
            # bar_chart.render(app)
        ]
    )