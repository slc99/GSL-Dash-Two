from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from . import ids
import csv

class Policy:

    all_policies = []

    def __init__(self, title: str, description: str, affected: str, affect_type: str, delta: float):
        
        assert affect_type == ('proportion' or 'absolute'), f'Affect type: {affect_type}. Must be proportion or absolute.'

        if affect_type == 'proportion':
            assert delta >= 0, f'Delta: {delta}. Delta must be above 0 '

        self.title = title
        self.description = description
        self.affected = affected
        self.affect_type = affect_type
        self.delta = delta

        Policy.all_policies.append(self)

    def __repr__(self):
        return f'{self.title}, {self.description}, {self.affected}, {self.affect_type}, {self.delta}'


    @classmethod
    def instantiate_from_csv(cls,path: str):
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

Policy.instantiate_from_csv('data/policies.csv')

policy_options = [{'label':x.title, 'value':x} for x in Policy.all_policies]
print(policy_options)


def render(app: Dash) -> html.Div:

    Policy.instantiate_from_csv('data/policies.csv')

    policy_options = [{"label":x.title, "value":x} for x in Policy.all_policies]

    # policy_options = [
    #     {
    #         "label": 'Double the streamflow',
    #         "value": "double_streamflow"
    #     },
    #     {
    #         "label": 'Half the streamflow',
    #         "value": "half_streamflow"
    #     },
    #     {
    #         "label": 'QUADRUPLE THE FLOW',
    #         "value": "quad_streamflow"
    #     }
    # ]

    return html.Div(
        className='policy-checklist',
        children=[
            dcc.Checklist(
                id=ids.POLICY_CHECKLIST,
                options=policy_options, 
                value=[],
                labelStyle={
                    'display': 'block',
                    'text-indent': '-1.25em'
                }
            ),
        ]
    )