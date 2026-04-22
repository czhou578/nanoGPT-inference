# Empty file
"""
To enforce structure, the pipeline needs to know where it is in the generation process. We will code a simple, hardcoded deterministic FSM for a basic schema (e.g., `{"name": "<STRING>", "age": <NUMBER>}`). 

Our states will look like this:
*   `STATE_0`: System is waiting to generate exactly `{"name": "`
*   `STATE_1`: System is generating string characters (waiting for a closing `"`)
*   `STATE_2`: System is waiting to generate exactly `, "age": `
*   `STATE_3`: System is generating sequential digits `[0-9]+`
*   `STATE_4`: System is waiting to generate exactly `}` and then `<EOS>`
"""

class FSM:
    def __init__(self):
        self.state = "STATE_0"
        self.transitions = {
            "STATE_0": {
                "{": "STATE_1",
            },
            "STATE_1": {
                "\"": "STATE_2",
            },
            "STATE_2": {
                "\"": "STATE_3",
            },
            "STATE_3": {
                "0": "STATE_4",
                "1": "STATE_4",
                "2": "STATE_4",
                "3": "STATE_4",
                "4": "STATE_4",
                "5": "STATE_4",
                "6": "STATE_4",
                "7": "STATE_4",
                "8": "STATE_4",
                "9": "STATE_4",
            },
            "STATE_4": {
                "0": "STATE_4",
                "1": "STATE_4",
                "2": "STATE_4",
                "3": "STATE_4",
                "4": "STATE_4",
                "5": "STATE_4",
                "6": "STATE_4",
                "7": "STATE_4",
                "8": "STATE_4",
                "9": "STATE_4",
            },
        }