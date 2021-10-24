import json
import time

class Json_creator:

    def __init__(self):
        self.__data = {"name": "root"}

    def clear_tree(self):
        self.__data = {"name": "root"}
        
    def update_data(self, events, value, timestep):
        actions = []
        history = events[0]["round_state"]["action_histories"]
        for r in history.keys():
            for action in history[r]:
                act = action['action']
                uuid = action['uuid']
                actions.append(f'{str(uuid)[0]}: {str(act)[0]}')
        ref_data = self.__data
        self.__data = self.__step_deeper(ref_data, actions, 0, value, timestep)

    def __step_deeper(self, data, actions, id, timestep, value):
        if id == timestep and not '=' in data['name']:
            data['name'] = data['name']+'='+str(round(value, 2))

        if id == len(actions):
            return data

        if not 'children' in data:
            data['children'] = []

        for idx in range(len(data['children'])):
            if data['children'][idx]["name"].split('=')[0] == actions[id]:
                data['children'][idx] = self.__step_deeper(data['children'][idx], 
                                                           actions, 
                                                           id+1,
                                                           timestep,
                                                           value)
                return data

        data['children'].append({"name": actions[id]})
        data['children'][-1]['parent'] = data['name']
        data['children'][-1] = self.__step_deeper(data['children'][-1],
                                                   actions,
                                                   id+1,
                                                   timestep,
                                                   value)
        return data
        

    def save_file(self, path):
        with open(path, "w") as file:
            json.dump([self.__data], file)

            
        


            



