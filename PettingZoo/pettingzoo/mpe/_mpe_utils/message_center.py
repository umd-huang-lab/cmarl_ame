import numpy as np
from collections import defaultdict
from typing import List

class Message:
    def __init__(self, sender, seek_id, share_id, position) -> None:
        self.sender_id = sender
        self.seeking = seek_id
        self.sharing = share_id
        self.position = position

    def show(self):
        print("""*Msg*: I am seeking landmark {}. I would like to share landmark {} 
                with position {}""".format(self.seeking, self.sharing, self.position))

class MessageCenter:
    def __init__(self) -> None:
        self.messages = defaultdict(list)
    
    def reset(self):
        self.messages = defaultdict(list)

    def send_message(self, from_id, to_id, seek_id, share_id, position):
        msg = Message(from_id, seek_id, share_id, position)
        self.messages[to_id].append(msg)
        # print(from_id, "sent", to_id, "a message:")
        # msg.show()

    def get_message(self, to_id):
        msgs = self.messages[to_id]
        if msgs:
            return msgs.pop(0)
        else:
            return None

    def get_all_messages(self, to_id):
        return self.messages[to_id]
    
    def decode_message(self, msg, filter, num_agents, dim_p):
        '''decode a message
        filter: the goal of the receiver, 
            accept the message only if the shared id is equal to the receiver's goal
        '''
        if msg:
            # print("received message")
            # msg.show()
            sender = np.zeros(num_agents)
            sender[msg.sender_id] = 1
            num_lms = num_agents
            goal = np.zeros(num_lms)
            goal[msg.seeking] = 1
            if msg.sharing == filter:
                # print("Decoded Msg:", np.concatenate([sender] + [msg.position]))
                return np.concatenate([sender] + [goal] + [msg.position])
            else:
                # if msg:
                #     print("Goal does not match")
                return np.concatenate([sender] + [goal] + [np.zeros(dim_p)])
        else:
            # print("No message found")
            return np.zeros(num_agents * 2 + dim_p)