import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNModel(nn.Module):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.out(x)

class Agent:
    def __init__(self, epsilon=1, epsilon_decay=0.998, epsilon_end=0.01, gamma=0.99):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, 2)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state).detach().cpu().numpy()
            action = np.argmax(q_values[0])

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return action

    def learn(self, experiences):
        states = torch.tensor([exp.state for exp in experiences], dtype=torch.float32).to(self.device)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([exp.next_state for exp in experiences], dtype=torch.float32).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32).to(self.device)

        current_q_values = self.model(states)
        next_q_values = self.model(next_states).detach()

        target_q_values = current_q_values.clone()
        for i in range(len(experiences)):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])

        self.optimizer.zero_grad()
        loss = self.criterion(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

    def load_if_exists(self, file_path):

        if os.path.isfile(file_path):
            self.load(file_path)

    def load(self, file_path):
        self.model.load_state_dict(torch.load(file_path))

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)
