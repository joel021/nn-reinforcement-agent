import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNModel(nn.Module):
    def __init__(self, grid_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(grid_size**2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, grid_size, epsilon=1, epsilon_decay=0.998, epsilon_end=0.01, gamma=0.99):
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNModel(grid_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, 4)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            action = torch.argmax(q_values).item()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return action

    def learn(self, experiences):
        states = torch.tensor(np.array([e.state for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32, device=self.device)

        current_q_values = self.model(states)
        next_q_values = self.model(next_states).detach()

        target_q_values = current_q_values.clone()
        for i in range(len(experiences)):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        self.model.to(self.device)
