import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.data import DataLoader
from pytorch_lightning import Callback
import wandb
from utils import train


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RLAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.target_net = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_net(state)
        return np.argmax(action_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class ValidRunsLogger(Callback):
    def on_epoch_end(self, trainer, pl_module):
        valid_runs = sum([len(batch) > 0 for batch in trainer.train_dataloader])
        print(f"Epoch {trainer.current_epoch}: Number of valid runs: {valid_runs}")
        wandb.log({"valid_runs": valid_runs})


def get_state(model, dataset, channel_subsets):
    state = []
    for channels in channel_subsets:
        dataset.channels = channels
        dataloader = DataLoader(
            dataset, batch_size=len(channels) // 2, shuffle=False, num_workers=4
        )
        accuracy = evaluate_model(model, dataloader)
        state.append(accuracy)
    return np.array(state)


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            if data is None:
                continue
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0
    return accuracy


def evaluate_action(model, dataset, action, channel_subsets):
    initial_accuracy = get_state(model, dataset, channel_subsets)
    selected_channels = channel_subsets[action]
    dataset.channels = selected_channels
    dataloader = DataLoader(
        dataset, batch_size=len(selected_channels) // 2, shuffle=False, num_workers=4
    )
    new_accuracy = evaluate_model(model, dataloader)
    reward = new_accuracy - initial_accuracy[action]
    new_state = get_state(model, dataset, channel_subsets)
    return reward, new_state


def update_channels(current_channels, action, channel_subsets):
    if action < len(channel_subsets):
        selected_channels = channel_subsets[action]
        current_channels = selected_channels
    return current_channels


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def run_multi_agent_system(
    model_name,
    dataset,
    channel_subsets,
    state_size,
    action_size,
    n_agents=3,
    n_iterations=5,
    epochs=10,
):
    agents = [RLAgent(state_size, action_size) for _ in range(n_agents)]
    initial_channels = channel_subsets[:state_size]
    channels = initial_channels

    for iteration in range(n_iterations):
        models = train(model_name, [channels], epochs)
        model = models[0]
        state = get_state(model, dataset, channel_subsets)

        for agent in agents:
            action = agent.act(state)
            reward, new_state = evaluate_action(model, dataset, action, channel_subsets)
            done = iteration == n_iterations - 1
            agent.remember(state, action, reward, new_state, done)
            agent.replay()
            agent.update_target_net()
            channels = update_channels(channels, action, channel_subsets)

    wandb.log({"final_accuracy": get_state(model, dataset, channel_subsets)})
