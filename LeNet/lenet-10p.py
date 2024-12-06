import torch

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

# this is one of Hyper parameter, but let's select given below value
batch_size = 512

from torchvision.utils import make_grid
# this will help us to create Grid of images

import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from deap import base, creator, tools, algorithms
import struct

class LeNet5(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )



    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def accuracy(output, labels):
    _, preds = torch.max(output, dim = 1)

    return torch.sum(preds == labels).item() / len(preds)


device = get_default_device()
device

def eevaluate(model, loss_fn, val_dl, metric = None, device='cuda'):

    with torch.no_grad():

        results = [loss_batch(model, loss_fn, x, y, metric = metric) for x, y in val_dl]

        losses, nums, metrics = zip(*results)

        total = np.sum(nums)

        avg_loss = np.sum(np.multiply(losses, nums)) / total

        avg_metric = None

        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total

    return avg_loss, total, avg_metric

def loss_batch(model, loss_func, x, y, opt = None, metric = None):

    pred = model(x)

    loss = loss_func(pred, y)

    if opt is not None:

        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None

    if metric is not None:

        metric_result = metric(pred, y)

    return loss.item(), len(x), metric_result


model = torch.load('lenet.pth', map_location=device)

import torchvision
# transform is used to convert data into Tensor form with transformations
import torchvision.transforms as transforms


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trans = transforms.Compose([
    # To resize image
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    # To normalize image
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.MNIST(
root = './data',
train = True,
download = True,
transform = trans
)

test_set = torchvision.datasets.MNIST(
root = './data',
train = False,
download = True,
transform = trans
)


test_loader = DeviceDataLoader(DataLoader(test_set, batch_size=256), device)
result = eevaluate(model, F.cross_entropy, test_loader, metric = accuracy)
result
Accuracy = result[2] * 100
Accuracy
loss = result[0]
print("Total Losses: {}, Accuracy: {}".format(loss, Accuracy))


######....................parameter number............................######
layer_names = [name for name in model.state_dict().keys() if 'weight' in name]
num_weights_per_layer = {name: model.state_dict()[name].numel() for name in layer_names}
n = 0
for _ in layer_names:
  n = n + num_weights_per_layer[_]
print("Parameters Number:", n)


model = model.to('cuda')
# Define the evaluation function
def evaluate(individual):
    # print("Evaluating individual:", individual)
    model_copy = torch.load('lenet.pth')
    model_copy = model_copy.to('cuda')

###################################################################

    state_dict = model_copy.state_dict()
    for layer_name, weight_idx in individual:
        weight = state_dict[layer_name].view(-1).to('cuda')
        # print(weight)
        # weight[weight_idx] += 0.01  # Perturb the weight slightly
        
        ######...........................Injection..............................######
        #weight[weight_idx] = weight[weight_idx] + 0.1
        binary_value = struct.pack('!f', weight[weight_idx])
        int_value = struct.unpack('!I', binary_value)[0]
        bit_position = random.randint(0, 30)
        flipped_value = int_value ^ (1 << bit_position)
        flipped_binary = struct.pack('!I', flipped_value)
        weight[weight_idx] = struct.unpack('!f', flipped_binary)[0]


    # Load the perturbed weights back into the model
    model_copy.load_state_dict(state_dict)

    # Evaluate the perturbed model on a validation set
    model_copy.eval()
    result = eevaluate(model_copy, F.cross_entropy, test_loader, metric = accuracy)
    Accuracy = result[2] * 100
    loss = result[0]

    ########..............SOLVE INF,NAN PROBLEM..................########
    if loss == float('inf') or loss != loss:  # Check for infinity or NaN
        loss = 1e6  # Assign a large value if there's a problem


    # Return the loss as fitness (higher loss indicates more critical weight)
    return loss,

# Create the fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize the function
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def custom_mutate(individual, indpb):
    # print("Before mutation:", individual)
    for i in range(len(individual)):
        if random.random() < indpb:
            layer, index = individual[i]
            new_index = random.randint(0, num_weights_per_layer[layer] - 1)
            individual[i] = (layer, new_index)
    # print("After mutation:", individual)
    return individual,

def custom_crossover(ind1, ind2):
    # print("Before crossover:", ind1, ind2)
    tools.cxTwoPoint(ind1, ind2)
    # print("After crossover:", ind1, ind2)
    return ind1, ind2


# Attribute generator: (layer, index) pair
layer_names = [name for name in model.state_dict().keys() if 'weight' in name]
num_weights_per_layer = {name: model.state_dict()[name].numel() for name in layer_names}
def random_weight():
    layer = random.choice(layer_names)
    index = random.randint(0, num_weights_per_layer[layer] - 1)
    # print(layer,index)
    return (layer, index)

ind_size = int(0.0001 * n)
print("Individual Size =", ind_size)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, random_weight, n=ind_size)  # Each individual perturbs n weights
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the genetic operators
toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("mate", custom_crossover)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


from deap import tools, base, creator, algorithms

def eaSimpleWithDebugging(population, toolbox, cxpb, mutpb, ngen, stats=None,
                          halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        print(f"Generation {gen}")

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                # print(f"Before mate: {child1}, {child2}")
                toolbox.mate(child1, child2)
                # print(f"After mate: {child1}, {child2}")
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                # print(f"Before mutate: {mutant}")
                toolbox.mutate(mutant)
                # print(f"After mutate: {mutant}")
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def main():
    random.seed(42)

    # Create an initial population of 100 individuals
    population = toolbox.population(n=5)

    # Define statistics to keep track of the progress
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # Extract the first element of the fitness tuple
    stats.register("avg", np.mean)
    stats.register("min", min)
    stats.register("max", max)

    # Hall of Fame to keep the best individual
    hof = tools.HallOfFame(1)

    # Run the genetic algorithm
    population, logbook = eaSimpleWithDebugging(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                                                stats=stats, halloffame=hof, verbose=True)

    # Print the best individual
    print("Best individual is: ", hof[0])
    print("Fitness: ", hof[0].fitness.values[0])

if __name__ == "__main__":
    main()
