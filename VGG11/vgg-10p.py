import os

import torch
import torch.nn as nn
import timeit
import struct


######################## ADAPT ########################
#from adapt.approx_layers import axx_layers as approxNN

#set flag for use of AdaPT custom layers or vanilla PyTorch
#use_adapt=True

#set axx mult. default = accurate
#axx_mult_global = 'mul8s_acc'
#adaPT_conv2d_counter = 0
#repeat = 0
#######################################################

  
__all__ = [
    "VGG",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
]


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # CIFAR 10 (7, 7) to (1, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            # nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    #global adaPT_conv2d_counter
    #global repeat
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, device, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        # script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
             arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def vgg11_bn(pretrained=False, progress=True, device='cuda', axx_mult = 'mul8s_acc', **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """   
    global axx_mult_global
    axx_mult_global = axx_mult
    
    return _vgg("vgg11_bn", "A", True, pretrained, progress, device, **kwargs)


def vgg13_bn(pretrained=False, progress=True, device='cuda', axx_mult = 'mul8s_acc' , **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """   
    global axx_mult_global
    axx_mult_global = axx_mult
    
    return _vgg("vgg13_bn", "B", True, pretrained, progress, device, **kwargs)


def vgg16_bn(pretrained=False, progress=True, device="cpu", axx_mult = 'mul8s_acc', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """    
    global axx_mult_global
    axx_mult_global = axx_mult
    
    return _vgg("vgg16_bn", "D", True, pretrained, progress, device, **kwargs)


def vgg19_bn(pretrained=False, progress=True, device="cpu", axx_mult = 'mul8s_acc', **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    global axx_mult_global
    axx_mult_global = axx_mult
    
    return _vgg("vgg19_bn", "E", True, pretrained, progress, device, **kwargs)



import os
import zipfile
import torch
import io

import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn
import random

model = vgg11_bn(pretrained=True)

# model.eval()

def val_dataloader(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)):

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = CIFAR10(root="datasets/cifar10_data", train=False, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        shuffle= False,
        batch_size=128,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )
    return dataloader

transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)),
        ]
    )
dataset = CIFAR10(root="datasets/cifar10_data", train=True, download=True, transform=transform)

evens = list(range(0, len(dataset), 10))
trainset_1 = torch.utils.data.Subset(dataset, evens)

data = val_dataloader()


import timeit
correct = 0
total = 0

model.eval()
start_time = timeit.default_timer()
with torch.no_grad():
   for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
      images, labels = images.to("cpu"), labels.to("cpu")
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
print(timeit.default_timer() - start_time)
print('Accuracy of the network on the 10000 test images: %.4f %%' % (
   100 * correct / total))


######....................parameter number............................######
layer_names = [name for name in model.state_dict().keys() if 'weight' in name]
num_weights_per_layer = {name: model.state_dict()[name].numel() for name in layer_names}
n = 0
for _ in layer_names:
  n = n + num_weights_per_layer[_]
print("Parameters Number:", n)




import random
import numpy as np
from deap import base, creator, tools, algorithms


def testt(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for iteraction, (images, labels) in enumerate(data):
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()        

    return(100 * correct / total)


model = model.to('cuda')
result = testt(model)
print(result)

model = model.to('cuda')
# Define the evaluation function
def evaluate(individual):
    # print("Evaluating individual:", individual)
    model_copy = vgg11_bn(pretrained=True)
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
        bit_position = random.randint(0, 31)
        #while(bit_position == 30):
            #bit_position = random.randint(0, 31)
        flipped_value = int_value ^ (1 << bit_position)
        flipped_binary = struct.pack('!I', flipped_value)
        weight[weight_idx] = struct.unpack('!f', flipped_binary)[0]


    # Load the perturbed weights back into the model
    model_copy.load_state_dict(state_dict)

    # Evaluate the perturbed model on a validation set
    model_copy.eval()
    result = testt(model_copy)
    Accuracy = result
    loss = 92.3878 - Accuracy


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

alpha = 0.0001
ind_size = int(alpha * n)
print("Individual Size =", ind_size)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, random_weight, n=ind_size)  # Each individual perturbs 5 weights
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
    start_time2 = timeit.default_timer()
    random.seed(42)

    # Create an initial population of 100 individuals
    population = toolbox.population(n=100)

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
    print('Genetic Time:', timeit.default_timer() - start_time2)

if __name__ == "__main__":
    main()
