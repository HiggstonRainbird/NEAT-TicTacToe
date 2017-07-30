"""
Tic-Tac-Toe legal move finder.

Written July 30th, 2017.

The goal of this version is to make a tic-tac-toe bot that only makes legal moves.
The inputs consist of matricies representing boards:
    -1 as the opponent's moves
    +1 as the program's
    0 as blank spaces.
Legal moves consist of an output matrix with all 0 positions replaced with 1s, and all other values replaced with 0.

The fitness function is the sum of the squared differences between each output element and the correct output element, divided by nine.

"""

from __future__ import print_function
import os
import neat
import visualize

import random

tictactoe_inputs = [tuple([random.choice([-1,0,1]) for i in range(9)]) for j in range(25)]
tictactoe_outputs = [[int(i==0) for i in board] for board in tictactoe_inputs]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        numberOfBoards = float(len(tictactoe_inputs))
        genome.fitness = 1.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        #fitnessBoard = [[0 for j in range(len(tictactoe_inputs))] for i in range(9)]

        for xi, xo in zip(tictactoe_inputs, tictactoe_outputs):
            output = net.activate(xi)

            for i in range(9):
                genome.fitness -= float((xo[i] - output[i]) ** 2) / (9.0 * numberOfBoards)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
     neat.DefaultSpeciesSet, neat.DefaultStagnation,
     config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1000-1))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 5000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(tictactoe_inputs, tictactoe_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'i 1,1', -2: 'i 1,2', -3: 'i 1,3', -4: 'i 2,1', -5: 'i 2,2', -6: 'i 2,3', -7: 'i 3,1', -8: 'i 3,2', -9: 'i 3,3', 0: 'o 1,1', 1: 'o 1,2', 2: 'o 1,3', 3: 'o 2,1', 4: 'o 2,2', 5: 'o 2,3', 6: 'o 3,1', 7: 'o 3,2', 8: 'o 3,3'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-tictactoe')
    run(config_path)
