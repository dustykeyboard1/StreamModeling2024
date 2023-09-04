'''
Author:
Michael Scoleri
Violet Shi 
James Gallagher 

Data: 09/03/2023

Filename: graphing_example.py

Function: 
Demonstrates graphing capabilities of software and saves to a PDF. 
Includes a coninuous heatmap,
'''

#Imports
import plotly.express as px
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

def create_heatmap():
    '''
    Parameters: none
    Functionality: Creates a random continous heatmap
    Return: Figure with heatmap
    '''

    random.seed()

    harvest = [[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0]]
        
    harvest = []
    for x in range(1000):
        newlist = []
        for i in range(1000):
            newlist.append(random.random())
        harvest.append(newlist)

    fig = px.imshow(harvest)
    # fig.show()
    return fig


def plot_random_lines():
    
    '''
    Create a figure with 6 random line plots
    Params: None
    Returns: Figure with 6 lines plots
    '''

    fig, axs = plt.subplots(3, 2, figsize=(6, 4))
    axs = axs.flatten()
    
    # Generate 6 random line plots
    for ax in axs:
        x = np.linspace(0, 10, 100)  
        y = np.random.rand(100) * 10 
        ax.plot(x, y)
        ax.set_title('Random Line Plot', size='smaller')
        ax.set_xlabel('X-axis', size='smaller')
        ax.set_ylabel('Y-axis', size='smaller')
        
    plt.tight_layout(pad=0.2)
    
    return fig


import matplotlib.pyplot as plt
import numpy as np

def plot_random_bar_chart():
    
    '''
    Create a random bar chart with positive and negative values
    Params: None
    Returns: Figure with bar chart
    '''
    
    fig, ax = plt.subplots()

    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    values = np.random.randint(-10, 11, len(labels))  

    bars = plt.bar(labels, values, color=['r' if x < 0 else 'g' for x in values])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom' if yval < 0 else 'top')

    plt.title('Random Bar Chart with Positive and Negative Values')
    plt.xlabel('Labels')
    plt.ylabel('Values')

    return fig


def plot_random_cosine_sine():
    '''
    Create a single plot with multiple trig functions.
    Params: None
    Returns: Figure with plot
    '''
    fig, ax = plt.subplots()

    # Generate x-values
    x = np.linspace(0, 2*np.pi, 100)
    num_plots = 2

    for i in range(num_plots):
        # Generate random amplitude and frequency for sine and cosine functions
        amp_sin = np.random.uniform(0.5, 2)
        freq_sin = np.random.uniform(1, 3)

        amp_cos = np.random.uniform(0.5, 2)
        freq_cos = np.random.uniform(1, 3)

        # Calculate the sine and cosine values
        y_sin = amp_sin * np.sin(freq_sin * x)
        y_cos = amp_cos * np.cos(freq_cos * x)

        ax.plot(x, y_sin, label=f'Sine {i+1}', linewidth=0.8)
        ax.plot(x, y_cos, label=f'Cosine {i+1}', linewidth=0.8)

    plt.title('Random Sine and Cosine Functions')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    return fig


def plot_random_decreasing_scatter_and_trig():
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    
    # Create a random decreasing scatter plot with a line of best fit
    x1 = np.linspace(0, 10, 100)
    y1 = -2 * x1 + 5 + np.random.normal(scale=2, size=len(x1))
    
    axs[0].scatter(x1, y1, label='Scatter Points')
    
    coef = np.polyfit(x1, y1, 1)
    y1_fit = coef[0] * x1 + coef[1]
    
    axs[0].plot(x1, y1_fit, label='Line of Best Fit', linestyle='--')
    
    axs[0].set_title('Random Decreasing Scatter with Line of Best Fit')
    axs[0].set_xlabel('X-axis')
    axs[0].set_ylabel('Y-axis')
    axs[0].legend()

    # Create a plot with a trigonometric function
    x2 = np.linspace(0, 4 * np.pi, 100)
    y2 = np.sin(x2) + np.cos(2 * x2)
    
    axs[1].plot(x2, y2, label='Sin(x) + Cos(2x)')
    
    axs[1].set_title('Trigonometric Function')
    axs[1].set_xlabel('X-axis')
    axs[1].set_ylabel('Y-axis')
    axs[1].legend()
    
    plt.tight_layout()
    
    return fig

def main():
    heatmap = create_heatmap()
    lineplots = plot_random_lines()
    barchart = plot_random_bar_chart()
    trig_plots = plot_random_cosine_sine()
    scatter_plot = plot_random_decreasing_scatter_and_trig()
    write_to_pdf(heatmap, lineplots, barchart, trig_plots, scatter_plot)

if __name__ == '__main__':
    main()