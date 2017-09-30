<h1> Super Mario Brothers Deep Q Network </h1>

<h4>Super Mario Brothers Deep Q Network is a Reinforcement Learning module that aims to make it easier to run experiments with the goal of beating levels in Super Mario Brothers (1984)</h4>

<h5>This project utilizes the wonderful work provided by ppaquette (https://github.com/ppaquette/gym-super-mario) which provides the lua files and an NES environment object which can work with openAI gym to allow you to interact with FCEUltra's emulator, sending controller commands and reading memory values from the game, which include things like screen pixels, score, level, etc. Follow his instructions for installing the package.</h5>

<h5>Note, however, that for this project you will want to heavily modify the file super_mario_bros.py, because it is here that you can define custom reward functions for things like Mario dying, eating mushrooms, etc. Reward design is up to you. I have provided a few examples of how to use the info object in the custom super_mario_bros.py file in the function _process_data_message().</h5>

<h5>The learner itself uses a Deep Q Network with a target network, and prioritized SARST replay memory as per the groundbreaking paper by DeepMind. The prioritized SARST memory is efficiently implemented using a SumTree to provide logarithmic probability access to samples that have larger rewards associated with them</h5>

<h5>The Deep Q Network uses a convolutional network to read screen pixels, converting the game from RGB to a single channel to save on computational resources.</h5>

<h3>Usage:</h3>
<ol>
<li>Install necessary packages below</li>
<li>Install ppaquette's Super Mario Bros package linked above to hook it into OpenAI Gym</li>
<li>Define custom super_mario_bros.py rewards and whatever else you feel you want access to at runtime and copy this file into the gym environment for Super Mario Bros. On my OS, this lives in /usr/local/lib/python2.7/dist-packages/gym/envs/ppaquette_gym_super_mario/</li>
<li>Open run_experiment.ipynb and execute the blocks. Note that the default is to run on GPU, so you might want to change that line.</li>
</ol>

<h3>Requirements:</h3>
<ol>
<li>Tensorflow >= 1.0</li>
<li>Numpy >= 1.12</li>
<li>OpenAI Gym >= 0.8</li>
<li>Matplotlib (for visualizations, not necessarily crucial)</li>
</ol>
