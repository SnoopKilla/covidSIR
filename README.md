# About the Project
In this project I model the spread of COVID infection across a country with a SIR (Susceptible-Infected-Recovered) parametric model and use Bayesian inference (specifically, Markov Chain Monte Carlo methods) to estimate the value of the parameters. A detailed description of the model assumptions and the methodology adopted can be found [here](https://github.com/SnoopKilla/covidSIR/blob/main/SIR_model.pdf).

# Live Demo
I created a [live demo](https://huggingface.co/spaces/SnoopKilla/covidSIR) of the model on HuggingFace spaces. The user can select a country of interest, the desired number of breakpoints and the number of iterations/burn-in for the Markov Chain and the results of the estimate will be provided as outputs.
