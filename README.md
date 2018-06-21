# Pommerman X

![](https://media.giphy.com/media/9xySe58jgkoarfDWgO/giphy.gif)

## Introduction

**Q: What is happening here?**

**A:** Here we train and evaluate AI agents for the Pommerman competition

**Q: What is Pommerman?**

**A:** It is almost like the famous Bomberman, but written especially for the AI agents.
Rules are simple: on the board 11x11 you have three enemies, a lot of obstacles and a bomb. 
The main goal is to become a lone survivor. 
You can find more information [here](https://github.com/MultiAgentLearning/playground) and [there](https://www.pommerman.com/).

**Q: AI?**

**A:** Yes, and plenty of other buzzwords. 
Although the Pommerman hosts don't restrict the technology stack, we mainly used reinforcement learning to train our agents.

**Q: Who are you?**

**A:** We are Master's students of the [Institute of Computer Science in the University of Tartu](https://www.cs.ut.ee/et). 
This was our course project for the Introduction to Computational Neuroscience in spring 2017/18, which we enjoyed a lot.

NB! Although the course has ended we plan to maintain this repository and prepare ourselves for the next competition.
Therefore things could change here in a while.

**Q: How can I use it?**

**A:** You can use our notebooks to create your own agents and evaluate them in a comprehensive way. 
Also, [here]() we keep the evaluation results of our models, so we can compare our results.

**Q: Does everything work?**

**A:** We sincerely hope so! Feel free to create an issue if you discover any problem. Note that you should specify some paths manually

## Installation 

We encourage you to use conda environment for the project. You can create it with all required dependencies like this:

```
conda create --name pommenv --file spec-file.txt
```

You will also need the Pommerman library, you can install it using the bash script:

```
bash ./setup.sh
```

Or you can manually clone the repo and specify the path to it:

```
git clone https://github.com/MultiAgentLearning/playground.git
```

## Current results

We managed to achieve Simple Agent (the baseline heuristics) performance on average and overcom it in a top-left corner of the board

Average comparison:

![](https://i.imgur.com/NCg3fO3.png)

In-corner comparison:

![](https://i.imgur.com/SO3iAV0.png)

The best imitation one is AlphaGo-inspired architecture, the best RL is an Atari-inspired model 
(however AlphaGo performed better in the top-left corner).
From the episode length plot you can see that our trained agents are mostly cowards, they try to avoid bombs and enemies to survive as long as possible. 
We have just began to solve this issue.

## Links

* [project presentation](https://docs.google.com/presentation/d/10MCSWuFEfYOpVyUePKt8Rmv49Q33xiP-W3BbjTrFgN8/edit?usp=sharing)
* [project report]

## Contributors

* [Novin Shahroudi](https://github.com/novinsh)
* [Mikhail Papkov](https://github.com/papkov)
* [Anton Potapchuk](https://github.com/AntonPotapchuk)
* [Sofiya Demchuk](https://github.com/SofiyaDemchuk)

## Supervisor

* [Tambet Matiisen](https://github.com/tambetm)
