# PokerBot

## Abstract


Main goal of the project is to present author’s implementation of the Deep CFR algorithm with 
the Heads Limit Poker Texas Hold’em. 
It is a modern method for creating artificial intelligence in large partial-observable games.
Such environments have always been a great challenge and the main barrier  
to the development of machine learning. The project will show this problem 
by implementation of Deep CFR and analysis of the results.
For this purpose, five recognition models were created every 10 iterations of the algorithm.
Next step was to create games with that models. The results allowed to select the 
best playing models and to see how Deep CFR learned over time.
Additionally, the quality of the models was tested against a simple
program which simulate beginner player.

1. [Goal](#goal)
2. [Technologies](#Technologies)
3. [Build](#Build)
4. [Results](#Results)
5. [Current page layout](#layout)

## Goal <a name="goal"></a>

- [x] Implementation of Deep CFR (*Counterfactual Regret Minimization*) and MCCFR ES (*Monte  Carlo  Conterfactual  
  RegretMinimization*).
- [x] Create emulator for HULH (*Heads Up Limit Texas Poker Hold’em*) by [PyPokerEngine](https://github.com/ishikota/PyPokerEngine)
- [x] Train AI by 50 iterations and create 5 models ([M10](./models/M10), [M20](./models/M20), [M30](./models/M30),
    [M40](./models/M40), [M50](./models/M50)) every 10 epochs.
- [ ] Implementation RNN
- [ ] Train MCCFR ES for many rounds

## Technologies <a name="technologies"></a>

For local development all libraries can be installed with command:


    $ pip install -r requirements.txt 


   Neural Network             | DCFR                        | HULH
------------------------------|------------------------------|------------                                                                     
   tensorflow 2.6                | tqdm                        | pypokerengine
   numpy 1.21                    | numpy 1.21                  | numpy 1.21

## Build <a name="Build"></a>

### Neural Network Architecture


![alt text](./praca/img/nn.pdf?raw=true)

### Replay Memory


![alt text](./praca/img/bzd.pdf?raw=true)

### HULH parametrs

   parametr                         | value                        
------------------------------------|----------------------------                                                                 
ante                                | 0                       
small blind                         | 5
big blind                           | 10
reset environment after this round  | 1
stock                               | 80
number of players                   | 2

### DCFR parametrs


   parametr                         | value                        
------------------------------------|----------------------------                                                                 
iterations of MCCFR ES              | 270
DCFR iterations                     | 50
save model each iterations          | 10

### UML

![alt text](./praca/img/uml.pdf?raw=true)

## Results  <a name="Results"></a>

![alt text](./praca/img/mecze.pdf?raw=true)
![alt text](./praca/img/mecze_ps.pdf?raw=true)
![alt text](./praca/img/mecze_pw.pdf?raw=true)
![alt text](./praca/img/akcje.pdf?raw=true)


