import random
import math
import numpy as np
import matplotlib.pyplot as plt

def setSimulatingCount(simulating_count):
    while(1):
        print('No. of games to be played:')
#         simulating_count = int(input())
        
        if simulating_count <= 0:
            print('You have to play the game more than once')
        else:
            break
            
    return simulating_count

def spin_roulette(kind_of_bet):
    slots = np.arange(38)
    reslt = 0
#     val = 1
    if(kind_of_bet == 0):
        possible_outcomes = np.random.choice(2, 38, p=[18/38, 20/38])
        index = random.choice(slots)
        reslt = possible_outcomes[index]
    if(kind_of_bet == 1):
        x = random.choice(slots)
        if(x <=19):
            reslt = 1
        else:
            reslt = 0
        
    return reslt

def gamble(initial_amount = 50, count=10000):
    Ei, Pi, Ci = [], [], []
    num_of_games = count
    played = 0
    print("number of games to be played", num_of_games)
    print("Initial amount:", initial_amount)
    current_amount = initial_amount
    N = 1000
    while(num_of_games != 0):
        print("Place bet on: 0 -> colour, 1 -> high/low number")
        kind_of_bet = np.random.choice(2)
        if(kind_of_bet == 0): #unequal prob
            print("Place bet. 0 -> Red, 1 -> Black")
            bet = np.random.choice(2)
            if(bet == 0):
                p = 18/38
                r = (1-p)/p
#                 pi = (1 - (r**current_amount))/(1 - (r**N))
#                 Pi.append(pi)
#                 ei = (current_amount/(1 - 2*p)) - (N/(1-2*p)) - (((r**current_amount)-1)/((r**N) - 1))
#                 Ei.append(ei)
            if(bet == 1):
                p = 20/38
                r = (1-p)/p
#                 pi = (1 - (r**current_amount))/(1 - (r**N))
#                 Pi.append(pi)
#                 ei = (current_amount/(1 - 2*p)) - (N/(1-2*p)) - (((r**current_amount)-1)/((r**N) - 1))
#                 Ei.append(ei)
        if(kind_of_bet == 1): #equal prob
            print("Place bet. [1, 19] for low and [20,38] for high")
            bet = np.random.choice(38) + 1
            p = 1/2
#             r = (1-p)/p
#             pi = current_amount/N
#             Pi.append(pi)
#             ei = current_amount*(N - current_amount)
#             Ei.append(ei)
        
        if(p == 0.5):
            pi = initial_amount/N
            ei = initial_amount * (N-initial_amount)
        else:
            pi = (1 - (r**initial_amount))/(1 - (r**N))
            ei = (initial_amount/(1 - 2*p)) - (N/(1-2*p)) - (((r**initial_amount)-1)/((r**N) - 1))
                
        outcome  = spin_roulette(kind_of_bet)
        
        win_count, win_flag = 0, 0
        if (bet == outcome):
            win_flag = 1
            win_count += 1
            current_amount += 1
        else:
            current_amount -= 1
            
        if(win_flag == 1):
            print("\t\tGambler won the bet.")
            print("Amount with gambler: {}, Profit: {}".format(current_amount, current_amount-initial_amount))
            print("------------------\n")
        if(win_flag == 0):
            print("\t\tGambler lost the bet.")
            print("Amount with gambler: {}, Loss: {}".format(current_amount, current_amount-initial_amount))
            print("------------------\n")
            
#             X.append(current_amount)
#             Y.append((current_amount/N))
        Ci.append(current_amount)
        num_of_games -= 1
        played += 1
        if(current_amount == 0 or current_amount == N):
            print("games played:", played)
            break
    print("Gambler has {} left.".format(current_amount), played)
    if(current_amount == 0):
        print("\nGambler is ruined!!")
    if(current_amount == N):
        print("\nGambler won!!")
    
    Ci, Pi, Ei = np.array(Ci), np.array(Pi), np.array(Ei)
#     return played, Ci, Pi, Ei
    return [played, pi, ei]

gamble(1000, 50)
# gamble(1000)

x, y = gamble(50, 150)
plt.plot(np.arange(x)+1, y, z)
plt.show()

plt.plot(y, np.arange(x))
plt.show()


ip = np.random.randint(10,100,10)
xx = []
for i in range(len(ip)):
    xx.append((gamble(ip[i])))
xx = np.array(xx)
xx[:,0]
plt.scatter(ip,xx[:,0])
# plt.plot(ip, xx[:,0])
plt.show()  #i vs played

xy = []
for i in range(len(ip)):
    xy.append((gamble(ip[i])))
    
xy = np.array(xy)

xz = []
iq = np.random.randint(100, 500, 50)
iq = np.sort(iq)
# iq
for i in range(len(iq)):
    xz.append((gamble(iq[i])))

    xz = np.array(xz)
plt.plot(iq, xz[:,0])
plt.xlabel('Initial Amount')
plt.ylabel('No. of games played till ruin')

plt.scatter(iq, xz[:,2])
plt.xlabel('Initial Amount')
plt.ylabel('Probability of winning')

plt.scatter(iq, xz[:,3])
plt.xlabel('Initial Amount')
plt.ylabel('Expected duration of game')


data = []
inpt = np.random.randint(100, 500, 1000)
inpt = np.sort(inpt)
for i in range(len(inpt)):
    data.append((gamble(inpt[i])))
data = np.array(data)

X = data[:,1:3]
y = data[:,0]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_train, y_train))
print(reg.score(X_test, y_test))
y_p = reg.predict(X_test)
print(reg.score(X_test, y_p))

plt.plot(inpt,data[:,0])
plt.xlabel('Initial Amount')
plt.ylabel('No. of games played till ruin')

plt.scatter(inpt, data[:,1])
plt.xlabel('Initial Amount')
plt.ylabel('Probability of winning')

plt.scatter(inpt, data[:,2])
plt.xla('Initial Amount')