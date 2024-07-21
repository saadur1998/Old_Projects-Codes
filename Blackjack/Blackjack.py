#!/usr/bin/env python
# coding: utf-8

# In[1]:


suits = ('Hearts', 'Diamonds', 'Spades', 'Clubs')
ranks = ('Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace')
values = {'Two':2, 'Three':3, 'Four':4, 'Five':5, 'Six':6, 'Seven':7, 'Eight':8, 
            'Nine':9, 'Ten':10, 'Jack':10, 'Queen':10, 'King':10,'Ace':11}
import random


# In[2]:


class Card:
    
    def __init__(self,suit,rank):
        self.suit = suit
        self.rank = rank
        try:
            self.value = values[rank]
        except:
            pass
        
    def __str__(self):
        return self.rank + ' of ' + self.suit


# In[3]:


class Deck:
    
    def __init__(self):
        self.deck = [] 
        for suit in suits:
            for rank in ranks:
                self.deck.append(Card(suit,rank))
                
    def shuffle(self):
        random.shuffle(self.deck)
        
    def deal_one(self):
        return self.deck.pop(0)


# In[4]:


class Player:
    
    def __init__(self):
        pass
    
    def hit(self):
        self.card1= deck.deal_one()
        return self.card1
    
    def hit_again(self):
        self.card2= deck.deal_one()
        return self.card2


# In[5]:


class BankRoll:
    balance=500
    def __init__(self):
        pass
        
    def bet(self):
        while True:
            try:
                self.amount = int(input('Enter the amount you wish to bet'))
                if self.amount > BankRoll.balance:
                    print('Not Enough Balance')
                else:
                    break
            except:
                print('please enter the correct amount')
    def loss():
        BankRoll.balance = BankRoll.balance - self.amount
        return self.balance
    
    def win():
        BankRoll.balance = BankRoll.balance + self.amount
        return BankRoll.balance
    
    def __str__(self):
        return f'Your Balance is {BankRoll.balance}'


# In[6]:


def gameon():  
    choice = 'wrong'
    while choice not in ['Y','N']:
        choice = input("Would you like to keep playing? Y or N ")
        if choice not in ['Y','N']:
            clear_output() 
            print("Sorry, I didn't understand. Please make sure to choose Y or N.")
    if choice == "Y":
        return True
    else:
        return False


# In[ ]:


game_on=True
bank=BankRoll()
while game_on:
    deck=Deck()
    deck.shuffle()
    player=Player()
    dealer=Player()
    player_list=[]
    dealer_list=[]
    v=0
    
# Assiging a Card to Dealer and Printing it

    print("Dealer's, 1st cards and it's values is")
    dealer_list.append(dealer.hit())
    print(dealer_list[0], end=' : ')
    print(f'{dealer_list[0].value}\n')
    print('Player 1, your cards and their values are')
    v1=dealer_list[0].value
    
# Assiging 2 cards to player and priting them and their total

    for i in range(2):
        player_list.append(player.hit())
        print(player_list[i], end=' : ')
        if player_list[i].rank != 'Ace':
            print(player_list[i].value)
        else:
            print('You will chose the value of Ace pretty soon')
    for i in player_list:
        if i.rank == 'Ace':
            while True:
                a=input('Please Chose Value of Ace as either 1 or 11 : ')
                if a == '1' or a == '11':
                    v=v+int(a)
                    break
                else:
                    print('Ace can only have two values : 1 or 11')
        else:
            v = v + i.value
    print(v)
    
# Asking player whether they want to hit or stay
    choice=True
    while choice:
        if v<=21:
            print(f"Total Value of Your Cards is {v}")
            b = input('Do you want to Hit or Stay : H or S : ')
            if b == 'H':
                player_list.append(player.hit())
                print('Your Card is')
                print(player_list[-1], end=' : ')
                if player_list[-1].rank != 'Ace':
                    print(player_list[-1].value)
                    v=v+player_list[-1].value
                else:
                    while True:
                        a=input('\nPlease Chose Value of Ace as either 1 or 11 : ')
                        if a == '1' or a == '11':
                            v=v+int(a)
                            break
                        else:
                            print('Ace can only have two values : 1 or 11')
            elif b == 'S':
                break
            else:
                print('Please Chose only H or S')   
        else:
            print(f'YOU SCORE IS {v}')
            print('BUSTEDDD !!! YOU LOST')
            game_on=gameon()
            choice=False
        
    while choice:
        if v1<v:
            dealer_list.append(player.hit())
            print("Dealer's New Card is")
            print(dealer_list[-1], end=' : ')
            print(dealer_list[-1].value)
            v1=v1+dealer_list[-1].value
            print(f"Total Value of Dealer's Card is {v1}")
        elif v1<=21:
            print("Dealer Won !!!!\nYOU LOST")
            game_on=gameon()
            break
        else:
            print("Dealer Lost !!\n YOUR WON !!!!")
            game_on=gameon()
            break


# In[ ]:




