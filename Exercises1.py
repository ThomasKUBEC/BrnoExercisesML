import random
import numpy as np
import matplotlib.pyplot as plt 
import time
import pandas as pd


print("Exercise 1.4.1")
result = ""
for x in range(10):
    if(x<5):
        result += "x"            
    else:
        result = result[1:]
    print(result)


print("\nExercise 1.4.2")
input_str = "n45as29@#8ss6"
result = 0
for i in range(0,len(input_str)):
    if(input_str[i].isnumeric()):
        result += int(input_str[i])
print(result)
   

print("\nExercise 1.4.3")
number = random.randint(0,100)
result = ""
print("number:", number)
while(number!=0):
    if(number % 2 == 1):
        result = "1" +result
    else:
        result = "0" + result
    number = number // 2
    print(number)
print("result:", result)


print("\nExercise 1.5.1")
def fibonaci(upper_threshold: int) -> list:
    list = [0]
    if(upper_threshold >= 1):
        list.append(1)
    i=2
    while (upper_threshold > 1):
        temp = list[i-2] + list[i-1]
        if(temp <= upper_threshold):
            list.append(temp)
        else:
            break
        i += 1
    return list

print(fibonaci(10))


print("\nExercise 1.5.2")
def displayAsDigi(number: int) -> None:
    numbers = { 1: ['  x','  x','  x','  x','  x'],
                2: ['xxx','  x','xxx','x  ','xxx'],
                3: ['xxx','  x','xxx','  x','xxx'],
                4: ['x x','x x','xxx','  x','  x'],
                5: ['xxx','x  ','xxx','  x','xxx'],
                6: ['xxx','x  ','xxx','x x','xxx'],
                7: ['xxx','  x','  x','  x','  x'],
                8: ['xxx','x x','xxx','x x','xxx'],
                9: ['xxx','x x','xxx','  x','  x'],
                0: ['xxx','x x','x x','x x','xxx']}
    numberDigi = ''
    for i in range(5):
        for x in str(number):
            numberDigi += numbers[int(x)][i] + "  "
        numberDigi += '\n'
    print(numberDigi)

displayAsDigi(8765942310)


print("\n\n\nExercise 2\n1---")
x = np.arange(25,0,-1).reshape(5,5)
print(x)

def cutMatrixLoop(userThreshold: int, matrix):
    t1 = time.time()
    for index, x in np.ndenumerate(matrix):
        if(x <= userThreshold):
            matrix[index] = 0
    print("Execution time : ",time.time() - t1)
    return matrix

print(result)
t1 = time.time()
print("The threshold is 12 for loop\n", cutMatrixLoop(12,x))
print("Execution time : ",time.time() - t1)


print(np.where(x>20, x, -0))


print("\n2---")

def show_in_digi(input_integer: int) -> None:
    numbersBool = {
                1: [[False,False,True],[False,False,True],[False,False,True],[False,False,True],[False,False,True]],
                2: [[True,True,True],[False,False,True],[True,True,True],[True,False,False],[True,True,True]],
                3: [[True,True,True],[False,False,True],[True,True,True],[False,False,True],[True,True,True]],
                4: [[True,False,True],[True,False,True],[True,True,True],[False,False,True],[False,False,True]],
                5: [[True,True,True],[True,False,False],[True,True,True],[False,False,True],[True,True,True]],
                6: [[True,True,True],[True,False,False],[True,True,True],[True,False,True],[True,True,True]],
                7: [[True,True,True],[False,False,True],[False,False,True],[False,False,True],[False,False,True]],
                8: [[True,True,True],[True,False,True],[True,True,True],[True,False,True],[True,True,True]],
                9: [[True,True,True],[True,False,True],[True,True,True],[False,False,True],[False,False,True]],
                0: [[True,True,True],[True,False,True],[True,False,True],[True,False,True],[True,True,True]],
                10: [[False,False],[False,False],[False,False],[False,False],[False,False]]} 
    result = numbersBool[10]
    for x in str(input_integer):
        result = np.concatenate((result,numbersBool[int(x)]), axis = 1)
        result = np.concatenate((result,numbersBool[10]), axis = 1)

    plt.imshow(result,cmap= 'binary')    
    plt.show()

show_in_digi(2081)
