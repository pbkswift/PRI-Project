# teaching.py

import datetime

def AddTwoNumbers(num1, num2):
    return num1 + num2;

sum = AddTwoNumbers(5,6)
print(sum)


def SayHelloWithTime(name):
    return "Hello " + name + ", the time is " + datetime.datetime.now().strftime("%H:%M:%S")

print(SayHelloWithTime("Prarthana"))
print(SayHelloWithTime("Ashwin"))

