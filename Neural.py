
x = [ 0.05 , 0.10 ] # Input values
w1 = [ 0.3 , -0.2 , 0.4 , 0.1 ] # Ws values from 2n inputs to 2n in heddin 1 
b1  = 0.5 # bias for input - Hidden 
b2 = 0.7 # From Hidden to Output
w2 = [ -0.5 , 0.3 , 0.2 , 0.4 ]


"""
# Hidden layer calculation 
h1_pre = x[0]*w1[0] + x[1]*w1[1] + b1
h2_pre = x[0]*w1[2] + x[1]*w1[3] + b1

"""

# Make an Exponintial Calculation for the Tanh AF

def exp(x):
    result = 1.0
    term = 1.0
    for i in range (1,20):
        term *= x / i
        result += term 
    return result 


# Make a function for Tanh Activation Function 

def Tanh(x):
    positive = exp(x)
    negative = exp(-x)

    return (positive - negative) / (positive + negative)


# Hidden layer calculation 2n
h1_pre = x[0]*w1[0] + x[1]*w1[1] + b1
h2_pre = x[0]*w1[2] + x[1]*w1[3] + b1


# From Hidden to Output 2n
h1_act = Tanh(h1_pre) 
h2_act = Tanh(h2_pre) 


# Calculate oytput layer 2n 
o1 = h1_act*w2[0] + h2_act*w2[1] + b2
o2 = h1_act*w2[2] + h2_act*w2[3] + b2 

# Print results with 6 decimal places
print(f"Hidden 1 activation: {h1_act:.6f}")
print(f"Hidden 2 activation: {h2_act:.6f}")
print(f"\nOutput 1: {o1:.6f}")
print(f"Output 2: {o2:.6f}")