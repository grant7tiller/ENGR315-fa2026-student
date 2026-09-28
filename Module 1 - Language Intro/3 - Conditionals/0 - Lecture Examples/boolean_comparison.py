# some random variable to compare
import random
random_num = random.randint(0, 67)

t = random_num

print("Using t = "+str(t))

# perform boolean equality comparison
print(str(t)+" == 40 is " + str(t == 40))

# perform boolean not equality
print(str(t)+" != 40 is " + str(t != 40))

# perform greater than or equal
print(str(t)+" >= 40 is " + str(t >= 40))

# perform greater than
print(str(t)+" > 40 is " + str(t > 40))

# perform less than
print(str(t)+" < 40 is " + str(t < 40))
