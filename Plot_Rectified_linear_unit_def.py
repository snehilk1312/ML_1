from matplotlib import pyplot

#rectified linear function
def Relu(x):
    return max(0,x)

input_=[i for i in range(-20,20)]

output_=[Relu(i) for i in input_]

pyplot.plot(input_,output_)
pyplot.show()


