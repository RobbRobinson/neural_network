from numpy import exp, array, random, dot

class neural_network:
    def __init__(self):
        random.seed(1)
        self.weights = 2 * random.random((2,1))-1 # could have also used random._sample()

    def train(self, inputs, outputs, num):
        for _ in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = 0.01 * dot(inputs.T, error)
            self.weights += adjustment

    def think(self, inputs):
        return (dot(inputs, self.weights))



if __name__ == '__main__':

    Neural_Net = neural_network()
    inputs = array([[2,3], [1,1],[5,2],[12,3]]) # given a set of inputs (underpants)
    outputs = array([[10,4,14,30]]).T # and a set of expected or desired outputs (profit)
    Neural_Net.train(inputs, outputs, 1000000) # the neural net can now repeat n times (figure out step 2)

    # to learn what to expect the output to look like.
    # without ever knowing the formula.
    print(Neural_Net.think((array([5321564,3165132]))))

# the given relationship between the data was 'output = (a+b) * 2'
# but the neural net was never explicitly told what it was
# and yet it can predict what the out of [a,b] should look like?
# magic is the only explanation...

# further experimentation shows that the larger the numbers as inputs the accuracy decreases
# EX : [a,b] = [5321564,3165132]... output = 16,973,392; but the real answer is 16,973,374.
# increasing computation by a factor of 10 (i.e. adding a zero to the n times repeated), showed no improvement of accuracy.

