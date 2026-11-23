# Debugging of CNN V1
# V1 is the original CNN
# cnn stream allows us to stream the outputs of the CNN for debugging
# cnn pool linear allowed us to verify the pooling layer where we think the error is
# cnn pool words outputs the conv3 -> pool layer divider inputs and outputs to see if the error is upstream or downstream of division
# cnn pool bytes checks if the movement of conv3 bytes from using the party bit to store the 9th bit in the conv3 output is shifted to another word fixes the computational error.
it did.
# cnn pool bytes is the final functioning version of the CNN V1