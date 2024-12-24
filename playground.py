

import pickle
import numpy as np
with open('frames_each_episode.pkl', 'rb') as f:
    data = pickle.load(f)
array = data.astype(np.int32)


# Save the array to a text file with a comma after each number
with open("array_with_commas.txt", "w") as file:
    file.write(str([num for num in array]))
