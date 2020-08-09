# Only used once to change the delimiter from whitespace to commas
import io
with io.open('C://users/dwsou/downloads/cleveland.data', "r", newline=None) as fd:
    f = open('C://users/dwsou/downloads/cleveland_edited.data', 'w')
    for line in fd:
        words = line.split()
        if words[-1] != 'name':
            line = line.replace("\n", "")
        line2 = ','.join(words)
        f.write(line2)
    f.close()
    fd.close()

# read the csv into a pandas dataframe
import pandas as pd
d = pd.read_csv('C://users/dwsou/downloads/cleveland_edited.data', header=None, engine='python', delimiter= ',')
print(d.head())

# select only the wanted columns from the dataframe and rename the columns
cols = [2, 3, 8, 9 , 11 ,15,18,31,37,39,40,43,50,58]
heart = d.iloc[:,cols]
heart = heart.rename(columns = {2: 'age', 3: 'sex', 8: 'cp', 9: 'trestbps', 11: 'clol', 15: 'fbs', 18: 'restecg', 
                               31: 'thalach', 37: 'exang', 39: 'oldpeak', 40: 'slope', 43: 'ca', 50: 'thal', 58: 'target'})

#change the target column from values 0-4 to binary data
heart.target[heart.target > 0] = 1
print(heart.head())

#save the dataframe as a csv file
heart.to_csv(r'C://users/dwsou/downloads/Heart.csv', index=False)