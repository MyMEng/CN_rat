# Load data
#  for loading neuron 1 - 4 and time
N = []
for n in range(1,5) :
	f = open("data/neuron"+str(n)+".csv")
	N.append( map( lambda x: int(x.strip()), f.readlines() ) )
#  for loading x and y
D = []
for i in ['x', 'y'] :
	f = open("data/"+i+".csv")
	D.append( map( lambda x: float(x.strip()), f.readlines() ) )
#  for time
f = open("data/time.csv")
T = map( lambda x: float(x.strip()), f.readlines() )


