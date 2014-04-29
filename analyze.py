#! /usr/bin/python

# import
from math import ceil
from pprint import pprint
import matplotlib.pyplot as plt

# Round up to nearest 10
def round10( x ) :
	if x%10 == 0 :
		return x
	else :
		return int( ceil( x/10.0 ) * 10 )

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

# Get maze size
Xmin = round10( min( D[0] ) )
Xmax = round10( max( D[0] ) )
Ymin = round10( min( D[1] ) )
Ymax = round10( max( D[1] ) )

# Get number of cells and Divide into them
Visited = []
TimeAtPosition = []
Xsize = Xmax /10
Ysize = Ymax /10
for i in range(Xsize) :
	Visited.append([])
	TimeAtPosition.append([])
	for j in range(Ysize) :
		Visited[i].append([])
		TimeAtPosition[i].append([])

AtPosition = []
PositionChange = [-1]
# put times at which the mouse was at given square
for i, (x, y) in enumerate( zip(D[0], D[1]) ) :
	currentX = int( x/10 )
	currentY = int( y/10 )
	TimeAtPosition[currentX][currentY].append(T[i])
	AtPosition.append( (currentX, currentY) )
	# if last position differ from current one add | otherwise skip
	if PositionChange[-1] != (currentX, currentY) :
		PositionChange.append( (currentX, currentY) )
	# check if visited and so make something
	if Visited[currentX][currentY] == [] :
		Visited[currentX][currentY].append(0)
	if Visited[currentX][currentY] == [0] :
		Visited[currentX][currentY][-1] = 1


# print PositionChange
plt.figure(1)
# plt.xlabel("time 't' in seconds")
# plt.ylabel("voltage V in Volts")
# plt.title('Spiking Integrate-and-Fire Model')
# plt.axis([ 0, 1, -0.075, -0.035 ])
# plt.savefig("figure1.png", dpi=300, pad_inches=0.2)
for x in PositionChange :
	if x ==-1 :
		continue
	plt.plot(x[0], x[1], 'o')
plt.show()

print len(PositionChange)

plt.figure(2)
for i in range(Xsize) :
	for j in range(Ysize) :
		if Visited[i][j] == [0] :
			plt.plot(i, j, 'or')
		else :
			plt.plot(i, j, 'og')
plt.show()

# or give just order of squares as a list --- which shuare [ square(1,1), (1,2), etc.. ]

# check the position and remember whether mouse was already here if so check
# the change of firing rate

# split the maze into bricks

