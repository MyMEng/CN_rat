#! /usr/bin/python


# import
from math import ceil
from pprint import pprint
import matplotlib.pyplot as plt


# Function definitions
# Round up to nearest 10
def round10( x ) :
	if x%20 == 0 :
		return x
	else :
		return int( ceil( x/20.0 ) * 20 )

# take closes number to given one
def takeClosest( ls, t ) :
	return min( ls, key=lambda x:abs( x-t ))

# Load data
#  for loading neuron 1 - 4 and time
N = []
M = []
for n in range(1,5) :
	f = open("data/neuron"+str(n)+".csv")
	N.append( map( lambda x: float(x.strip()), f.readlines() ) )
	M.append([])
#  for loading x and y
D = []
for i in ['x', 'y'] :
	f = open("data/"+i+".csv")
	D.append( map( lambda x: float(x.strip()), f.readlines() ) )
#  for time
f = open("data/time.csv")
T = map( lambda x: float(x.strip()), f.readlines() )

# Start program
# modify neuron firing to time recordings T
for i, Nn in enumerate( N ) :
	for n in Nn :
		M[i].append( takeClosest( T, n ) )













# find max difference
maxDiff = 0
for i in range(1,len(T)) :
	d = T[i]-T[i-1]
	if d > maxDiff :
		maxDiff = d
print maxDiff

for i in N :
	maxDiff = 0
	for i in range(1,len(i)) :
		d = T[i]-T[i-1]
		if d > maxDiff :
			maxDiff = d
	print maxDiff

# find and map firing to position




# Get maze size
Xmin = round10( min( D[0] ) )
Xmax = round10( max( D[0] ) )
Ymin = round10( min( D[1] ) )
Ymax = round10( max( D[1] ) )

# Get number of cells and Divide into them
Visited = []
TimeAtPosition = []
Xsize = Xmax /20
Ysize = Ymax /20
for i in range(Xsize) :
	Visited.append([])
	TimeAtPosition.append([])
	for j in range(Ysize) :
		Visited[i].append([])
		TimeAtPosition[i].append([])

AtPosition = []
PositionChange = [-1]
Accum = 0
FiringRate = [-1]
Move = False
# put times at which the mouse was at given square
for i, (x, y) in enumerate( zip(D[0], D[1]) ) :
	currentX = int( x/20 )
	currentY = int( y/20 )
	currentT = T[i]


	if currentT in M[0] :
		print "Spike1"
	if currentT in M[1] :
		print "Spike2"
	if currentT in M[2] :
		print "Spike3"
	if currentT in M[3] :
		print "Spike4"

	TimeAtPosition[currentX][currentY].append( currentT )
	AtPosition.append( (currentX, currentY) )
	# if last position differ from current one add | otherwise skip
	if PositionChange[-1] != (currentX, currentY) and PositionChange[-1] != (currentX+1, currentY) and PositionChange[-1] != (currentX-1, currentY) and PositionChange[-1] != (currentX, currentY+1) and PositionChange[-1] != (currentX, currentY-1) and PositionChange[-1] != (currentX-1, currentY-1) and PositionChange[-1] != (currentX-1, currentY+1) and PositionChange[-1] != (currentX+1, currentY-1) and PositionChange[-1] != (currentX+1, currentY+1):
		PositionChange.append( (currentX, currentY) )
		Move = True
	# calculate firing rate for current square
	if Move :
		FiringRate.append(Accum)
		Move = False
		if T[i] in M[3] :
			Accum = 1
	else :
		if T[i] in M[3] :
			Accum += 1


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

plt.figure(3)
lol = range(len(FiringRate))
plt.plot(lol, FiringRate)
plt.show()


# or give just order of squares as a list --- which shuare [ square(1,1), (1,2), etc.. ]

# check the position and remember whether mouse was already here if so check
# the change of firing rate

# split the maze into bricks

