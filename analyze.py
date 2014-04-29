#! /usr/bin/python


# import
from math import ceil
from pprint import pprint
import matplotlib.pyplot as plt


# Globals definitions
Resolution = 25


# Function definitions
# Round up to nearest 10
def round10( x ) :
	if x%Resolution == 0 :
		return x
	else :
		return int( ceil( x/(Resolution*1.0) ) * Resolution )

# take closes number to given one
def takeClosest( ls, t ) :
	return min( ls, key=lambda x:abs( x-t ))

# maximal difference between 2 consecutive numbers in a list
def maxDiff( L ) :
	maxDiff = 0
	for i in range(1,len(L)) :
		d = L[i]-L[i-1]
		if d > maxDiff :
			maxDiff = d
	return maxDiff

# detrend time series
def detrend( Visited ) :
	Vdt = []
	for i in range(1,len(Visited)) :
		Vdt.append(Visited[i] - Visited[i-1])
	return Vdt

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

# Get maze size
Xmin = round10( min( D[0] ) )
Xmax = round10( max( D[0] ) )
Ymin = round10( min( D[1] ) )
Ymax = round10( max( D[1] ) )

# Get number of cells and Divide into them
Visited = []
TimeAtPosition = []
Xsize = Xmax /Resolution
Ysize = Ymax /Resolution
for i in range(Xsize) :
	# Visited.append([])
	TimeAtPosition.append([])
	for j in range(Ysize) :
		# Visited[i].append([])
		TimeAtPosition[i].append([])

AtPosition = []
PositionChange = [-1]
Accum = 0
FiringRate = [-1]
Move = False
# put times at which the mouse was at given square
for i, (x, y) in enumerate( zip(D[0], D[1]) ) :
	currentX = int( x/Resolution )
	currentY = int( y/Resolution )
	currentT = T[i]


	TimeAtPosition[currentX][currentY].append( currentT )
	AtPosition.append( (currentX, currentY) )
	# if last position differ from current one add | otherwise skip
	if (PositionChange[-1] != (currentX, currentY) and
		PositionChange[-1] != (currentX+1, currentY) and
		PositionChange[-1] != (currentX-1, currentY) and
		PositionChange[-1] != (currentX, currentY+1) and
		PositionChange[-1] != (currentX, currentY-1) and
		PositionChange[-1] != (currentX-1, currentY-1) and
		PositionChange[-1] != (currentX-1, currentY+1) and
		PositionChange[-1] != (currentX+1, currentY-1) and
		PositionChange[-1] != (currentX+1, currentY+1)) :

		PositionChange.append( (currentX, currentY) )
		Move = True

	# check if visited and so put indicator make something
	if Move :
		# Just appended new one: check whether it already existed
		if PositionChange[-1] in PositionChange[:-1] :
			counter = PositionChange[:-1].count(PositionChange[-1])
			counter += 1
			Visited.append(counter)
		else :
			Visited.append(1)

	# calculate firing rate for current square
	if Move :
		FiringRate.append(Accum)
		Move = False
		if T[i] in M[2] :
			Accum = 1
	else :
		if T[i] in M[2] :
			Accum += 1



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

# plt.figure(2)
# for i in range(Xsize) :
# 	for j in range(Ysize) :
# 		if Visited[i][j] == [0] :
# 			plt.plot(i, j, 'or')
# 		else :
# 			plt.plot(i, j, 'og')
# plt.show()

plt.figure(3)
lol = range(len(FiringRate))
plt.plot(lol, FiringRate, 'r')
Visited1 = detrend(Visited)
lol1 = range(len(Visited1))
plt.plot(lol1, Visited1, 'g')
plt.show()


# or give just order of squares as a list ---
#  which shuare [ square(1,1), (1,2), etc.. ]

# check the position and remember whether mouse was already here if so check
# the change of firing rate

