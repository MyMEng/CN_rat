#! /usr/bin/python


# import
from math import ceil
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np

import time


# Globals definitions
Resolution = 15
currentS=1


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

# check for duplicates
def unify(X, Y, Z) :
	A, B, C = [], [], []
	for x,y,z in zip(X, Y, Z) :
		if x in A and y in B :
			if A.index(x) == B.index(y) :
				C[A.index(x)] += z
			else :
				A.append(x)
				B.append(y)
				C.append(z)
		else :
			A.append(x)
			B.append(y)
			C.append(z)
	return (A, B, C)


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
		# Visited[i].append(0)
		TimeAtPosition[i].append([])

test3d = [ [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]] ]
AtPosition = []
PositionChange = [-1]
Accum = [0, 0, 0, 0]
FiringRate = [ [-1], [-1], [-1], [-1] ]
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
		for zc in range(4) :
			FiringRate[zc].append(Accum[zc])

			# Visited[ PositionChange[-1][0] ][ PositionChange[-1][1] ]+=Accum
			test3d[zc][0].append(PositionChange[-1][0])
			test3d[zc][1].append(PositionChange[-1][1])
			test3d[zc][2].append(Accum[zc])

			if T[i] in M[zc] :
				Accum[zc] = 1
			else :
				Accum[zc] = 0
		Move = False

	else :
		for zc in range(4) :
			if T[i] in M[zc] :
				Accum[zc] += 1



# print PositionChange
# plt.ion()
plt.figure(1)
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title('Discretized maze')
plt.axis([ -2, 22, -2, 16 ])
for i, x in enumerate(PositionChange) :
	if x ==-1 :
		continue
	plt.plot(x[0], x[1], 'ko')
	# plt.draw()
	# time.sleep(0.1)
	# if i%50 == 0 :
		# plt.clf()
		# plt.axis([ -2, 22, -2, 16 ])

plt.savefig("figure1.png", dpi=300, pad_inches=0.2)
plt.show()


# Print firing rate along position change
for i in range(2,6) :
	plt.figure(i)
	axe = range(len(FiringRate[i-2]))
	plt.plot(axe, FiringRate[i-2], 'g')
	plt.xlabel("Position change")
	plt.ylabel("Firing rate")
	plt.title('Firing rate along position change')
		# Visited1 = detrend(Visited)
		# lol1 = range(len(Visited1))
		# plt.plot(lol1, Visited1, 'g')
		# plt.plot(Visited)
	plt.savefig("figure"+str(i)+".png", dpi=300, pad_inches=0.2)
	plt.show()

# Generate 3d plots
for dd in range(6,10) :
	# create a plot
	hf = plt.figure(dd)
	ha = hf.add_subplot(111, projection='3d')

	test3d[dd-6][0], test3d[dd-6][1], test3d[dd-6][2] = unify( test3d[dd-6][0], test3d[dd-6][1], test3d[dd-6][2] )

	# transform (X,Y,Z) to mesh grid
	xi = np.linspace(min(test3d[dd-6][0]), max(test3d[dd-6][0]))
	yi = np.linspace(min(test3d[dd-6][1]), max(test3d[dd-6][1]))
	X, Y = np.meshgrid(xi, yi)
	Z = griddata(test3d[dd-6][0], test3d[dd-6][1], test3d[dd-6][2], xi, yi)

	# Get gradient to plot colours
	Gx, Gy = np.gradient(Z)
	G = ( Gx**2.0 + Gy**2.0 )**.5
	N = G/G.max()
	# N = Z/(Z.max()*1.0)

	# plot contour
	cset = ha.contour(X, Y, Z, zdir='z', offset=-5, cmap=cm.coolwarm)
	cset = ha.contour(X, Y, Z, zdir='x', offset=max(xi), cmap=cm.coolwarm)
	cset = ha.contour(X, Y, Z, zdir='y', offset=max(yi), cmap=cm.coolwarm)

	# Set axis range
	# ax.set_xlim3d(-pi, 2*pi);
	# ax.set_ylim3d(0, 3*pi);
	ha.set_zlim3d(-5, 18);

	# Print position scatter
	for x in PositionChange :
		if x ==-1 :
			continue
		ha.scatter(x[0], x[1], -5)

	# arrange angle and elevation
	ha.view_init(elev=27, azim=-114)

	# give names
	ha.set_xlabel("X coordinate")
	# plt.xlabel("X coordinate")
	ha.set_ylabel("Y coordinate")
	# plt.ylabel("Y coordinate")
	ha.set_zlabel("Firing rate")
	# plt.zlabel("Firing rate")
	plt.title('Firing rate at given position')

	# save before grid
	plt.savefig("figure"+str(dd)+"_a.png", dpi=300, pad_inches=0.2)

	p = ha.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25, facecolors=cm.jet(N), cmap=cm.Oranges, linewidth=0, antialiased=False)
	# cb = hf.colorbar(p, shrink=0.5)

	# save after grid
	plt.savefig("figure"+str(dd)+"_b.png", dpi=300, pad_inches=0.2)

	plt.show()
