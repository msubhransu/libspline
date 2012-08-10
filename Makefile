# This is the Makefile under linux

CXX ?= g++
CFLAGS=-Wall -Wconversion -O3 -fPIC

all:splinedemo

splinedemo:splinedemo.cpp splinemodel.o
	$(CXX) $(CFLAGS) splinedemo.cpp splinemodel.o -o splinedemo

splinemodel.o: splinemodel.cpp splinemodel.h
	$(CXX) $(CFLAGS) -c splinemodel.cpp
clean:
	rm -f splinedemo *.o *~
backup:
	cp *cpp *h bkup