MAKE = make
CC = gcc
CFLAGS = -O5 -Wall -march=native -std=c99 -fopenmp

all: graph-gen

graph-gen: graph-gen.c ffprng.h
	$(CC) $(CFLAGS) -o graph-gen graph-gen.c -lm

.PHONY: graph-gen
clean:
	rm -f *~ graph-gen
	rm -rf *.dSYM
	rm -f core*
