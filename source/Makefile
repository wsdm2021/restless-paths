MAKE = make
CC = gcc
CFLAGS = -O5 -Wall -march=native -std=c99 -fopenmp

#TARGETS = LISTER_SIN_GENF1\
#		  LISTER_SIN_GENF2\
#		  LISTER_PAR_GENF1\
#		  LISTER_PAR_GENF2


TARGETS = LISTER_DEFAULT

INPUT = lister.c builds.h gf.h ffprng.h

all: $(TARGETS)

LISTER_DEFAULT: $(INPUT)
	$(CC) $(CFLAGS) -DLISTER_PAR_GENF2 -o LISTER_DEFAULT lister.c

LISTER_PAR_GENF2: $(INPUT)
	$(CC) $(CFLAGS) -DLISTER_PAR_GENF2 -o LISTER_PAR_GENF2 lister.c

LISTER_SIN_GENF2: $(INPUT)
	$(CC) $(CFLAGS) -DLISTER_SIN_GENF2 -o LISTER_SIN_GENF2 lister.c

clean:	
	rm -f *.o *.a *~
	rm -f $(TARGETS)
	rm -f core*
	rm -rf *.dSYM
	rm -f *.pdf
